"""Unit tests for the pre-training verification checks."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from src.training.preflight import (
    check_drive,
    check_gpus,
    check_image_sets,
    check_labels,
    check_nccl,
    check_packages,
    count_images,
    read_image_size,
    run_preflight,
    scan_images,
)


def _write_image(path: Path, width: int, height: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))


# ==========================================
# Dependency Checks
# ==========================================
def test_check_packages_records_versions():
    """Black Box: Report a version for every importable required package."""
    result = check_packages()
    assert "torch" in result["details"]["versions"]
    assert "python" in result["details"]


def test_check_packages_fails_on_missing_import():
    """White Box: A failed import marks the check as failed and names the package."""
    with patch(
        "src.training.preflight.importlib.import_module",
        side_effect=ImportError("no module"),
    ):
        result = check_packages()

    assert result["passed"] is False
    assert result["details"]["failed"]


# ==========================================
# GPU Checks
# ==========================================
def test_check_gpus_fails_when_fewer_than_expected():
    """Black Box: One visible GPU cannot satisfy a two-GPU run."""
    with patch("src.training.preflight.sample_gpus", return_value=[]):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                result = check_gpus(expected_gpus=2)

    assert result["passed"] is False
    assert result["details"]["torch_device_count"] == 1


def test_check_gpus_passes_with_two_devices():
    """Black Box: Two visible devices satisfy a two-GPU run."""
    with patch("src.training.preflight.sample_gpus", return_value=[]):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=2):
                with patch("torch.cuda.get_device_name", return_value="Tesla T4"):
                    result = check_gpus(expected_gpus=2)

    assert result["passed"] is True
    assert result["details"]["devices"] == ["Tesla T4", "Tesla T4"]


def test_check_gpus_survives_absent_cuda():
    """White Box: A CUDA-less environment fails the check without raising."""
    with patch("src.training.preflight.sample_gpus", return_value=[]):
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.cuda.device_count", return_value=0):
                result = check_gpus(expected_gpus=2)

    assert result["passed"] is False


def test_check_nccl_skipped_for_single_gpu():
    """White Box: A single-GPU run has no collective to verify."""
    result = check_nccl(expected_gpus=1)
    assert result["passed"] is True
    assert "skipped" in result["details"]


def test_check_nccl_passes_when_spawn_succeeds():
    """Black Box: A completed all-reduce marks the collective as working."""
    with patch("torch.multiprocessing.spawn") as mock_spawn:
        result = check_nccl(expected_gpus=2)

    mock_spawn.assert_called_once()
    assert result["passed"] is True
    assert result["details"]["world_size"] == 2


def test_check_nccl_reports_failure_reason():
    """White Box: A hanging or broken collective is reported, not raised."""
    with patch("torch.multiprocessing.spawn", side_effect=RuntimeError("NCCL timeout")):
        result = check_nccl(expected_gpus=2)

    assert result["passed"] is False
    assert "NCCL timeout" in result["details"]["error"]


# ==========================================
# Dataset Checks
# ==========================================
def test_count_images_ignores_other_files(tmp_path: Path):
    """White Box: Only image extensions are counted."""
    _write_image(tmp_path / "a.jpg", 10, 10)
    _write_image(tmp_path / "b.png", 10, 10)
    (tmp_path / "notes.txt").write_text("x")
    assert count_images(tmp_path) == 2


def test_count_images_on_missing_directory(tmp_path: Path):
    """Black Box: A missing directory counts as zero rather than raising."""
    assert count_images(tmp_path / "nope") == 0


def test_scan_images_returns_count_and_stems(tmp_path: Path):
    """White Box: One pass yields both the count and the stems for comparison."""
    _write_image(tmp_path / "v_a_0001.jpg", 8, 8)
    _write_image(tmp_path / "v_a_0002.JPG", 8, 8)
    (tmp_path / "labels.txt").write_text("x")
    (tmp_path / "subdir").mkdir()

    count, stems = scan_images(tmp_path)

    assert count == 2
    assert stems == {"v_a_0001", "v_a_0002"}


def test_scan_images_on_missing_directory(tmp_path: Path):
    """Black Box: A missing directory yields an empty inventory."""
    assert scan_images(tmp_path / "nope") == (0, set())


def test_read_image_size(tmp_path: Path):
    """Black Box: Report width and height in pixel order."""
    _write_image(tmp_path / "img.jpg", 640, 360)
    assert read_image_size(tmp_path / "img.jpg") == (640, 360)


def test_read_image_size_on_unreadable_file(tmp_path: Path):
    """White Box: A non-image file yields None instead of an exception."""
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    assert read_image_size(tmp_path / "broken.jpg") is None


def test_check_labels_requires_both_splits(tmp_path: Path):
    """Black Box: A missing val split fails the check."""
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "a.txt").write_text("0 0 0 0 0 0 0 0 0")
    result = check_labels(tmp_path)
    assert result["passed"] is False
    assert result["details"]["counts"] == {"train": 1, "val": 0}


def test_check_labels_passes_with_both_splits(tmp_path: Path):
    """Black Box: Both splits populated passes and reports the counts."""
    for split, n in [("train", 3), ("val", 2)]:
        (tmp_path / split).mkdir()
        for i in range(n):
            (tmp_path / split / f"f{i}.txt").write_text("0 0 0 0 0 0 0 0 0")

    result = check_labels(tmp_path)
    assert result["passed"] is True
    assert result["details"]["counts"] == {"train": 3, "val": 2}


def test_check_image_sets_passes_when_comparable(tmp_path: Path):
    """Black Box: Same names and same resolution make C1 and C3 comparable."""
    raw, lama = tmp_path / "raw", tmp_path / "lama"
    for stem in ["v_a_0001", "v_a_0002"]:
        _write_image(raw / f"{stem}.jpg", 640, 360)
        _write_image(lama / f"{stem}.jpg", 640, 360)

    result = check_image_sets(raw, lama)
    assert result["passed"] is True
    assert result["details"]["counts_match"] is True
    assert result["details"]["resolutions_match"] is True


def test_check_image_sets_detects_resolution_mismatch(tmp_path: Path):
    """Black Box: A LaMa set left at full resolution confounds the ablation."""
    raw, lama = tmp_path / "raw", tmp_path / "lama"
    _write_image(raw / "v_a_0001.jpg", 640, 360)
    _write_image(lama / "v_a_0001.jpg", 1920, 1080)

    result = check_image_sets(raw, lama)
    assert result["passed"] is False
    assert result["details"]["resolutions_match"] is False


def test_check_image_sets_detects_count_mismatch(tmp_path: Path):
    """Black Box: LaMa applied to only part of the set must be flagged."""
    raw, lama = tmp_path / "raw", tmp_path / "lama"
    _write_image(raw / "v_a_0001.jpg", 640, 360)
    _write_image(raw / "v_a_0002.jpg", 640, 360)
    _write_image(lama / "v_a_0001.jpg", 640, 360)

    result = check_image_sets(raw, lama)
    assert result["passed"] is False
    assert result["details"]["counts_match"] is False
    assert result["details"]["only_in_raw"] == 1


# ==========================================
# Drive Checks
# ==========================================
def test_check_drive_fails_without_service(tmp_path: Path):
    """Black Box: Absent credentials fail before any upload is attempted."""
    with patch("src.utils.io_manager.IOManager") as mock_manager:
        mock_manager.return_value.drive_service = None
        result = check_drive(tmp_path / "token.json", {"results": "abc"})

    assert result["passed"] is False
    assert "error" in result["details"]


def test_check_drive_reports_credential_error_without_raising(tmp_path: Path):
    """White Box: A preflight reports; it must never abort the remaining checks."""
    with patch(
        "src.utils.io_manager.IOManager",
        side_effect=RuntimeError("secret not attached"),
    ):
        result = check_drive(tmp_path / "token.json", {"results": "abc"})

    assert result["passed"] is False
    assert "secret not attached" in result["details"]["error"]


def test_check_drive_probes_every_folder(tmp_path: Path):
    """White Box: Each configured destination is resolved by ID."""
    mock_service = MagicMock()
    mock_service.files().get().execute.return_value = {
        "id": "abc",
        "name": "ia_article",
    }
    with patch("src.utils.io_manager.IOManager") as mock_manager:
        mock_manager.return_value.drive_service = mock_service
        result = check_drive(
            tmp_path / "token.json",
            {"results": "abc", "checkpoints": "def"},
        )

    assert result["passed"] is True
    assert set(result["details"]["folders"]) == {"results", "checkpoints"}


def test_check_drive_reports_unreachable_folder(tmp_path: Path):
    """Black Box: An unreachable checkpoints folder fails the preflight."""
    mock_service = MagicMock()
    mock_service.files().get().execute.side_effect = RuntimeError("404 not found")
    with patch("src.utils.io_manager.IOManager") as mock_manager:
        mock_manager.return_value.drive_service = mock_service
        result = check_drive(tmp_path / "token.json", {"checkpoints": "bad_id"})

    assert result["passed"] is False
    assert result["details"]["folders"]["checkpoints"]["reachable"] is False


# ==========================================
# Aggregation
# ==========================================
def test_run_preflight_aggregates_and_fails_fast_on_any_check(tmp_path: Path):
    """Black Box: The overall verdict is the conjunction of every check."""
    with patch("src.training.preflight.check_packages") as packages:
        with patch("src.training.preflight.check_gpus") as gpus:
            with patch("src.training.preflight.check_labels") as labels:
                with patch("src.training.preflight.check_image_sets") as images:
                    with patch("src.training.preflight.check_drive") as drive:
                        packages.return_value = {
                            "name": "p",
                            "passed": True,
                            "details": {},
                        }
                        gpus.return_value = {"name": "g", "passed": True, "details": {}}
                        labels.return_value = {
                            "name": "l",
                            "passed": True,
                            "details": {},
                        }
                        images.return_value = {
                            "name": "i",
                            "passed": False,
                            "details": {},
                        }
                        drive.return_value = {
                            "name": "d",
                            "passed": True,
                            "details": {},
                        }

                        report = run_preflight(
                            labels_dir=tmp_path / "labels",
                            raw_images_dir=tmp_path / "raw",
                            lama_images_dir=tmp_path / "lama",
                            token_path=tmp_path / "token.json",
                            folder_ids={"results": "abc"},
                            expected_gpus=2,
                            check_collective=False,
                        )

    assert report["passed"] is False
    assert len(report["checks"]) == 5


def test_run_preflight_includes_collective_when_requested(tmp_path: Path):
    """White Box: The NCCL probe is part of the report unless skipped."""
    with patch("src.training.preflight.check_packages") as packages:
        with patch("src.training.preflight.check_gpus") as gpus:
            with patch("src.training.preflight.check_labels") as labels:
                with patch("src.training.preflight.check_image_sets") as images:
                    with patch("src.training.preflight.check_drive") as drive:
                        with patch("src.training.preflight.check_nccl") as nccl:
                            for mock in (packages, gpus, labels, images, drive, nccl):
                                mock.return_value = {
                                    "name": "x",
                                    "passed": True,
                                    "details": {},
                                }
                            report = run_preflight(
                                labels_dir=tmp_path / "labels",
                                raw_images_dir=tmp_path / "raw",
                                lama_images_dir=tmp_path / "lama",
                                token_path=tmp_path / "token.json",
                                folder_ids={"results": "abc"},
                                expected_gpus=2,
                                check_collective=True,
                            )

    assert report["passed"] is True
    assert len(report["checks"]) == 6
