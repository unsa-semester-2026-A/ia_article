import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from src.training.trainers.train_base_1 import DRIVE_DESTINATIONS, Base1Trainer


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    """Provide a default configuration dictionary for Base1Trainer."""
    return {
        "output_dir": str(tmp_path / "runs"),
        "model_weights": "yolo26s-obb.pt",
        "labels_path": str(tmp_path / "yolo_obb_labels.zip"),
        "labels_zip_path": str(tmp_path / "yolo_obb_labels.zip"),
        "data_yaml_path": str(tmp_path / "dataset" / "smart_dataset.yaml"),
        "images_dir": str(tmp_path / "images_source"),
        "dataset_workspace": str(tmp_path / "dataset"),
        "drive_folder_id": "dummy_folder_id",
        "experiment_condition": "Base_1_Raw_Data",
    }


@pytest.fixture
@patch("src.training.trainers.train_base_1.IOManager")
def trainer(mock_io_manager: MagicMock, config: dict[str, Any]) -> Base1Trainer:
    """Provide a Base1Trainer instance with a mocked IOManager."""
    trainer = Base1Trainer(config)
    trainer.io_manager = mock_io_manager.return_value
    trainer.io_manager.drive_service = MagicMock()
    return trainer


# ===================================================================
# Black-Box Tests
# ===================================================================


def test_get_hyperparameters_defaults(trainer: Base1Trainer) -> None:
    """Test get_hyperparameters returns default values when no overrides exist."""
    params = trainer.get_hyperparameters()
    for key, value in Base1Trainer.DEFAULT_HYPERPARAMS.items():
        assert params[key] == value


def test_default_epoch_budget_matches_plan(trainer: Base1Trainer) -> None:
    """Test the epoch budget follows the pilot calibration of 06_training.md §2."""
    params = trainer.get_hyperparameters()
    assert params["epochs"] == 40
    assert params["patience"] == 5
    # RAM caching does not fit this dataset and degrades silently if requested.
    assert params["cache"] is False


def test_condition_defaults_to_c1(trainer: Base1Trainer) -> None:
    """Test the trainer defaults to the raw-data baseline condition."""
    assert trainer.condition == "c1"
    assert trainer.run_name == "f1_c1"


def test_every_condition_has_isolated_drive_destinations() -> None:
    """Production conditions never write checkpoints into each other's folders."""
    assert set(DRIVE_DESTINATIONS) == {"c1", "c2", "c3"}
    assert len({entry["results"] for entry in DRIVE_DESTINATIONS.values()}) == 3
    assert len({entry["checkpoints"] for entry in DRIVE_DESTINATIONS.values()}) == 3


def test_unknown_condition_is_rejected(config: dict[str, Any]) -> None:
    """Test an unsupported condition fails fast instead of training silently."""
    config["condition"] = "c9"
    with pytest.raises(ValueError, match="Unknown condition"):
        Base1Trainer(config)


@pytest.mark.parametrize(
    ("condition", "expect_mosaic"),
    [("c1", 0.0), ("c2", 1.0), ("c3", 0.0)],
)
@patch("src.training.trainers.train_base_1.IOManager")
def test_augmentation_profile_per_condition(
    mock_io_manager: MagicMock,
    config: dict[str, Any],
    condition: str,
    expect_mosaic: float,
) -> None:
    """Test only C2 enables object-combining augmentation."""
    config["condition"] = condition
    params = Base1Trainer(config).get_hyperparameters()
    assert params["mosaic"] == expect_mosaic


@patch("src.training.trainers.train_base_1.IOManager")
def test_c1_and_c3_share_every_hyperparameter(
    mock_io_manager: MagicMock, config: dict[str, Any]
) -> None:
    """Test C1 and C3 differ only in the dataset, which the intra-family gain requires."""
    c1_params = Base1Trainer({**config, "condition": "c1"}).get_hyperparameters()
    c3_params = Base1Trainer({**config, "condition": "c3"}).get_hyperparameters()
    assert c1_params == c3_params


@patch("src.training.trainers.train_base_1.IOManager")
def test_c3_reads_the_lama_images(
    mock_io_manager: MagicMock, config: dict[str, Any]
) -> None:
    """Test C3 points at the cleaned images while sharing the labels."""
    config["lama_images_dir"] = "/data/smart_lama_corrected/train"
    c1 = Base1Trainer({**config, "condition": "c1"}).get_dataset_config()
    c3 = Base1Trainer({**config, "condition": "c3"}).get_dataset_config()
    assert c1["images_dir"] == config["images_dir"]
    assert c3["images_dir"] == "/data/smart_lama_corrected/train"
    assert c1["labels_path"] == c3["labels_path"]


def test_get_hyperparameters_overrides(trainer: Base1Trainer) -> None:
    """Test get_hyperparameters applies allowed config overrides."""
    trainer.config["epochs"] = 50
    trainer.config["batch"] = 16
    params = trainer.get_hyperparameters()
    assert params["epochs"] == 50
    assert params["batch"] == 16


def test_get_dataset_config(trainer: Base1Trainer, config: dict[str, Any]) -> None:
    """Test get_dataset_config returns correct paths."""
    dataset_config = trainer.get_dataset_config()
    assert dataset_config["data_yaml_path"] == config["data_yaml_path"]
    assert dataset_config["model_weights"] == config["model_weights"]
    assert dataset_config["labels_path"] == config["labels_path"]
    assert dataset_config["images_dir"] == config["images_dir"]


def test_prepare_dataset_valid(trainer: Base1Trainer, tmp_path: Path) -> None:
    """Test prepare_dataset with valid zip and images dir."""
    labels_zip = Path(trainer.config["labels_path"])
    images_dir = Path(trainer.config["images_dir"])

    # Create dummy files
    labels_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(labels_zip, "w") as zf:
        zf.writestr("dummy.txt", "data")

    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "train").mkdir()
    (images_dir / "val").mkdir()

    yaml_path = trainer.prepare_dataset()

    assert yaml_path.name == "smart_dataset.yaml"
    workspace = Path(trainer.config["dataset_workspace"])
    assert (workspace / "labels").exists()
    assert (workspace / "images" / "train").exists()


def test_prepare_dataset_missing_zip(trainer: Base1Trainer) -> None:
    """Test prepare_dataset raises FileNotFoundError if zip is missing."""
    with pytest.raises(FileNotFoundError):
        trainer.prepare_dataset()


def test_prepare_dataset_pre_existing_workspace(
    trainer: Base1Trainer, tmp_path: Path
) -> None:
    """Test prepare_dataset skips if workspace is already populated."""
    yaml_path = Path(trainer.config["data_yaml_path"])
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.touch()

    labels_dir = yaml_path.parent / "labels"
    labels_dir.mkdir()
    (labels_dir / "dummy.txt").touch()

    images_dir = yaml_path.parent / "images"
    images_dir.mkdir()
    (images_dir / "dummy.jpg").touch()

    # If it tries to process zip, it will fail because zip doesn't exist
    # If it skips properly, it returns yaml_path
    result = trainer.prepare_dataset()
    assert result == yaml_path


def test_health_check_valid(trainer: Base1Trainer, tmp_path: Path) -> None:
    """Test health_check with valid zip, images dir, and Drive service."""
    labels_zip = Path(trainer.config["labels_path"])
    images_dir = Path(trainer.config["images_dir"])

    labels_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(labels_zip, "w") as zf:
        zf.writestr("dummy.txt", "data")

    images_dir.mkdir(parents=True, exist_ok=True)

    with patch(
        "src.training.base_training.torch.cuda.is_available", return_value=False
    ):
        health = trainer.health_check()

    assert health["passed"] is True
    assert health["details"]["labels"]["exists"] is True
    assert health["details"]["images_dir"]["exists"] is True
    assert health["details"]["drive_service"]["available"] is True


def test_health_check_corrupt_zip(trainer: Base1Trainer, tmp_path: Path) -> None:
    """Test health_check with a corrupt zip file."""
    labels_zip = Path(trainer.config["labels_path"])
    images_dir = Path(trainer.config["images_dir"])

    labels_zip.parent.mkdir(parents=True, exist_ok=True)
    labels_zip.write_text("not a zip file")
    images_dir.mkdir(parents=True, exist_ok=True)

    with patch(
        "src.training.base_training.torch.cuda.is_available", return_value=False
    ):
        health = trainer.health_check()

    assert health["passed"] is False
    assert health["details"]["labels"]["valid"] is False


def test_health_check_missing_images_dir(trainer: Base1Trainer) -> None:
    """Test health_check with missing images dir."""
    labels_zip = Path(trainer.config["labels_path"])
    labels_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(labels_zip, "w") as zf:
        zf.writestr("dummy.txt", "data")

    with patch(
        "src.training.base_training.torch.cuda.is_available", return_value=False
    ):
        health = trainer.health_check()

    assert health["passed"] is False
    assert health["details"]["images_dir"]["exists"] is False


# ===================================================================
# White-Box Tests
# ===================================================================


def test_get_hyperparameters_unallowed_override(trainer: Base1Trainer) -> None:
    """Test get_hyperparameters only accepts specific overrides."""
    trainer.config["mosaic"] = 1.0  # Not in override_keys
    params = trainer.get_hyperparameters()
    assert params["mosaic"] == 0.0  # Default value, not overridden


@patch("src.training.trainers.train_base_1.zipfile.ZipFile")
@patch("src.training.trainers.train_base_1.shutil.copytree")
@patch("src.training.trainers.train_base_1.Path.symlink_to")
def test_prepare_dataset_symlink_and_fallback(
    mock_symlink: MagicMock,
    mock_copytree: MagicMock,
    mock_zip: MagicMock,
    trainer: Base1Trainer,
    tmp_path: Path,
) -> None:
    """Test prepare_dataset calls extractall, symlink, and falls back to copytree on OSError."""
    labels_zip = Path(trainer.config["labels_zip_path"])
    images_dir = Path(trainer.config["images_dir"])

    labels_zip.parent.mkdir(parents=True, exist_ok=True)
    labels_zip.touch()

    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "train").mkdir()
    (images_dir / "val").mkdir()

    # Simulate OSError on the second symlink call (val)
    mock_symlink.side_effect = [None, OSError("Symlink not supported")]

    trainer.prepare_dataset()

    mock_zip.return_value.__enter__.return_value.extractall.assert_called_once()
    assert mock_symlink.call_count == 2
    assert mock_copytree.call_count == 1
    mock_copytree.assert_called_once_with(
        str(images_dir / "val"),
        str(Path(trainer.config["dataset_workspace"]) / "images" / "val"),
    )


def test_save_and_upload_error_branch(trainer: Base1Trainer, tmp_path: Path) -> None:
    """Test _save_and_upload handles Drive upload error."""
    local_path = tmp_path / "data.json"
    trainer.io_manager.upload_file_to_drive.side_effect = Exception("Upload failed")

    result = trainer._save_and_upload({"data": 1}, local_path, "folder_id")

    trainer.io_manager.save_json.assert_called_once_with({"data": 1}, local_path)
    assert result["local"] == str(local_path)
    assert result["drive_id"] is None


def test_upload_file_mime_types(trainer: Base1Trainer, tmp_path: Path) -> None:
    """Test _upload_file selects correct MIME types for different extensions."""
    test_files = [
        ("data.csv", "text/csv"),
        ("config.yaml", "text/yaml"),
        ("plot.png", "image/png"),
        ("model.pt", "application/octet-stream"),
        ("unknown.xyz", "application/octet-stream"),
    ]

    for fname, expected_mime in test_files:
        fpath = tmp_path / fname
        trainer._upload_file(fpath, "folder_id")
        trainer.io_manager.upload_file_to_drive.assert_called_with(
            fpath,
            "folder_id",
            mime_type=expected_mime,
            remote_name=f"{trainer.run_name}_{fname}",
        )


@patch("ultralytics.YOLO")
@patch.object(Base1Trainer, "detect_device")
@patch.object(Base1Trainer, "prepare_dataset")
@patch.object(Base1Trainer, "parse_results_csv")
def test_execute_success(
    mock_parse: MagicMock,
    mock_prepare: MagicMock,
    mock_detect_device: MagicMock,
    mock_yolo: MagicMock,
    trainer: Base1Trainer,
    tmp_path: Path,
) -> None:
    """Test execute method including results parsing and drive uploads."""
    mock_prepare.return_value = Path("dummy.yaml")
    mock_parse.return_value = [{"epoch": 1.0}]
    mock_detect_device.return_value = "0,1"

    # Setup YOLO mock to succeed on train
    mock_model = MagicMock()
    mock_model.train.return_value = None
    mock_yolo.return_value = mock_model

    # Create dummy output files
    train_dir = Path(trainer.config["output_dir"]) / trainer.run_name
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "results.csv").touch()
    (train_dir / "results.png").touch()
    (train_dir / "weights").mkdir()
    (train_dir / "weights" / "best.pt").touch()

    # Set drive IDs
    trainer.io_manager.upload_file_to_drive.return_value = "dummy_id"

    result = trainer.execute()

    # Assertions
    assert result["status"] == "success"
    assert "metrics" in result
    assert result["metrics"]["oom_fallback_triggered"] is False

    assert mock_model.train.call_count == 1
    mock_parse.assert_called_once()

    # Check uploads
    uploaded_files = [f["local"] for f in result["files"]]
    assert any("training_metrics.json" in f for f in uploaded_files)
    assert any("results.csv" in f for f in uploaded_files)
    assert any("results.png" in f for f in uploaded_files)
    assert any("best.pt" in f for f in uploaded_files)


@patch("src.training.trainers.train_base_1.IOManager")
def test_smoke_run_keeps_the_production_recipe(
    mock_io_manager: MagicMock, config: dict[str, Any]
) -> None:
    """Black Box: A smoke run may only shrink the workload, never the recipe."""
    real = Base1Trainer({**config, "condition": "c1"}).get_hyperparameters()
    smoke = Base1Trainer(
        {**config, "condition": "c1", "smoke_test": True}
    ).get_hyperparameters()

    # Only the amount of work changes
    assert smoke["epochs"] == 3
    assert smoke["save_period"] == 1
    assert smoke["batch"] < real["batch"]

    # Everything that defines the experiment stays identical
    for key in [
        "imgsz",
        "optimizer",
        "lr0",
        "lrf",
        "weight_decay",
        "amp",
        "seed",
        "mosaic",
        "mixup",
        "copy_paste",
        "degrees",
        "fliplr",
        "patience",
    ]:
        assert smoke[key] == real[key], f"smoke run altered '{key}'"


@patch("ultralytics.YOLO")
@patch.object(Base1Trainer, "detect_device")
@patch.object(Base1Trainer, "prepare_dataset")
@patch.object(Base1Trainer, "parse_results_csv")
@patch("src.training.trainers.train_base_1.IOManager")
def test_smoke_save_period_survives_the_config(
    mock_io_manager: MagicMock,
    mock_parse: MagicMock,
    mock_prepare: MagicMock,
    mock_detect_device: MagicMock,
    mock_yolo: MagicMock,
    config: dict[str, Any],
    tmp_path: Path,
) -> None:
    """White Box: The smoke cadence of 1 epoch must not be overridden by config."""
    mock_prepare.return_value = tmp_path / "data.yaml"
    mock_detect_device.return_value = "0,1"
    mock_parse.return_value = {"total_epochs_completed": 3}

    trainer = Base1Trainer({**config, "smoke_test": True, "save_period": 5})
    trainer.execute()

    assert mock_yolo.return_value.train.call_args.kwargs["save_period"] == 1


@patch("src.training.trainers.train_base_1.IOManager")
def test_smoke_batch_is_even_and_at_least_two(
    mock_io_manager: MagicMock, config: dict[str, Any]
) -> None:
    """White Box: The smoke batch must divide across 2 GPUs and allow >1 step."""
    for images in [2, 4, 10, 16, 25]:
        trainer = Base1Trainer({**config, "smoke_test": True, "smoke_images": images})
        batch = trainer.get_hyperparameters()["batch"]
        assert batch >= 2
        assert batch % 2 == 0
        assert batch <= images


@patch("src.training.trainers.train_base_1.IOManager")
def test_smoke_run_name_is_separate(
    mock_io_manager: MagicMock, config: dict[str, Any]
) -> None:
    """Black Box: Smoke artifacts must not collide with real results in Drive."""
    real = Base1Trainer({**config, "condition": "c1"})
    smoke = Base1Trainer({**config, "condition": "c1", "smoke_test": True})
    assert real.run_name == "f1_c1"
    assert smoke.run_name == "f1_c1_smoke"
    assert real.remote_name_for("last.pt") == "f1_c1_last.pt"
    assert smoke.remote_name_for("last.pt") == "f1_c1_smoke_last.pt"


@patch("src.training.trainers.train_base_1.IOManager")
def test_weights_and_results_target_different_folders(
    mock_io_manager: MagicMock, config: dict[str, Any]
) -> None:
    """Black Box: Checkpoints go to their own Drive folder when configured."""
    trainer = Base1Trainer({**config, "drive_checkpoints_folder_id": "ckpt_folder"})
    trainer.io_manager.drive_service = MagicMock()
    assert trainer.drive_results_folder_id == "dummy_folder_id"
    assert trainer.drive_checkpoints_folder_id == "ckpt_folder"


@patch("src.training.trainers.train_base_1.IOManager")
def test_checkpoints_fall_back_to_results_folder(
    mock_io_manager: MagicMock, config: dict[str, Any]
) -> None:
    """White Box: Without a dedicated folder, weights land beside the results."""
    trainer = Base1Trainer(config)
    trainer.io_manager.drive_service = MagicMock()
    assert trainer.drive_checkpoints_folder_id == "dummy_folder_id"


def test_link_split_images_is_deterministic_and_limited(
    trainer: Base1Trainer, tmp_path: Path
) -> None:
    """White Box: The subset is the sorted prefix of the labels, so it is stable."""
    labels = tmp_path / "labels"
    images = tmp_path / "images"
    dest = tmp_path / "dest"
    labels.mkdir()
    images.mkdir()
    stems = [f"v_clip_{i:04d}" for i in range(20)]
    for stem in stems:
        (labels / f"{stem}.txt").write_text("0 0 0 0 0 0 0 0 0")
        (images / f"{stem}.jpg").write_bytes(b"x")

    linked = trainer._link_split_images(
        labels_dir=labels,
        images_dir=images,
        split_subdir=None,
        destination=dest,
        limit=5,
    )

    assert linked == 5
    assert sorted(p.stem for p in dest.iterdir()) == stems[:5]


def test_link_split_images_without_limit_takes_all(
    trainer: Base1Trainer, tmp_path: Path
) -> None:
    """Black Box: A production run links every labelled image of the split."""
    labels = tmp_path / "labels"
    images = tmp_path / "images"
    dest = tmp_path / "dest"
    labels.mkdir()
    images.mkdir()
    for i in range(7):
        (labels / f"f{i}.txt").write_text("0 0 0 0 0 0 0 0 0")
        (images / f"f{i}.jpg").write_bytes(b"x")

    assert (
        trainer._link_split_images(
            labels_dir=labels,
            images_dir=images,
            split_subdir=None,
            destination=dest,
            limit=None,
        )
        == 7
    )


def test_link_split_images_skips_labels_without_image(
    trainer: Base1Trainer, tmp_path: Path
) -> None:
    """White Box: A label with no image is not counted, keeping the count honest."""
    labels = tmp_path / "labels"
    images = tmp_path / "images"
    dest = tmp_path / "dest"
    labels.mkdir()
    images.mkdir()
    (labels / "present.txt").write_text("0 0 0 0 0 0 0 0 0")
    (labels / "missing.txt").write_text("0 0 0 0 0 0 0 0 0")
    (images / "present.jpg").write_bytes(b"x")

    linked = trainer._link_split_images(
        labels_dir=labels,
        images_dir=images,
        split_subdir=None,
        destination=dest,
        limit=None,
    )

    assert linked == 1
    assert [p.stem for p in dest.iterdir()] == ["present"]


def test_prepare_dataset_uses_common_raw_lama_train_manifest(
    trainer: Base1Trainer, tmp_path: Path
) -> None:
    """C1/C2/C3 must train on only the frames present in both image variants."""
    labels = tmp_path / "labels"
    raw = tmp_path / "raw"
    lama = tmp_path / "lama"
    for split, stems in {"train": ["a", "b", "missing"], "val": ["v"]}.items():
        for stem in stems:
            path = labels / split / f"{stem}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("0 0 0 0 0 0 0 0 0")
    for stem in ["a", "b", "missing", "v"]:
        (raw / f"{stem}.jpg").parent.mkdir(parents=True, exist_ok=True)
        (raw / f"{stem}.jpg").write_bytes(b"raw")
    for stem in ["a", "b"]:
        (lama / f"{stem}.jpg").parent.mkdir(parents=True, exist_ok=True)
        (lama / f"{stem}.jpg").write_bytes(b"lama")

    trainer.config.update(
        {
            "labels_path": str(labels),
            "images_dir": str(raw),
            "lama_images_dir": str(lama),
            "data_yaml_path": str(tmp_path / "workspace" / "smart_dataset.yaml"),
            "dataset_workspace": str(tmp_path / "workspace"),
        }
    )
    trainer.prepare_dataset()

    workspace = Path(trainer.config["dataset_workspace"])
    assert (workspace / "common_train_stems.txt").read_text().splitlines() == ["a", "b"]
    assert sorted(p.stem for p in (workspace / "labels" / "train").iterdir()) == [
        "a",
        "b",
    ]
    assert sorted(p.stem for p in (workspace / "images" / "train").iterdir()) == [
        "a",
        "b",
    ]
    assert [p.stem for p in (workspace / "images" / "val").iterdir()] == ["v"]


def test_image_stems_accepts_case_insensitive_extensions(tmp_path: Path) -> None:
    """The Raw/LaMa manifest includes supported image extensions consistently."""
    (tmp_path / "a.JPG").write_bytes(b"image")
    (tmp_path / "b.png").write_bytes(b"image")
    (tmp_path / "notes.txt").write_text("not an image")

    assert Base1Trainer._image_stems(tmp_path) == {"a", "b"}


def test_report_gpu_usage_flags_single_gpu_fallback(trainer: Base1Trainer) -> None:
    """Black Box: Requesting two GPUs but engaging one must not pass silently."""
    trainer.config["expected_gpus"] = 2
    hardware = {
        "gpu_sampling": {
            "available": True,
            "gpus_engaged": 1,
            "devices": [
                {
                    "index": 0,
                    "name": "Tesla T4",
                    "memory_total_mib": 15360.0,
                    "peak_memory_used_mib": 9000.0,
                    "peak_utilization_pct": 98.0,
                    "mean_utilization_pct": 80.0,
                },
                {
                    "index": 1,
                    "name": "Tesla T4",
                    "memory_total_mib": 15360.0,
                    "peak_memory_used_mib": 3.0,
                    "peak_utilization_pct": 0.0,
                    "mean_utilization_pct": 0.0,
                },
            ],
        }
    }

    report = trainer.report_gpu_usage(hardware)

    assert report["multi_gpu_verified"] is False
    assert report["expected_gpus"] == 2
    assert report["gpus_engaged"] == 1


def test_report_gpu_usage_confirms_both_gpus(trainer: Base1Trainer) -> None:
    """Black Box: Two engaged GPUs satisfy the multi-GPU expectation."""
    trainer.config["expected_gpus"] = 2
    hardware = {
        "gpu_sampling": {
            "available": True,
            "gpus_engaged": 2,
            "devices": [
                {
                    "index": i,
                    "name": "Tesla T4",
                    "memory_total_mib": 15360.0,
                    "peak_memory_used_mib": 9000.0,
                    "peak_utilization_pct": 97.0,
                    "mean_utilization_pct": 82.0,
                }
                for i in range(2)
            ],
        }
    }

    assert trainer.report_gpu_usage(hardware)["multi_gpu_verified"] is True
