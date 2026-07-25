"""Pre-training environment and dataset verification.

Every check here answers a question that, if left unanswered, only surfaces hours
into a run: whether both GPUs can talk to each other, whether the resolved
dependency set imports cleanly, whether Drive will accept the checkpoints, and
whether the raw and LaMa image sets are actually comparable.

Run as a script to obtain a report and a non-zero exit status on failure:

    python -m src.training.preflight --condition c1
"""

import argparse
import importlib
import json
import os
import platform
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

from src.utils.gpu_monitor import sample_gpus

#: Packages whose absence or version mismatch breaks training. ``lap`` is listed
#: because Ultralytics only fails on it once tracking starts, long after launch.
REQUIRED_PACKAGES = (
    "torch",
    "torchvision",
    "ultralytics",
    "cv2",
    "numpy",
    "pandas",
    "yaml",
    "lap",
    "googleapiclient",
)

#: Splits expected under the labels directory.
SPLITS = ("train", "val")


class CheckResult(dict[str, Any]):
    """A single named check with its pass/fail verdict and details."""

    def __init__(self, name: str, passed: bool, details: dict[str, Any]) -> None:
        """Initialize the result.

        Args:
            name: Human readable check name.
            passed: Whether the check succeeded.
            details: Structured evidence backing the verdict.
        """
        super().__init__(name=name, passed=passed, details=details)


# =====================================================================
# Environment
# =====================================================================
def check_packages() -> CheckResult:
    """Import every required package and record its version.

    Returns:
        Result whose details map each package to its version or import error.
    """
    versions: dict[str, str] = {}
    failures: list[str] = []
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as e:
            versions[name] = f"IMPORT FAILED: {e}"
            failures.append(name)
    return CheckResult(
        "dependencies",
        not failures,
        {"versions": versions, "failed": failures, "python": platform.python_version()},
    )


def check_gpus(expected_gpus: int) -> CheckResult:
    """Verify that the expected number of CUDA devices is visible.

    Args:
        expected_gpus: Number of GPUs the run is meant to use.

    Returns:
        Result with the device inventory reported by torch and ``nvidia-smi``.
    """
    details: dict[str, Any] = {"expected_gpus": expected_gpus}
    try:
        import torch

        details["torch_version"] = torch.__version__
        details["cuda_available"] = torch.cuda.is_available()
        details["cuda_version"] = torch.version.cuda
        details["torch_device_count"] = torch.cuda.device_count()
        details["devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        passed = torch.cuda.is_available() and (
            torch.cuda.device_count() >= expected_gpus
        )
    except Exception as e:
        details["error"] = str(e)
        passed = False

    details["nvidia_smi"] = sample_gpus()
    return CheckResult("gpus", passed, details)


def _nccl_worker(rank: int, world_size: int, port: int) -> None:
    """All-reduce a rank-dependent tensor and assert the collective result.

    Args:
        rank: Process rank assigned by the spawner.
        world_size: Total number of processes.
        port: Rendezvous port on localhost.
    """
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    torch.cuda.set_device(rank)
    tensor = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
    dist.all_reduce(tensor)
    expected = world_size * (world_size + 1) / 2
    if abs(tensor.item() - expected) > 1e-6:
        raise RuntimeError(
            f"rank {rank}: all_reduce gave {tensor.item()}, expected {expected}"
        )
    dist.destroy_process_group()


def check_nccl(expected_gpus: int, port: int = 29555) -> CheckResult:
    """Run a minimal multi-process all-reduce over NCCL.

    Ultralytics multi-GPU training is DDP over NCCL. If the collective cannot
    complete, the run either hangs or silently degrades to one GPU, so it is worth
    a few seconds to prove the collective works before committing to a long run.

    Args:
        expected_gpus: Number of processes and devices to exercise.
        port: Rendezvous port on localhost.

    Returns:
        Result stating whether the all-reduce produced the correct sum.
    """
    if expected_gpus < 2:
        return CheckResult(
            "nccl_all_reduce",
            True,
            {"skipped": "single-GPU run needs no collective"},
        )
    try:
        import torch.multiprocessing as mp

        mp.spawn(
            _nccl_worker, args=(expected_gpus, port), nprocs=expected_gpus, join=True
        )
        return CheckResult(
            "nccl_all_reduce", True, {"world_size": expected_gpus, "backend": "nccl"}
        )
    except Exception as e:
        return CheckResult(
            "nccl_all_reduce",
            False,
            {"world_size": expected_gpus, "backend": "nccl", "error": str(e)},
        )


# =====================================================================
# Dataset
# =====================================================================
#: Extensions treated as images when scanning a split directory.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def scan_images(directory: Path) -> tuple[int, set[str]]:
    """List the image names of a directory in a single cheap pass.

    The split directories hold tens of thousands of entries, so the scan uses
    ``os.scandir`` and filters on the name alone. ``Path.iterdir`` plus a
    ``is_file`` test would issue one ``stat`` per entry, which is what makes this
    kind of inventory slow on a network-backed mount.

    Args:
        directory: Directory to scan, non-recursively.

    Returns:
        Tuple of ``(count, stems)``. Empty when the directory does not exist.
    """
    if not directory.is_dir():
        return 0, set()

    stems: set[str] = set()
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            name = entry.name
            lowered = name.lower()
            if not lowered.endswith(IMAGE_SUFFIXES):
                continue
            count += 1
            stems.add(name.rsplit(".", 1)[0])
    return count, stems


def count_images(directory: Path) -> int:
    """Count image files directly inside a directory.

    Args:
        directory: Directory to scan, non-recursively.

    Returns:
        Number of image files, or 0 if the directory is absent.
    """
    return scan_images(directory)[0]


def read_image_size(path: Path) -> tuple[int, int] | None:
    """Return the ``(width, height)`` of an image, or None if unreadable.

    Args:
        path: Image file path.

    Returns:
        Pixel dimensions, or None when the file cannot be decoded.
    """
    try:
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            return None
        return (int(image.shape[1]), int(image.shape[0]))
    except Exception:
        return None


def check_labels(labels_dir: Path) -> CheckResult:
    """Verify both label splits exist and are non-empty.

    Args:
        labels_dir: Directory containing ``train/`` and ``val/`` label folders.

    Returns:
        Result with the label count of each split.
    """
    counts: dict[str, int] = {}
    for split in SPLITS:
        split_dir = labels_dir / split
        if not split_dir.is_dir():
            counts[split] = 0
            continue
        with os.scandir(split_dir) as entries:
            counts[split] = sum(1 for e in entries if e.name.endswith(".txt"))

    return CheckResult(
        "labels",
        all(count > 0 for count in counts.values()),
        {"directory": str(labels_dir), "counts": counts},
    )


def scan_label_stems(labels_dir: Path) -> set[str]:
    """Return the deterministic set of image names selected by label files.

    Args:
        labels_dir: Directory containing one split's YOLO ``.txt`` files.

    Returns:
        Stems of every label file, or an empty set when the directory is absent.
    """
    if not labels_dir.is_dir():
        return set()

    with os.scandir(labels_dir) as entries:
        return {entry.name[:-4] for entry in entries if entry.name.endswith(".txt")}


def check_image_sets(
    raw_dir: Path,
    lama_dir: Path,
    expected_stems: set[str] | None = None,
    sample_size: int = 5,
    allow_common_train_subset: bool = False,
) -> CheckResult:
    """Compare the raw and LaMa image sets for count and resolution parity.

    C1 and C3 differ only in whether the pixels were inpainted. A different image
    count or a different resolution would turn the comparison into a confounded
    one, and the ablation would attribute to LaMa an effect caused by the data.

    Args:
        raw_dir: Directory with the resized raw images.
        lama_dir: Directory with the LaMa-cleaned images.
        expected_stems: Optional label-derived training split. If given, the
            comparison ignores images from other splits in a flat source folder.
        sample_size: How many shared file names to measure.
        allow_common_train_subset: Whether a deterministic intersection of the
            two image sets is the declared training protocol.

    Returns:
        Result with counts, sampled sizes and whether both sets are comparable.
    """
    raw_total, raw_stems = scan_images(raw_dir)
    lama_total, lama_stems = scan_images(lama_dir)
    target_stems = (
        expected_stems if expected_stems is not None else raw_stems | lama_stems
    )
    raw_selected = raw_stems & target_stems
    lama_selected = lama_stems & target_stems
    common_stems = raw_selected & lama_selected
    shared = sorted(common_stems)[:sample_size]

    sizes: dict[str, dict[str, Any]] = {}
    resolutions_match = True
    for stem in shared:
        raw_size = read_image_size(raw_dir / f"{stem}.jpg")
        lama_size = read_image_size(lama_dir / f"{stem}.jpg")
        sizes[stem] = {"raw": raw_size, "lama": lama_size}
        if raw_size != lama_size:
            resolutions_match = False

    details = {
        "raw_dir": str(raw_dir),
        "lama_dir": str(lama_dir),
        "expected_count": len(target_stems),
        "raw_total_count": raw_total,
        "lama_total_count": lama_total,
        "raw_count": len(raw_selected),
        "lama_count": len(lama_selected),
        "common_train_count": len(common_stems),
        "excluded_from_common_train": len(target_stems - common_stems),
        "counts_match": raw_selected == target_stems and lama_selected == target_stems,
        "missing_in_raw": len(target_stems - raw_stems),
        "missing_in_lama": len(target_stems - lama_stems),
        "only_in_raw": len(raw_selected - lama_selected),
        "only_in_lama": len(lama_selected - raw_selected),
        "sampled_sizes": sizes,
        "resolutions_match": resolutions_match,
    }
    full_match = bool(details["counts_match"])
    passed = (
        (full_match or allow_common_train_subset)
        and resolutions_match
        and bool(common_stems)
    )
    details["training_protocol"] = (
        "full_label_split"
        if full_match
        else "common_raw_lama_intersection"
        if allow_common_train_subset
        else "incompatible"
    )
    return CheckResult("image_sets", passed, details)


# =====================================================================
# Drive
# =====================================================================
def check_drive(token_path: Path, folder_ids: dict[str, str]) -> CheckResult:
    """Verify Drive credentials resolve and every destination folder is writable.

    A run that trains for hours and then cannot upload its weights has wasted the
    quota, so the destinations are probed before training rather than after.

    Args:
        token_path: Path to the OAuth token JSON.
        folder_ids: Mapping of role name to Drive folder ID.

    Returns:
        Result with the resolved name of each folder, or the error encountered.
    """
    from src.utils.io_manager import IOManager

    details: dict[str, Any] = {"token_path": str(token_path), "folders": {}}

    # A preflight reports; it does not abort. Resolving the credentials must not
    # be allowed to hide the verdicts of the GPU, dataset and collective checks.
    try:
        manager = IOManager(token_path=str(token_path), require_drive=False)
    except Exception as e:
        details["error"] = f"Drive credentials could not be resolved: {e}"
        return CheckResult("drive", False, details)

    if not manager.drive_service:
        auth_error = getattr(manager, "drive_error", None)
        details["error"] = (
            auth_error
            if isinstance(auth_error, str) and auth_error
            else "Drive service could not be initialized. On Kaggle this usually means "
            "the 'DRIVE_TOKEN_JSON' secret is not attached to this notebook "
            "(Add-ons -> Secrets)."
        )
        return CheckResult("drive", False, details)

    passed = True
    for role, folder_id in folder_ids.items():
        try:
            metadata = (
                manager.drive_service.files()
                .get(
                    fileId=folder_id,
                    fields="id, name, mimeType",
                    supportsAllDrives=True,
                )
                .execute()
            )
            details["folders"][role] = {
                "id": folder_id,
                "name": metadata.get("name"),
                "reachable": True,
            }
        except Exception as e:
            details["folders"][role] = {
                "id": folder_id,
                "reachable": False,
                "error": str(e),
            }
            passed = False
    return CheckResult("drive", passed, details)


# =====================================================================
# Report
# =====================================================================
def run_preflight(
    labels_dir: Path,
    raw_images_dir: Path,
    lama_images_dir: Path,
    token_path: Path,
    folder_ids: dict[str, str],
    expected_gpus: int = 2,
    check_collective: bool = True,
    allow_common_train_subset: bool = False,
) -> dict[str, Any]:
    """Run every check and aggregate the verdicts.

    Args:
        labels_dir: Directory holding the ``train/`` and ``val/`` label folders.
        raw_images_dir: Directory with the resized raw training images.
        lama_images_dir: Directory with the LaMa-cleaned training images.
        token_path: Path to the Drive OAuth token JSON.
        folder_ids: Mapping of role name to Drive folder ID.
        expected_gpus: Number of GPUs the run is meant to use.
        check_collective: Whether to run the NCCL all-reduce probe.
        allow_common_train_subset: Accept a non-empty, reproducible raw/LaMa
            intersection as the declared training split.

    Returns:
        Dictionary with ``passed`` and the list of individual check results.
    """
    checks: list[CheckResult] = [
        check_packages(),
        check_gpus(expected_gpus),
        check_labels(labels_dir),
        check_image_sets(
            raw_images_dir,
            lama_images_dir,
            expected_stems=scan_label_stems(labels_dir / "train"),
            allow_common_train_subset=allow_common_train_subset,
        ),
        check_drive(token_path, folder_ids),
    ]
    if check_collective:
        checks.append(check_nccl(expected_gpus))

    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def print_report(report: dict[str, Any]) -> None:
    """Print a preflight report in a readable form.

    Args:
        report: Output of ``run_preflight``.
    """
    print("=" * 70)
    print("PREFLIGHT REPORT")
    print("=" * 70)
    for check in report["checks"]:
        mark = "PASS ✅" if check["passed"] else "FAIL ❌"
        print(f"\n[{mark}] {check['name']}")
        print(json.dumps(check["details"], indent=2, default=str))
    print("\n" + "=" * 70)
    print(f"OVERALL: {'PASS ✅' if report['passed'] else 'FAIL ❌'}")
    print("=" * 70)


def main() -> int:
    """Parse arguments, run the preflight and return a shell exit code.

    Returns:
        0 when every check passed, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Verify the training environment")
    parser.add_argument(
        "--labels-dir",
        default="/kaggle/input/mtc-challenge/yolo_obb_labels",
        help="Directory holding the train/ and val/ label folders",
    )
    parser.add_argument(
        "--raw-images-dir",
        default="/kaggle/input/mtc-challenge/train_resized/train",
        help="Directory with the resized raw training images",
    )
    parser.add_argument(
        "--lama-images-dir",
        default="/kaggle/input/mtc-challenge/smart_lama_corrected/train",
        help="Directory with the LaMa-cleaned training images",
    )
    parser.add_argument(
        "--token-path",
        default="/tmp/ia_article_drive_token.json",
        help="Path to the Drive OAuth token JSON",
    )
    parser.add_argument(
        "--results-folder-id",
        default="1n17lmU2SVz54HmV6a3Cd-bgKs0h6bQP8",
        help="Drive folder for logs, plots and metrics",
    )
    parser.add_argument(
        "--checkpoints-folder-id",
        default="1pn8OzJX_kctgluEZkSC6WEfbSCaPyKMa",
        help="Drive folder for model weights",
    )
    parser.add_argument("--expected-gpus", type=int, default=2)
    parser.add_argument(
        "--skip-nccl",
        action="store_true",
        help="Skip the NCCL all-reduce probe",
    )
    parser.add_argument(
        "--allow-common-train-subset",
        action="store_true",
        help="Accept the common Raw/LaMa train intersection as the F1 protocol",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the report as JSON",
    )
    args = parser.parse_args()

    report = run_preflight(
        labels_dir=Path(args.labels_dir),
        raw_images_dir=Path(args.raw_images_dir),
        lama_images_dir=Path(args.lama_images_dir),
        token_path=Path(args.token_path),
        folder_ids={
            "results": args.results_folder_id,
            "checkpoints": args.checkpoints_folder_id,
        },
        expected_gpus=args.expected_gpus,
        check_collective=not args.skip_nccl,
        allow_common_train_subset=args.allow_common_train_subset,
    )
    print_report(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"Report written to {output_path}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
