"""Base 2 trainer: raw SMART data with controlled classic augmentation."""

import os
import sys
from pathlib import Path
from typing import Any

from src.training.trainers.train_base_1 import Base1Trainer


class Base2Trainer(Base1Trainer):
    """Train Base 2 with the same pipeline as Base 1 and classic augmentation.

    Base 2 deliberately reuses Base 1's data preparation, checkpoint-resume,
    Drive preflight, artifact verification, hardware monitoring, and reporting.
    Its only experimental difference is the documented classic augmentation
    configuration.
    """

    RUN_NAME = "base2"
    EXPERIMENT_CONDITION = "Base_2_Classic_Augmentation"
    DEFAULT_HYPERPARAMS: dict[str, Any] = {
        **Base1Trainer.DEFAULT_HYPERPARAMS,
        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.3,
        "erasing": 0.4,
        "close_mosaic": 10,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
    }


def _find_dataset_dir() -> Path:
    """Discover the mounted dataset path in Kaggle or Colab."""
    if not os.path.exists("/kaggle/working"):
        return Path("/content/drive/MyDrive/ia_article")

    candidates = [
        Path("/kaggle/input/datasets/alvaroquispeunsa/mtc-challenge"),
        Path("/kaggle/input/mtc-challenge"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Base 2 classic augmentation trainer")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    args = parser.parse_args()

    is_kaggle = os.path.exists("/kaggle/working")
    dataset_dir = _find_dataset_dir()
    labels_path = dataset_dir / "yolo_obb_labels"
    if not labels_path.exists():
        labels_path = dataset_dir / "yolo_obb_labels.zip"

    resized_zip = dataset_dir / "train_resized.zip"
    if not resized_zip.exists():
        resized_zip = Path("")

    dataset_workspace = Path("/tmp/dataset") if is_kaggle else Path("/content/dataset")
    output_dir = Path("/kaggle/working/runs") if is_kaggle else Path("/content/runs")
    default_token_path = (
        Path("/kaggle/working/token.json")
        if is_kaggle
        else Path("/content/drive/MyDrive/ia_article/token/token.json")
    )

    config: dict[str, Any] = {
        "output_dir": str(output_dir),
        "model_weights": "yolo26s-obb.pt",
        "labels_path": str(labels_path),
        "data_yaml_path": str(dataset_workspace / "smart_dataset.yaml"),
        "images_dir": str(dataset_dir / "train-001" / "train"),
        "resized_zip_path": str(resized_zip),
        "dataset_workspace": str(dataset_workspace),
        "save_period": 10,
        "hardware_name": "Tesla_T4x2_Kaggle" if is_kaggle else "Colab_GPU",
        "experiment_condition": Base2Trainer.EXPERIMENT_CONDITION,
        "token_path": os.environ.get("DRIVE_TOKEN_PATH", str(default_token_path)),
        "drive_folder_id": os.environ.get("TRAINING_DRIVE_ROOT_FOLDER_ID"),
        "fast_dev_run": args.fast_dev_run,
    }
    for key in ("epochs", "batch", "fraction"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    trainer = Base2Trainer(config)
    health = trainer.health_check()
    for section, detail in health["details"].items():
        print(f"{section}: {detail}")
    if not health["passed"]:
        print("[FATAL] Base 2 health check failed.")
        sys.exit(1)

    result = trainer.execute()
    print(f"Base 2 completed: {result['train_dir']}")
