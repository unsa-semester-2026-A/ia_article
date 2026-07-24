"""Base 1 (Raw Data Baseline) trainer for YOLO26s-OBB with minimal augmentation."""

import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

from src.training.base_training import BaseTrainingPipeline
from src.utils.io_manager import IOManager


class Base1Trainer(BaseTrainingPipeline):
    """Training pipeline for Base 1 condition: raw data with minimal augmentation.

    Base 1 trains YOLO26s-OBB on the original unmodified SMART Challenge dataset
    without synthetic augmentations (Mosaic, MixUp, CopyPaste, Erasing disabled)
    to establish a baseline for the ablation study.
    """

    RUN_NAME = "base1"
    EXPERIMENT_CONDITION = "Base_1_Raw_Data"

    # Default hyperparameters for Base 1 (minimal augmentation)
    DEFAULT_HYPERPARAMS: dict[str, Any] = {
        "epochs": 100,
        "patience": 20,
        "batch": 96,  # DDP splits to 48/GPU on 2x Tesla T4 (~6-7 GB VRAM each, safe for OBB)
        "imgsz": 640,
        "cache": True,  # Load dataset into RAM to bypass 2-core CPU I/O bottleneck
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "weight_decay": 0.0005,
        "amp": True,
        "workers": 2,  # Match physical CPU cores to avoid context-switch saturation
        "seed": 42,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "degrees": 180.0,
        "fliplr": 0.5,
        "flipud": 0.5,
        "scale": 0.5,
        "translate": 0.1,
        "exist_ok": True,
        "plots": True,
        "val": True,
        "verbose": True,
    }

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Base1Trainer with configuration and IOManager.

        Args:
            config: Configuration dictionary containing:
                - output_dir: Local directory for training outputs.
                - model_weights: Path to pretrained model weights.
                - labels_zip_path: Path to yolo_obb_labels.zip.
                - data_yaml_path: Path to smart_dataset.yaml.
                - images_dir: Path to images root directory.
                - dataset_workspace: Path to scratch workspace for dataset.
                - drive_folder_id: Google Drive folder ID for sync (optional).
                - token_path: Path to Drive token.json (optional).
                - save_period: Checkpoint save frequency in epochs (default: 10).
                - experiment_condition: Ablation condition name.
                - hardware_name: Hardware platform description.
        """
        super().__init__(config)
        self.io_manager = IOManager(token_path=config.get("token_path"))

    # ===================================================================
    # Abstract Method Implementations
    # ===================================================================

    def get_hyperparameters(self) -> dict[str, Any]:
        """Return Base 1 hyperparameters with minimal augmentation disabled.

        Returns:
            Dictionary of YOLO training arguments for Base 1 condition.
        """
        params = dict(self.DEFAULT_HYPERPARAMS)

        # Apply config overrides for hyperparameters (allows CLI/notebook tuning)
        override_keys = [
            "epochs",
            "patience",
            "batch",
            "imgsz",
            "workers",
            "seed",
            "fraction",
        ]
        for key in override_keys:
            if key in self.config:
                params[key] = self.config[key]

        # Fast dev run (smoke test mode for rapid validation)
        if self.config.get("fast_dev_run", False):
            params["epochs"] = 1
            params["fraction"] = 0.01  # Use only 1% of data (~430 images)
            params["save_period"] = 1
            print(
                "[Base1Trainer] ⚡ Fast Dev Run active: 1 epoch on 1% dataset.",
                flush=True,
            )

        return params

    def get_dataset_config(self) -> dict[str, Any]:
        """Return Base 1 dataset paths and configuration.

        Returns:
            Dictionary with paths to dataset YAML, model weights, labels,
            images directory, and optional resized images zip.
        """
        labels_src = self.config.get("labels_path") or self.config.get(
            "labels_zip_path", ""
        )
        return {
            "data_yaml_path": self.config.get("data_yaml_path", ""),
            "model_weights": self.config.get("model_weights", "yolo26s-obb.pt"),
            "labels_path": labels_src,
            "images_dir": self.config.get("images_dir", ""),
            "resized_zip_path": self.config.get("resized_zip_path", ""),
        }

    def _prepare_drive_run_folder(
        self, output_dir: Path, drive_root_folder_id: str | None
    ) -> str:
        """Create and verify the isolated Drive folder for this training run.

        A successful round trip for a tiny manifest is required before GPU work
        starts. This prevents a long training job from finishing with weights
        that only exist in an ephemeral Kaggle or Colab filesystem.

        Args:
            output_dir: Local output directory where the probe is written.
            drive_root_folder_id: Parent folder ID configured outside Git.

        Returns:
            Verified Drive child-folder ID for this condition.

        Raises:
            RuntimeError: If the root ID is missing or Drive cannot persist a
                small verification artifact.
        """
        if not drive_root_folder_id:
            raise RuntimeError(
                "TRAINING_DRIVE_ROOT_FOLDER_ID is required for training artifact sync."
            )

        run_folder_id = self.io_manager.get_or_create_drive_folder(
            drive_root_folder_id, self.RUN_NAME
        )
        probe_path = output_dir / f"{self.RUN_NAME}_drive_preflight.json"
        self.io_manager.save_json(
            {
                "experiment_condition": self.config.get(
                    "experiment_condition", self.EXPERIMENT_CONDITION
                ),
                "run_name": self.RUN_NAME,
                "purpose": "verify Drive write, metadata, and checksum before training",
            },
            probe_path,
        )
        self.io_manager.upload_and_verify_file(
            probe_path, run_folder_id, mime_type="application/json"
        )
        print(
            f"[Base1Trainer] ✅ Drive preflight verified for {self.RUN_NAME}.",
            flush=True,
        )
        return run_folder_id

    def verify_drive_persistence(self) -> str:
        """Verify Drive write access without loading data or allocating a GPU.

        This is the mandatory first Kaggle action for a new condition. It creates
        the condition folder and performs the verified JSON round trip used by
        the training flow, then returns without importing Ultralytics.

        Returns:
            Verified Drive folder ID dedicated to this experiment condition.
        """
        output_dir = Path(self.config.get("output_dir", "/kaggle/working/runs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        return self._prepare_drive_run_folder(
            output_dir, self.config.get("drive_folder_id")
        )

    def prepare_dataset(self) -> Path:
        """Prepare Base 1 dataset workspace: copy/unzip labels, symlink images.

        Performs the following steps:
            1. Create dataset workspace directory structure.
            2. Extract or symlink yolo_obb_labels into workspace/labels/.
            3. Symlink or copy images into workspace/images/.
            4. Generate smart_dataset.yaml if not already present.

        Returns:
            Path to the dataset YAML file ready for training.

        Raises:
            FileNotFoundError: If labels or images directory is missing.
        """
        dataset_config = self.get_dataset_config()
        workspace = Path(self.config.get("dataset_workspace", "/tmp/dataset"))

        labels_source = Path(dataset_config["labels_path"])
        images_dir = Path(dataset_config["images_dir"])
        data_yaml = Path(dataset_config["data_yaml_path"])

        # Skip preparation if YAML already exists and workspace is fully populated
        if data_yaml.exists():
            labels_dir = workspace / "labels"
            images_dest = workspace / "images"
            if (
                labels_dir.exists()
                and any(labels_dir.iterdir())
                and images_dest.exists()
                and any(images_dest.iterdir())
            ):
                return data_yaml

        # Validate source files exist
        if labels_source.name and not labels_source.exists():
            raise FileNotFoundError(f"Labels source not found: {labels_source}")

        if images_dir.name and not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Create workspace structure
        workspace.mkdir(parents=True, exist_ok=True)
        labels_dest = workspace / "labels"
        images_dest = workspace / "images"

        # Handle labels source (directory vs zip file)
        if labels_source.name and labels_source.exists():
            labels_dest.mkdir(parents=True, exist_ok=True)
            if labels_source.is_dir():
                for split in ["train", "val"]:
                    src = labels_source / split
                    dst = labels_dest / split
                    if src.exists() and not dst.exists():
                        try:
                            dst.symlink_to(src)
                        except OSError:
                            shutil.copytree(str(src), str(dst))
            elif labels_source.is_file() and labels_source.suffix == ".zip":
                with zipfile.ZipFile(labels_source, "r") as zf:
                    zf.extractall(labels_dest)

        # --- Resolve image source: prefer train_resized.zip if available ---
        resized_zip = Path(dataset_config.get("resized_zip_path", ""))
        resized_extract_dir = workspace / "_resized_images"

        if resized_zip.name and resized_zip.exists() and resized_zip.suffix == ".zip":
            # Extract resized images to /tmp workspace (avoids /kaggle/working quota)
            if not resized_extract_dir.exists() or not any(
                resized_extract_dir.iterdir()
            ):
                print(
                    f"[Base1Trainer] 📦 Extracting resized images: {resized_zip.name}",
                    flush=True,
                )
                resized_extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(resized_zip, "r") as zf:
                    zf.extractall(resized_extract_dir)
                print(
                    f"[Base1Trainer] ✅ Resized images extracted to {resized_extract_dir}",
                    flush=True,
                )
            # Use the extracted directory as effective images source
            images_dir = resized_extract_dir
            print(
                "[Base1Trainer] Using resized images for training.",
                flush=True,
            )

        # Populate images for train and val splits
        if images_dir.name and images_dir.exists():
            images_dest.mkdir(parents=True, exist_ok=True)

            # Check if images_dir has train/ and val/ subdirectories directly
            has_split_subdirs = (images_dir / "train").exists() or (
                images_dir / "val"
            ).exists()

            if has_split_subdirs:
                for split in ["train", "val"]:
                    src = images_dir / split
                    dst = images_dest / split
                    if src.exists() and not dst.exists():
                        try:
                            dst.symlink_to(src)
                        except OSError:
                            shutil.copytree(str(src), str(dst))
            else:
                # Flat images directory: link images corresponding to labels in each split
                print(
                    "[Base1Trainer] Linking flat images directory to train/val splits...",
                    flush=True,
                )
                for split in ["train", "val"]:
                    split_img_dest = images_dest / split
                    split_img_dest.mkdir(parents=True, exist_ok=True)

                    split_labels_dir = labels_dest / split
                    if split_labels_dir.exists():
                        for txt_path in split_labels_dir.glob("*.txt"):
                            stem = txt_path.stem
                            # Try common image extensions
                            for ext in [
                                ".jpg",
                                ".png",
                                ".jpeg",
                                ".JPG",
                                ".PNG",
                            ]:
                                img_src = images_dir / f"{stem}{ext}"
                                if img_src.exists():
                                    img_dst = split_img_dest / f"{stem}{ext}"
                                    if not img_dst.exists():
                                        try:
                                            img_dst.symlink_to(img_src)
                                        except OSError:
                                            shutil.copy2(str(img_src), str(img_dst))
                                    break

        # Generate smart_dataset.yaml if not already present
        if not data_yaml.exists():
            data_yaml.parent.mkdir(parents=True, exist_ok=True)
            yaml_content = f"""# SMART Challenge 2026 Dataset Configuration
path: {workspace.resolve()}
train: images/train
val: images/val

names:
  0: auto
  1: combi
  2: microbus
  3: minibus
  4: omnibus
  5: articulado
  6: camion
  7: mototaxi
  8: motocicleta
"""
            data_yaml.write_text(yaml_content, encoding="utf-8")
            print(f"[Base1Trainer] ✅ Generated smart_dataset.yaml at {data_yaml}")

        return data_yaml

    # ===================================================================
    # Health Check (Extended)
    # ===================================================================

    def health_check(self) -> dict[str, Any]:
        """Run comprehensive pre-flight validation for Base 1 training.

        Extends base health check with dataset-specific validations:
            1. GPU availability and count.
            2. Output directory writability.
            3. Labels existence (directory or zip) and integrity.
            4. Images directory existence.
            5. Dataset YAML existence (or readiness for generation).
            6. Google Drive service availability.

        Returns:
            Dictionary with check results and overall pass/fail status.
        """
        checks = super().health_check()
        dataset_config = self.get_dataset_config()

        # Labels check (directory or zip)
        labels_path = Path(dataset_config.get("labels_path", ""))
        if labels_path.name:
            if labels_path.exists():
                if labels_path.is_dir():
                    txt_count = len(list(labels_path.rglob("*.txt")))
                    checks["details"]["labels"] = {
                        "path": str(labels_path),
                        "type": "directory",
                        "exists": True,
                        "txt_file_count": txt_count,
                    }
                elif labels_path.suffix == ".zip":
                    try:
                        with zipfile.ZipFile(labels_path, "r") as zf:
                            file_count = len(zf.namelist())
                        checks["details"]["labels"] = {
                            "path": str(labels_path),
                            "type": "zip",
                            "exists": True,
                            "file_count": file_count,
                        }
                    except zipfile.BadZipFile:
                        checks["details"]["labels"] = {
                            "path": str(labels_path),
                            "type": "zip",
                            "exists": True,
                            "valid": False,
                            "error": "Corrupt zip file",
                        }
                        checks["passed"] = False
            else:
                checks["details"]["labels"] = {
                    "path": str(labels_path),
                    "exists": False,
                }
                checks["passed"] = False

        # Images directory check
        images_dir = Path(dataset_config["images_dir"])
        if images_dir.name:
            checks["details"]["images_dir"] = {
                "path": str(images_dir),
                "exists": images_dir.exists(),
            }
            if not images_dir.exists():
                checks["passed"] = False

        # Data YAML check (can exist or will be generated during prepare_dataset)
        data_yaml = Path(dataset_config["data_yaml_path"])
        checks["details"]["data_yaml"] = {
            "path": str(data_yaml),
            "exists": data_yaml.exists(),
            "status": "Will be generated automatically"
            if not data_yaml.exists()
            else "Ready",
        }

        # Drive service check (crucial for experiment artifact preservation)
        drive_available = self.io_manager.drive_service is not None
        checks["details"]["drive_service"] = {
            "available": drive_available,
        }
        if self.config.get("drive_folder_id") and not drive_available:
            checks["details"]["drive_service"]["error"] = (
                "drive_folder_id configured but Google Drive service is unavailable (Auth/token failed)."
            )
            checks["passed"] = False

        if not self.config.get("drive_folder_id"):
            checks["details"]["drive_root_folder"] = {
                "configured": False,
                "error": "Set TRAINING_DRIVE_ROOT_FOLDER_ID before training.",
            }
            checks["passed"] = False
        else:
            checks["details"]["drive_root_folder"] = {"configured": True}

        return checks

    # ===================================================================
    # Save & Upload Helper
    # ===================================================================

    def _save_and_upload(
        self, data: Any, local_path: Path, drive_folder_id: str | None
    ) -> dict[str, Any]:
        """Save data as JSON locally and optionally upload to Google Drive.

        Args:
            data: Content to serialize as JSON.
            local_path: Local file path for saving.
            drive_folder_id: Google Drive folder ID for upload (optional).

        Returns:
            Dictionary with 'local' path and 'drive_id' (or None).
        """
        self.io_manager.save_json(data, local_path)
        drive_id = None
        if drive_folder_id:
            try:
                drive_id = self.io_manager.upload_and_verify_file(
                    local_path, drive_folder_id, mime_type="application/json"
                )
            except Exception as e:
                print(
                    f"  ⚠️  Drive upload error for {local_path.name}: {e}",
                    flush=True,
                )
        return {"local": str(local_path), "drive_id": drive_id}

    def _upload_file(
        self, local_path: Path, drive_folder_id: str | None, *, strict: bool = False
    ) -> str | None:
        """Upload an existing file to Google Drive.

        Args:
            local_path: Local file path to upload.
            drive_folder_id: Google Drive folder ID for upload.
            strict: Re-raise a sync failure instead of only logging it. This is
                used for checkpoints, whose loss would make a training run
                unrecoverable.

        Returns:
            Google Drive file ID, or None on failure.
        """
        if not drive_folder_id or not self.io_manager.drive_service:
            return None
        try:
            mime = "application/octet-stream"
            if local_path.suffix == ".csv":
                mime = "text/csv"
            elif local_path.suffix == ".yaml":
                mime = "text/yaml"
            elif local_path.suffix == ".png":
                mime = "image/png"
            return self.io_manager.upload_and_verify_file(
                local_path, drive_folder_id, mime_type=mime
            )
        except Exception as e:
            print(
                f"  ⚠️  Drive upload error for {local_path.name}: {e}",
                flush=True,
            )
            if strict:
                raise RuntimeError(
                    f"Required Drive sync failed for {local_path.name}."
                ) from e
            return None

    # ===================================================================
    # Main Execution
    # ===================================================================

    def execute(self) -> dict[str, Any]:
        """Execute the complete Base 1 training pipeline.

        Workflow:
            1. Run health check and prepare dataset.
            2. Initialize YOLO model with pretrained weights.
            3. Train with OOM fallback strategy.
            4. Parse results.csv and record hardware metrics.
            5. Upload artifacts to Google Drive.

        Returns:
            Dictionary with training results, metrics, and generated file paths.
        """
        output_dir = Path(self.config.get("output_dir", "/kaggle/working/runs"))
        drive_folder_id = self.verify_drive_persistence()
        from ultralytics import YOLO

        self.start_hardware_monitoring()
        save_period: int = self.config.get("save_period", 10)

        generated_files: list[dict[str, Any]] = []

        # 1. Prepare dataset
        data_yaml = self.prepare_dataset()
        dataset_config = self.get_dataset_config()

        # 2. Build training arguments
        hyperparams = self.get_hyperparameters()
        device = self.detect_device()
        hyperparams["device"] = device
        hyperparams["data"] = str(data_yaml)
        hyperparams["project"] = str(output_dir)
        hyperparams["name"] = self.RUN_NAME
        hyperparams["save_period"] = save_period

        # 3. Check for existing last.pt checkpoint in Drive or local workspace for resume
        train_dir = output_dir / self.RUN_NAME
        local_last_pt = train_dir / "weights" / "last.pt"
        resumed_checkpoint: Path | None = None

        if drive_folder_id and self.io_manager.drive_service:
            print(
                "[Base1Trainer] Checking for existing last.pt checkpoint in Google Drive...",
                flush=True,
            )
            try:
                resumed_checkpoint = self.io_manager.download_file_from_drive(
                    file_name="last.pt",
                    drive_folder_id=drive_folder_id,
                    local_destination_path=local_last_pt,
                )
            except Exception as e:
                print(
                    f"[Base1Trainer] Checkpoint download check skipped: {e}",
                    flush=True,
                )

        if not resumed_checkpoint and local_last_pt.exists():
            resumed_checkpoint = local_last_pt

        if resumed_checkpoint and resumed_checkpoint.exists():
            print(
                f"[Base1Trainer] 🔄 Resuming training from checkpoint: {resumed_checkpoint}",
                flush=True,
            )
            model = YOLO(str(resumed_checkpoint))
            hyperparams["resume"] = True
        else:
            print(
                f"[Base1Trainer] 🚀 Starting fresh training from: {dataset_config['model_weights']}",
                flush=True,
            )
            model = YOLO(dataset_config["model_weights"])
            hyperparams["resume"] = False

        total_epochs = hyperparams.get("epochs", 100)

        def sync_checkpoint_callback(trainer_obj: Any) -> None:
            """Synchronize every checkpoint event and propagate any upload failure."""
            current_epoch = getattr(trainer_obj, "epoch", 0) + 1
            save_dir = Path(getattr(trainer_obj, "save_dir", train_dir))

            artifact_files = [
                "results.csv",
                "results.png",
                "args.yaml",
                "confusion_matrix.png",
                "PR_curve.png",
            ]
            for fname in artifact_files:
                fpath = save_dir / fname
                if fpath.exists():
                    self._upload_file(fpath, drive_folder_id, strict=True)

            weights_dir = save_dir / "weights"
            # Ultralytics writes ``last.pt`` every epoch, ``best.pt`` on
            # improvement, and ``epoch*.pt`` at ``save_period`` intervals.
            # Sync every checkpoint that exists, so periodic recovery points
            # are preserved in Drive as well as the rolling last/best files.
            for wpath in sorted(weights_dir.glob("*.pt")):
                self._upload_file(wpath, drive_folder_id, strict=True)

            print(
                f"  📤 Verified Drive sync (epoch {current_epoch}/{total_epochs})",
                flush=True,
            )

        model.add_callback("on_model_save", sync_checkpoint_callback)

        # 4. Train (static batch=96 splits to 48/GPU via DDP, safe on 2x T4)
        _ = model.train(**hyperparams)

        # 5. Locate training output directory
        train_dir = output_dir / self.RUN_NAME
        if not train_dir.exists():
            # Ultralytics may create base1, base12, etc.
            candidates = sorted(output_dir.glob(f"{self.RUN_NAME}*"))
            train_dir = candidates[-1] if candidates else output_dir

        # 6. Parse results.csv
        results_csv = train_dir / "results.csv"
        epochs_data: list[dict[str, Any]] = []
        if results_csv.exists():
            epochs_data = self.parse_results_csv(results_csv)

        # 7. Record hardware metrics
        hardware_metrics = self.record_hardware_metrics()

        # 8. Build training_metrics.json
        training_metrics = self.build_training_metrics(
            epochs_data=epochs_data,
            hardware_metrics=hardware_metrics,
            hyperparameters=hyperparams,
        )

        metrics_path = train_dir / "training_metrics.json"
        metrics_record = self._save_and_upload(
            training_metrics, metrics_path, drive_folder_id
        )
        generated_files.append(metrics_record)

        # 9. Upload key artifacts to Drive
        artifact_files = [
            "results.csv",
            "results.png",
            "args.yaml",
            "confusion_matrix.png",  # Recommended for paper
            "PR_curve.png",  # Recommended for paper
        ]
        for fname in artifact_files:
            fpath = train_dir / fname
            if fpath.exists():
                drive_id = self._upload_file(fpath, drive_folder_id)
                generated_files.append({"local": str(fpath), "drive_id": drive_id})

        # Upload every YOLO checkpoint, including periodic epoch*.pt files.
        weights_dir = train_dir / "weights"
        for wpath in sorted(weights_dir.glob("*.pt")):
            drive_id = self._upload_file(wpath, drive_folder_id)
            generated_files.append({"local": str(wpath), "drive_id": drive_id})

        print(
            f"[Base1Trainer] Training complete. {len(epochs_data)} epochs logged.",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()

        return {
            "status": "success",
            "train_dir": str(train_dir),
            "files": generated_files,
            "metrics": training_metrics,
        }


# ===================================================================
# CLI Entrypoint for Execution via python -m src.training.trainers.train_base_1
# ===================================================================
def _find_dataset_dir() -> Path:
    """Automatically discover dataset path in Kaggle or Colab environments."""
    if not os.path.exists("/kaggle/working"):
        return Path("/content/drive/MyDrive/ia_article")

    candidates = [
        Path("/kaggle/input/datasets/alvaroquispeunsa/mtc-challenge"),
        Path("/kaggle/input/mtc-challenge"),
    ]
    for c in candidates:
        if c.exists():
            return c

    return Path("/kaggle/input/datasets/alvaroquispeunsa/mtc-challenge")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Base 1 YOLO OBB Trainer")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run 1 epoch on 1% dataset for rapid smoke testing",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override total epochs"
    )
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--fraction", type=float, default=None, help="Dataset fraction (0.0 to 1.0)"
    )
    parser.add_argument(
        "--drive-preflight-only",
        action="store_true",
        help="Verify Drive write/checksum and exit before dataset or GPU use",
    )
    args = parser.parse_args()

    IS_KAGGLE = os.path.exists("/kaggle/working")

    dataset_dir = _find_dataset_dir()

    # Resolve labels path: unzipped directory vs zip file
    labels_path = dataset_dir / "yolo_obb_labels"
    if not labels_path.exists():
        labels_path = dataset_dir / "yolo_obb_labels.zip"

    # Resolve resized images zip (same level as train-001/)
    resized_zip = dataset_dir / "train_resized.zip"
    if not resized_zip.exists():
        resized_zip = Path("")  # Empty signals no resized zip available

    # Use /tmp for dataset workspace on Kaggle to avoid exceeding 20GB output disk limit
    dataset_workspace = Path("/tmp/dataset") if IS_KAGGLE else Path("/content/dataset")
    data_yaml_path = dataset_workspace / "smart_dataset.yaml"

    output_dir = Path("/kaggle/working/runs") if IS_KAGGLE else Path("/content/runs")

    config = {
        "output_dir": str(output_dir),
        "model_weights": "yolo26s-obb.pt",
        "labels_path": str(labels_path),
        "data_yaml_path": str(data_yaml_path),
        "images_dir": str(dataset_dir / "train-001" / "train"),
        "resized_zip_path": str(resized_zip),
        "dataset_workspace": str(
            Path("/tmp/dataset") if IS_KAGGLE else Path("/content/dataset")
        ),
        "save_period": 10,
        "hardware_name": "Tesla_T4x2_Kaggle" if IS_KAGGLE else "Colab_GPU",
        "experiment_condition": "Base_1_Raw_Data",
        "token_path": os.environ.get(
            "DRIVE_TOKEN_PATH",
            str(
                Path("/kaggle/working/token.json")
                if IS_KAGGLE
                else Path("/content/drive/MyDrive/ia_article/token/token.json")
            ),
        ),
        "drive_folder_id": os.environ.get("TRAINING_DRIVE_ROOT_FOLDER_ID"),
        "fast_dev_run": args.fast_dev_run,
    }

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch is not None:
        config["batch"] = args.batch
    if args.fraction is not None:
        config["fraction"] = args.fraction

    trainer = Base1Trainer(config=config)

    if args.drive_preflight_only:
        drive_run_folder = trainer.verify_drive_persistence()
        print(f"[SUCCESS] Drive preflight verified. Run folder: {drive_run_folder}")
        sys.exit(0)

    print("=" * 60)
    print("BASE 1 TRAINING - HEALTH CHECK")
    print("=" * 60)

    health = trainer.health_check()
    for section, detail in health["details"].items():
        print(f"  {section}: {detail}")
    print(f"  Overall: {'PASSED ✅' if health['passed'] else 'FAILED ❌'}")
    print("=" * 60)

    if not health["passed"]:
        print("[FATAL] Health check failed. Aborting.")
        sys.exit(1)

    results = trainer.execute()

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Status: {results['status']}")
    print(f"Train dir: {results['train_dir']}")
    print(f"Generated files: {len(results['files'])}")
    for f in results["files"]:
        status = f"Drive: {f['drive_id']}" if f["drive_id"] else "Local only"
        print(f"  📄 {Path(f['local']).name} -> {status}")
