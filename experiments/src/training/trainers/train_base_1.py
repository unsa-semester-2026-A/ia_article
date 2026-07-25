"""YOLO26s-OBB trainer for the F1 conditions of the ablation study (C1, C2, C3)."""

import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import torch
from src.training.base_training import BaseTrainingPipeline
from src.utils.io_manager import IOManager


def _is_primary_process() -> bool:
    """Report whether this process should perform Drive I/O.

    Ultralytics runs training callbacks inside every DDP worker. Without this
    guard each worker uploads the same checkpoint concurrently, which duplicates
    files and triggers conflicting writes against the Drive API.
    """
    return int(os.environ.get("RANK", -1)) in (-1, 0)


class Base1Trainer(BaseTrainingPipeline):
    """Training pipeline for the F1 family (YOLO26s-OBB) of the ablation study.

    Handles the three F1 conditions of ``06_training.md`` §4, selected with the
    ``condition`` config key:

    - ``c1``: raw data, minimal augmentation. Baseline, denominator of the
      intra-family gain.
    - ``c2``: raw data, classic YOLO augmentation (mosaic, mixup, copy-paste).
    - ``c3``: LaMa-cleaned data, minimal augmentation. Numerator of the gain.

    C1 and C3 must differ **only** in the pixel content of the training images,
    so their hyperparameters are shared by construction and only the augmentation
    profile of C2 departs from the baseline.
    """

    #: Hyperparameters shared by every F1 condition. Epoch budget and patience are
    #: calibrated on the pilot run documented in ``06_training.md`` §2: the model
    #: converged by epoch 6 and the following 33 epochs added ~0.01 mAP.
    DEFAULT_HYPERPARAMS: dict[str, Any] = {
        "epochs": 40,
        "patience": 5,
        "batch": 96,  # DDP splits to 48/GPU on 2x Tesla T4; pilot measured 13.6/15 GB
        "imgsz": 640,
        # RAM caching is not viable here: the pilot logged "41.9GB RAM required to
        # cache images [...] only 26.5/31.3GB available, not caching" and still
        # sustained 51 img/s reading from disk.
        "cache": False,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "weight_decay": 0.0005,
        "amp": True,
        "workers": 4,
        "seed": 42,
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

    #: Augmentation profile per condition. Everything that combines or pastes
    #: objects stays off in C1/C3 so the only variable between them is the dataset.
    CONDITION_PROFILES: dict[str, dict[str, Any]] = {
        "c1": {
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.3,
            "hsv_v": 0.2,
        },
        "c2": {
            "mosaic": 1.0,
            "mixup": 0.15,
            "copy_paste": 0.3,
            "erasing": 0.4,
            "close_mosaic": 10,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
        },
        "c3": {
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.3,
            "hsv_v": 0.2,
        },
    }

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the trainer with configuration and IOManager.

        Args:
            config: Configuration dictionary containing:
                - condition: Ablation condition, one of 'c1', 'c2', 'c3' (default 'c1').
                - output_dir: Local directory for training outputs.
                - model_weights: Path to pretrained model weights.
                - labels_zip_path: Path to yolo_obb_labels.zip.
                - data_yaml_path: Path to smart_dataset.yaml.
                - images_dir: Path to images root directory (raw variant).
                - lama_images_dir: Path to the LaMa-cleaned images, required by 'c3'.
                - dataset_workspace: Path to scratch workspace for dataset.
                - drive_folder_id: Google Drive folder ID for sync (optional).
                - token_path: Path to Drive token.json (optional).
                - save_period: Checkpoint save frequency in epochs (default: 5).
                - hardware_name: Hardware platform description.

        Raises:
            ValueError: If ``condition`` is not a known F1 condition.
        """
        super().__init__(config)
        condition = str(config.get("condition", "c1")).lower()
        if condition not in self.CONDITION_PROFILES:
            raise ValueError(
                f"Unknown condition '{condition}'. "
                f"Expected one of {sorted(self.CONDITION_PROFILES)}."
            )
        self.condition = condition
        self.smoke_test = bool(config.get("smoke_test", False))
        self.smoke_images = int(config.get("smoke_images", 10))
        #: Run identifier used for the output directory and as the prefix of every
        #: artifact uploaded to Drive, so concurrent runs never overwrite each
        #: other. Smoke runs carry their own suffix and stay separate from the real
        #: results.
        self.run_name = f"f1_{condition}" + ("_smoke" if self.smoke_test else "")
        config.setdefault("experiment_condition", f"F1_{condition.upper()}")
        self.io_manager = IOManager(token_path=config.get("token_path"))

    # ===================================================================
    # Drive Destinations
    # ===================================================================

    @property
    def drive_results_folder_id(self) -> str | None:
        """Drive folder for logs, plots and metrics, or None if Drive is off."""
        if not self.io_manager.drive_service:
            return None
        return self.config.get("drive_folder_id") or None

    @property
    def drive_checkpoints_folder_id(self) -> str | None:
        """Drive folder for model weights, falling back to the results folder."""
        if not self.io_manager.drive_service:
            return None
        return (
            self.config.get("drive_checkpoints_folder_id")
            or self.config.get("drive_folder_id")
            or None
        )

    def remote_name_for(self, local_name: str) -> str:
        """Prefix an artifact name with the run identifier.

        Every run produces files with identical generic names (``last.pt``,
        ``results.csv``), and they all land in one shared Drive folder, so the
        prefix is what keeps them from overwriting each other.

        Args:
            local_name: File name as written by the training framework.

        Returns:
            Name to use in Drive, e.g. ``f1_c1_last.pt``.
        """
        return f"{self.run_name}_{local_name}"

    # ===================================================================
    # Abstract Method Implementations
    # ===================================================================

    def get_hyperparameters(self) -> dict[str, Any]:
        """Return the hyperparameters for the active condition.

        Returns:
            Dictionary of YOLO training arguments: the shared defaults merged with
            the augmentation profile of the condition.
        """
        params = dict(self.DEFAULT_HYPERPARAMS)
        params.update(self.CONDITION_PROFILES[self.condition])

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

        if self.smoke_test:
            params.update(self.smoke_overrides())

        return params

    def smoke_overrides(self) -> dict[str, Any]:
        """Return the only hyperparameters a smoke run is allowed to change.

        The point of the smoke run is to exercise the production code path, so
        the optimizer, learning rate, image size, augmentation profile and AMP
        setting stay exactly as they are in a real run. Only the amount of work
        shrinks: a handful of images, a few epochs, and a batch small enough to
        still produce several optimizer steps and to divide across both GPUs.

        Returns:
            Dictionary of overrides to merge on top of the production recipe.
        """
        batch = max(2, self.smoke_images // 2)
        if batch % 2:
            batch -= 1
        return {
            "epochs": int(self.config.get("smoke_epochs", 3)),
            "batch": batch,
            "save_period": 1,
        }

    def get_dataset_config(self) -> dict[str, Any]:
        """Return dataset paths for the active condition.

        C3 reads the LaMa-cleaned images while C1 and C2 read the raw ones. The
        labels are shared: LaMa alters pixels, never annotations.

        Returns:
            Dictionary with paths to dataset YAML, model weights, labels,
            images directory, and optional resized images zip.
        """
        labels_src = self.config.get("labels_path") or self.config.get(
            "labels_zip_path", ""
        )
        if self.condition == "c3":
            images_dir = self.config.get("lama_images_dir") or self.config.get(
                "images_dir", ""
            )
        else:
            images_dir = self.config.get("images_dir", "")
        return {
            "data_yaml_path": self.config.get("data_yaml_path", ""),
            "model_weights": self.config.get("model_weights", "yolo26s-obb.pt"),
            "labels_path": labels_src,
            "images_dir": images_dir,
            "resized_zip_path": self.config.get("resized_zip_path", ""),
        }

    IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".JPG", ".PNG")

    def _link_split_images(
        self,
        labels_dir: Path,
        images_dir: Path,
        split_subdir: Path | None,
        destination: Path,
        limit: int | None = None,
    ) -> int:
        """Link the image matching each label of a split into ``destination``.

        The label files define the split, so the images are looked up from them
        rather than the other way around. Labels are traversed in sorted order:
        without it the subset picked by a smoke run would change between sessions
        and stop being reproducible.

        Args:
            labels_dir: Directory holding the ``.txt`` labels of the split.
            images_dir: Flat directory holding images of every split.
            split_subdir: Split-specific image directory, when the source is
                already divided by split. Searched before ``images_dir``.
            destination: Directory to populate with links.
            limit: Maximum number of images to link. None links all of them.

        Returns:
            Number of images linked.
        """
        destination.mkdir(parents=True, exist_ok=True)
        if not labels_dir.exists():
            return 0

        search_dirs = [d for d in (split_subdir, images_dir) if d is not None]
        linked = 0
        for txt_path in sorted(labels_dir.glob("*.txt")):
            if limit is not None and linked >= limit:
                break
            for search_dir in search_dirs:
                found = False
                for ext in self.IMAGE_EXTENSIONS:
                    img_src = search_dir / f"{txt_path.stem}{ext}"
                    if not img_src.exists():
                        continue
                    img_dst = destination / img_src.name
                    if not img_dst.exists():
                        try:
                            img_dst.symlink_to(img_src)
                        except OSError:
                            shutil.copy2(str(img_src), str(img_dst))
                    linked += 1
                    found = True
                    break
                if found:
                    break
        return linked

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
        workspace = Path(
            self.config.get("dataset_workspace", "/tmp/dataset")
        )

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
            if not resized_extract_dir.exists() or not any(resized_extract_dir.iterdir()):
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

            # A whole-directory symlink cannot express a subset, so a smoke run
            # always links image by image even when the source is already split.
            limit = self.smoke_images if self.smoke_test else None

            if has_split_subdirs and limit is None:
                for split in ["train", "val"]:
                    src = images_dir / split
                    dst = images_dest / split
                    if src.exists() and not dst.exists():
                        try:
                            dst.symlink_to(src)
                        except OSError:
                            shutil.copytree(str(src), str(dst))
            else:
                print(
                    f"[{self.run_name}] Linking images to train/val splits"
                    + (f" (limit {limit} per split)" if limit else "")
                    + "...",
                    flush=True,
                )
                for split in ["train", "val"]:
                    linked = self._link_split_images(
                        labels_dir=labels_dest / split,
                        images_dir=images_dir,
                        split_subdir=images_dir / split if has_split_subdirs else None,
                        destination=images_dest / split,
                        limit=limit,
                    )
                    print(f"[{self.run_name}]   {split}: {linked} images", flush=True)

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
            Dictionary with 'local' path, 'remote' name and 'drive_id' (or None).
        """
        self.io_manager.save_json(data, local_path)
        record: dict[str, Any] = {"local": str(local_path), "drive_id": None}
        if drive_folder_id:
            record["remote"] = self.remote_name_for(local_path.name)
            try:
                record["drive_id"] = self.io_manager.upload_file_to_drive(
                    local_path,
                    drive_folder_id,
                    remote_name=record["remote"],
                )
            except Exception as e:
                print(
                    f"  ⚠️  Drive upload error for {local_path.name}: {e}",
                    flush=True,
                )
        return record

    def _upload_file(self, local_path: Path, drive_folder_id: str | None) -> str | None:
        """Upload an existing file to Google Drive under this run's name.

        Args:
            local_path: Local file path to upload.
            drive_folder_id: Google Drive folder ID for upload.

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
            return self.io_manager.upload_file_to_drive(
                local_path,
                drive_folder_id,
                mime_type=mime,
                remote_name=self.remote_name_for(local_path.name),
            )
        except Exception as e:
            print(
                f"  ⚠️  Drive upload error for {local_path.name}: {e}",
                flush=True,
            )
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
        from ultralytics import YOLO

        self.start_hardware_monitoring()
        output_dir = Path(self.config.get("output_dir", "/kaggle/working/runs"))
        drive_folder_id: str | None = self.drive_results_folder_id
        checkpoints_folder_id: str | None = self.drive_checkpoints_folder_id

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
        hyperparams["name"] = self.run_name
        # A recipe that sets its own checkpoint cadence wins over the config: a
        # smoke run needs to save every epoch to exercise the sync callback.
        save_period: int = int(
            hyperparams.get("save_period") or self.config.get("save_period", 5)
        )
        hyperparams["save_period"] = save_period

        # 3. Resume only from a checkpoint that belongs to this run.
        #
        # Ultralytics reads the training arguments back from the checkpoint when
        # resume=True and ignores any override, so resuming from another run's
        # checkpoint silently restores that run's epoch budget and augmentation.
        # The run-prefixed remote name is what keeps the runs isolated inside the
        # shared checkpoints folder.
        train_dir = output_dir / self.run_name
        local_last_pt = train_dir / "weights" / "last.pt"
        resumed_checkpoint: Path | None = None

        if checkpoints_folder_id and not self.smoke_test:
            print(
                f"[{self.run_name}] Looking for a resumable checkpoint of this run...",
                flush=True,
            )
            try:
                resumed_checkpoint = self.io_manager.download_file_from_drive(
                    file_name=self.remote_name_for("last.pt"),
                    drive_folder_id=checkpoints_folder_id,
                    local_destination_path=local_last_pt,
                )
            except Exception as e:
                print(
                    f"[{self.run_name}] Checkpoint download check skipped: {e}",
                    flush=True,
                )

        # A smoke run always starts fresh: its purpose is to validate the cold-start
        # path, and resuming would make it non-deterministic between sessions.
        if not resumed_checkpoint and local_last_pt.exists() and not self.smoke_test:
            resumed_checkpoint = local_last_pt

        if resumed_checkpoint and resumed_checkpoint.exists():
            print(
                f"[{self.run_name}] 🔄 Resuming training from checkpoint: {resumed_checkpoint}",
                flush=True,
            )
            model = YOLO(str(resumed_checkpoint))
            hyperparams["resume"] = True
        else:
            print(
                f"[{self.run_name}] 🚀 Starting fresh training from: {dataset_config['model_weights']}",
                flush=True,
            )
            model = YOLO(dataset_config["model_weights"])
            hyperparams["resume"] = False

        total_epochs = hyperparams.get("epochs", 40)

        def sync_checkpoint_callback(trainer_obj: Any) -> None:
            """Incremental Drive sync triggered on_model_save.

            Uploads artifacts every ``save_period`` epochs and on the final epoch,
            which bounds Drive API traffic while keeping the run resumable after a
            session timeout.
            """
            if not self.io_manager.drive_service:
                return
            if not _is_primary_process():
                return

            current_epoch = getattr(trainer_obj, "epoch", 0) + 1  # 0-indexed -> 1-indexed
            is_final = current_epoch >= total_epochs

            if current_epoch % save_period != 0 and not is_final:
                return

            try:
                save_dir = Path(getattr(trainer_obj, "save_dir", train_dir))

                artifact_files = [
                    "results.csv",
                    "results.png",
                    "args.yaml",
                    "confusion_matrix.png",  # Recommended for paper
                    "PR_curve.png",  # Recommended for paper
                ]
                for fname in artifact_files:
                    fpath = save_dir / fname
                    if fpath.exists():
                        self._upload_file(fpath, drive_folder_id)

                weights_dir = save_dir / "weights"
                for wname in ["best.pt", "last.pt"]:
                    wpath = weights_dir / wname
                    if wpath.exists():
                        self._upload_file(wpath, checkpoints_folder_id)

                print(
                    f"  📤 Drive sync completed (epoch {current_epoch}/{total_epochs})",
                    flush=True,
                )
            except Exception as e:
                print(f"  ⚠️ Incremental sync callback error: {e}", flush=True)

        model.add_callback("on_model_save", sync_checkpoint_callback)

        # 4. Train (batch=96 splits to 48/GPU via DDP, 13.6/15 GB measured on T4)
        _ = model.train(**hyperparams)

        # 5. Locate training output directory
        if not train_dir.exists():
            # Ultralytics appends a suffix when the directory already exists
            candidates = sorted(output_dir.glob(f"{self.run_name}*"))
            train_dir = candidates[-1] if candidates else output_dir

        # 6. Parse results.csv
        results_csv = train_dir / "results.csv"
        epochs_data: list[dict[str, Any]] = []
        if results_csv.exists():
            epochs_data = self.parse_results_csv(results_csv)

        # 7. Record hardware metrics and confirm every GPU actually did work
        hardware_metrics = self.record_hardware_metrics()
        gpu_report = self.report_gpu_usage(hardware_metrics)
        hardware_metrics["multi_gpu_verified"] = gpu_report["multi_gpu_verified"]

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
                generated_files.append(
                    {
                        "local": str(fpath),
                        "remote": self.remote_name_for(fname),
                        "drive_id": drive_id,
                    }
                )

        # Upload best and last weights to the dedicated checkpoints folder
        weights_dir = train_dir / "weights"
        for wname in ["best.pt", "last.pt"]:
            wpath = weights_dir / wname
            if wpath.exists():
                drive_id = self._upload_file(wpath, checkpoints_folder_id)
                generated_files.append(
                    {
                        "local": str(wpath),
                        "remote": self.remote_name_for(wname),
                        "drive_id": drive_id,
                    }
                )

        print(
            f"[{self.run_name}] Training complete. {len(epochs_data)} epochs logged.",
            flush=True,
        )
        sys.stdout.flush()
        sys.stderr.flush()

        return {
            "status": "success",
            "run_name": self.run_name,
            "train_dir": str(train_dir),
            "files": generated_files,
            "metrics": training_metrics,
            "gpu_report": gpu_report,
        }

    # ===================================================================
    # GPU Usage Verification
    # ===================================================================

    def report_gpu_usage(self, hardware_metrics: dict[str, Any]) -> dict[str, Any]:
        """Print and summarize whether every visible GPU carried a workload.

        Requesting ``device='0,1'`` does not guarantee that both devices were used:
        Ultralytics falls back to a single GPU on several conditions, and the
        failure is silent because training still completes. The sampled per-device
        peaks are the evidence that DDP really engaged both cards.

        Args:
            hardware_metrics: Output of ``record_hardware_metrics``.

        Returns:
            Dictionary with ``expected_gpus``, ``gpus_engaged``, ``devices`` and the
            ``multi_gpu_verified`` verdict.
        """
        sampling = hardware_metrics.get("gpu_sampling", {})
        devices = sampling.get("devices", [])
        expected = int(
            self.config.get("expected_gpus", torch.cuda.device_count())
        )
        engaged = int(sampling.get("gpus_engaged", 0))
        verified = bool(devices) and engaged >= expected and expected > 0

        print("\n" + "-" * 60, flush=True)
        print(f"GPU USAGE [{self.run_name}]", flush=True)
        print("-" * 60, flush=True)
        if not sampling.get("available"):
            print("  nvidia-smi unavailable: GPU usage could not be verified.")
        else:
            for device in devices:
                print(
                    f"  GPU {device['index']} ({device['name']}): "
                    f"peak {device['peak_memory_used_mib']:.0f} MiB of "
                    f"{device['memory_total_mib']:.0f} MiB, "
                    f"peak util {device['peak_utilization_pct']:.0f}%, "
                    f"mean util {device['mean_utilization_pct']:.0f}%",
                    flush=True,
                )
            print(
                f"  Engaged {engaged} of {expected} expected GPUs -> "
                f"{'VERIFIED ✅' if verified else 'NOT VERIFIED ❌'}",
                flush=True,
            )
        print("-" * 60 + "\n", flush=True)

        return {
            "expected_gpus": expected,
            "gpus_engaged": engaged,
            "multi_gpu_verified": verified,
            "devices": devices,
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

    parser = argparse.ArgumentParser(description="F1 (YOLO26s-OBB) trainer")
    parser.add_argument(
        "--condition",
        choices=sorted(Base1Trainer.CONDITION_PROFILES),
        default="c1",
        help="Ablation condition: c1 raw data, c2 classic augmentation, c3 LaMa data",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Production recipe on a few images per split, to validate the pipeline",
    )
    parser.add_argument(
        "--smoke-images",
        type=int,
        default=10,
        help="Images per split for the smoke test (default: 10)",
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=3,
        help="Epochs for the smoke test (default: 3)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override total epochs"
    )
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--fraction", type=float, default=None, help="Dataset fraction (0.0 to 1.0)"
    )
    args = parser.parse_args()

    IS_KAGGLE = os.path.exists("/kaggle/working")

    dataset_dir = _find_dataset_dir()

    # Resolve labels path: unzipped directory vs zip file
    labels_path = dataset_dir / "yolo_obb_labels"
    if not labels_path.exists():
        labels_path = dataset_dir / "yolo_obb_labels.zip"

    # Prefer the already-extracted 640x360 directory over the zip: symlinking is
    # much cheaper than unpacking ~54k files on every session.
    resized_dir = dataset_dir / "train_resized" / "train"
    resized_zip = dataset_dir / "train_resized.zip"
    if resized_dir.exists():
        raw_images_dir = resized_dir
        resized_zip = Path("")
    elif resized_zip.exists():
        raw_images_dir = dataset_dir / "train-001" / "train"
    else:
        raw_images_dir = dataset_dir / "train-001" / "train"
        resized_zip = Path("")

    lama_images_dir = dataset_dir / "smart_lama_corrected" / "train"

    # Use /tmp for the dataset workspace on Kaggle to avoid the 20GB output limit.
    # One workspace per condition: C1 and C3 link different images under the same
    # file names, so a shared workspace would silently mix both variants.
    workspace_root = Path("/tmp") if IS_KAGGLE else Path("/content")
    workspace_suffix = f"{args.condition}{'_smoke' if args.smoke_test else ''}"
    dataset_workspace = workspace_root / f"dataset_{workspace_suffix}"
    data_yaml_path = dataset_workspace / "smart_dataset.yaml"

    output_dir = Path("/kaggle/working/runs") if IS_KAGGLE else Path("/content/runs")

    config = {
        "condition": args.condition,
        "output_dir": str(output_dir),
        "model_weights": "yolo26s-obb.pt",
        "labels_path": str(labels_path),
        "data_yaml_path": str(data_yaml_path),
        "images_dir": str(raw_images_dir),
        "lama_images_dir": str(lama_images_dir),
        "resized_zip_path": str(resized_zip),
        "dataset_workspace": str(dataset_workspace),
        "save_period": 5,
        "hardware_name": "Tesla_T4x2_Kaggle" if IS_KAGGLE else "Colab_GPU",
        "token_path": str(
            Path("/kaggle/working/token.json")
            if IS_KAGGLE
            else Path("/content/token.json")
        ),
        # Results (logs, plots, metrics) and weights live in different Drive folders
        "drive_folder_id": "1n17lmU2SVz54HmV6a3Cd-bgKs0h6bQP8",
        "drive_checkpoints_folder_id": "1pn8OzJX_kctgluEZkSC6WEfbSCaPyKMa",
        "smoke_test": args.smoke_test,
        "smoke_images": args.smoke_images,
        "smoke_epochs": args.smoke_epochs,
    }

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch is not None:
        config["batch"] = args.batch
    if args.fraction is not None:
        config["fraction"] = args.fraction

    trainer = Base1Trainer(config=config)

    print("=" * 60)
    print(f"F1 TRAINING [{trainer.run_name}] - HEALTH CHECK")
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
    print(f"Run: {results['run_name']}")
    print(f"Status: {results['status']}")
    print(f"Train dir: {results['train_dir']}")
    print(f"Epochs completed: {results['metrics']['total_epochs_completed']}")
    print(f"Multi-GPU verified: {results['gpu_report']['multi_gpu_verified']}")
    print(f"Generated files: {len(results['files'])}")
    for f in results["files"]:
        remote = f.get("remote", Path(f["local"]).name)
        status = f"Drive '{remote}'" if f["drive_id"] else "Local only"
        print(f"  📄 {Path(f['local']).name} -> {status}")

    if not results["gpu_report"]["multi_gpu_verified"]:
        print(
            "\n[WARNING] Not every expected GPU carried a workload. "
            "Review the GPU USAGE block above before launching a real run."
        )
