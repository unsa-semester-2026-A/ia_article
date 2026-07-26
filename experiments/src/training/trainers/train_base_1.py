"""YOLO26s-OBB trainer for all ablation-study F1 conditions."""

import json
import os
import resource
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import torch
from src.training.base_training import BaseTrainingPipeline
from src.utils.gpu_monitor import sample_gpus
from src.utils.io_manager import IOManager


def _is_primary_process() -> bool:
    """Report whether this process should perform Drive I/O.

    Ultralytics runs training callbacks inside every DDP worker. Without this
    guard each worker uploads the same checkpoint concurrently, which duplicates
    files and triggers conflicting writes against the Drive API.
    """
    return int(os.environ.get("RANK", -1)) in (-1, 0)


# Results and checkpoints deliberately use separate folders. The run prefix is
# still applied by ``remote_name_for`` so generic framework names never collide.
DRIVE_DESTINATIONS: dict[str, dict[str, str]] = {
    "c1": {
        "results": "1n17lmU2SVz54HmV6a3Cd-bgKs0h6bQP8",
        "checkpoints": "1pn8OzJX_kctgluEZkSC6WEfbSCaPyKMa",
    },
    "c2": {
        "results": "1tQ4j3sd0BajIiE1uGV11jD1UogJe3xlh",
        "checkpoints": "1navKsrapRDxJzbLVrDIHpmDbkU4zHtVN",
    },
    "c3": {
        "results": "1maJ2IelUfaV4DPMSzmd8gK-eaFOhOchs",
        "checkpoints": "1Hi8OmTIMNzLfadjFbk79OL8yiIZewhpz",
    },
    "mb": {
        "results": "15FfYezjs7wFRlaXF4LTFRfruTzG6opLs",
        "checkpoints": "1ag0EPL1f-7T93HjusTBUNXmUWNnZwrBy",
    },
    "mc": {
        "results": "1XJs4VVQXZ-cruuvwyOtG3rBvuCBO76-l",
        "checkpoints": "1_kAXLZNnG3m76gdbdLd3nJ5QtJ-MOt96",
    },
}


class Base1Trainer(BaseTrainingPipeline):
    """Training pipeline for the F1 family (YOLO26s-OBB) of the ablation study.

    Handles the F1 conditions of ``06_training.md`` §4, selected with the
    ``condition`` config key:

    - ``c1``: raw data, minimal augmentation. Baseline, denominator of the
      intra-family gain.
    - ``c2``: raw data, classic YOLO augmentation (mosaic, mixup, copy-paste).
    - ``c3``: LaMa-cleaned data, minimal augmentation. Numerator of the gain.
    - ``mb``: raw data plus a frozen, train-only synthetic delta (Mejora B).
    - ``mc``: LaMa data plus the same synthetic delta (Mejora C).

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

    #: C2 uses Mosaic and MixUp, which can make its per-image instance count much
    #: higher than C1/C3.  These are *global* DDP batch sizes; Ultralytics divides
    #: them between the two T4s.  Every lower candidate divides the target 96
    #: exactly, so it can retain the same effective optimizer update through
    #: gradient accumulation.
    C2_CALIBRATION_CANDIDATES: tuple[int, ...] = (96, 48, 32, 24)
    C2_TARGET_GLOBAL_BATCH = 96
    C2_BASELINE_NBS = 64
    C2_BASELINE_WEIGHT_DECAY = 0.0005

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
        # Synthetic images are the experimental variable in MB/MC. Do not add
        # online object-combining transforms on top of them.
        "mb": {
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.3,
            "hsv_v": 0.2,
        },
        "mc": {
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
        self.calibration_mode = bool(config.get("c2_calibration_mode", False))
        self.c2_calibration_batch = config.get("c2_calibration_batch")
        self.c2_selected_batch = config.get("c2_selected_batch")
        self.calibration_images = int(config.get("calibration_images", 384))
        if self.calibration_mode:
            if self.condition != "c2":
                raise ValueError(
                    "C2 batch calibration is only valid for condition 'c2'."
                )
            if self.c2_calibration_batch not in self.C2_CALIBRATION_CANDIDATES:
                raise ValueError(
                    "c2_calibration_batch must be one of "
                    f"{self.C2_CALIBRATION_CANDIDATES}."
                )
            if self.calibration_images < int(self.c2_calibration_batch):
                raise ValueError(
                    "calibration_images must be at least the selected global batch."
                )
        if self.c2_selected_batch is not None:
            if self.condition != "c2":
                raise ValueError("c2_selected_batch is only valid for condition 'c2'.")
            self.c2_batch_plan(int(self.c2_selected_batch))
        #: Run identifier used for the output directory and as the prefix of every
        #: artifact uploaded to Drive, so concurrent runs never overwrite each
        #: other. Smoke runs carry their own suffix and stay separate from the real
        #: results.
        if self.calibration_mode:
            self.run_name = f"f1_c2_batchcal_b{self.c2_calibration_batch}"
        else:
            run_names = {"mb": "f1_mejora_b", "mc": "f1_mejora_c"}
            self.run_name = run_names.get(condition, f"f1_{condition}") + (
                "_smoke" if self.smoke_test else ""
            )
        config.setdefault("experiment_condition", f"F1_{condition.upper()}")
        # A production run must not start if it cannot persist its weights: a
        # Kaggle session dies after 12 hours and the quota would be spent for
        # nothing. A smoke run produces disposable artifacts, so it proceeds and
        # still validates the GPU and dataset path when Drive is unavailable.
        self.io_manager = IOManager(
            token_path=config.get("token_path"),
            require_drive=not (self.smoke_test or self.calibration_mode),
        )

    @classmethod
    def c2_batch_plan(cls, global_batch: int) -> dict[str, Any]:
        """Build a reproducible C2 memory-calibration candidate.

        ``batch`` is the global DDP batch.  For a candidate below 96 we set
        ``nbs=96`` so Ultralytics accumulates enough batches to retain a global
        optimizer update of 96 samples.  Weight decay is adjusted to preserve the
        effective decay of the original ``batch=96, nbs=64`` recipe.

        Args:
            global_batch: Candidate global DDP batch size.

        Returns:
            Training overrides plus auditable derived quantities.
        """
        if global_batch not in cls.C2_CALIBRATION_CANDIDATES:
            raise ValueError(
                f"Unsupported C2 batch {global_batch}; expected one of "
                f"{cls.C2_CALIBRATION_CANDIDATES}."
            )

        if global_batch == cls.C2_TARGET_GLOBAL_BATCH:
            nbs = cls.C2_BASELINE_NBS
            accumulation = 1
            weight_decay = cls.C2_BASELINE_WEIGHT_DECAY
        else:
            nbs = cls.C2_TARGET_GLOBAL_BATCH
            accumulation = cls.C2_TARGET_GLOBAL_BATCH // global_batch
            # Ultralytics scales decay by batch * accumulation / nbs.
            # Preserve the original 96 / 64 multiplier exactly.
            weight_decay = (
                cls.C2_BASELINE_WEIGHT_DECAY
                * cls.C2_TARGET_GLOBAL_BATCH
                / cls.C2_BASELINE_NBS
            )

        return {
            "batch": global_batch,
            "nbs": nbs,
            "weight_decay": weight_decay,
            "expected_accumulate": accumulation,
            "effective_global_batch": global_batch * accumulation,
        }

    # ===================================================================
    # Drive Destinations
    # ===================================================================

    @property
    def drive_results_folder_id(self) -> str | None:
        """Drive folder for logs, plots and metrics, or None if Drive is off."""
        if self.calibration_mode or not self.io_manager.drive_service:
            return None
        return self.config.get("drive_folder_id") or None

    @property
    def drive_checkpoints_folder_id(self) -> str | None:
        """Drive folder for model weights, falling back to the results folder."""
        if self.calibration_mode or not self.io_manager.drive_service:
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
            "nbs",
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

        if self.c2_selected_batch is not None:
            batch_plan = self.c2_batch_plan(int(self.c2_selected_batch))
            params.update(
                {key: batch_plan[key] for key in ("batch", "nbs", "weight_decay")}
            )

        if self.calibration_mode:
            # The calibration is an operational probe, not an article result. It
            # deliberately keeps the full C2 augmentation profile but only runs
            # one epoch over a deterministic dense subset and produces no weights.
            batch_plan = self.c2_batch_plan(int(self.c2_calibration_batch))
            params.update(
                {key: batch_plan[key] for key in ("batch", "nbs", "weight_decay")}
            )
            params.update(
                {
                    "epochs": 1,
                    "patience": 0,
                    "val": False,
                    "save": False,
                    "plots": False,
                    # With one epoch, close_mosaic=10 would disable Mosaic at the
                    # first iteration. Keep it enabled so the probe is worst-case.
                    "close_mosaic": 0,
                    "save_period": -1,
                }
            )

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

        C3/MC read LaMa-cleaned images while C1/C2/MB read raw ones. MB and MC
        link the same train-only synthetic delta; validation remains raw.

        Returns:
            Dictionary with paths to dataset YAML, model weights, labels,
            images directory, and optional resized images zip.
        """
        labels_src = self.config.get("labels_path") or self.config.get(
            "labels_zip_path", ""
        )
        raw_images_dir = self.config.get("images_dir", "")
        lama_images_dir = self.config.get("lama_images_dir", "")
        train_images_dir = (
            lama_images_dir if self.condition in {"c3", "mc"} else raw_images_dir
        )
        use_synthetic_delta = self.condition in {"mb", "mc"}
        return {
            "data_yaml_path": self.config.get("data_yaml_path", ""),
            "model_weights": self.config.get("model_weights", "yolo26s-obb.pt"),
            "labels_path": labels_src,
            "images_dir": train_images_dir,
            "train_images_dir": train_images_dir,
            # Validation is always raw. LaMa changes the training pixels only;
            # keeping validation fixed makes every F1 metric directly comparable.
            "val_images_dir": raw_images_dir,
            "raw_images_dir": raw_images_dir,
            "lama_images_dir": lama_images_dir,
            "resized_zip_path": self.config.get("resized_zip_path", ""),
            "augmentation_delta_images_dir": (
                self.config.get("augmentation_delta_images_dir", "")
                if use_synthetic_delta else ""
            ),
            "augmentation_delta_labels_dir": (
                self.config.get("augmentation_delta_labels_dir", "")
                if use_synthetic_delta else ""
            ),
        }

    IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".JPG", ".PNG")

    @classmethod
    def _image_stems(cls, directory: Path) -> set[str]:
        """Return image stems directly contained in a directory.

        Args:
            directory: Flat image directory to scan.

        Returns:
            Set of filename stems with a supported image extension.
        """
        if not directory.is_dir():
            return set()
        suffixes = {
            extension.lower().removeprefix(".") for extension in cls.IMAGE_EXTENSIONS
        }
        with os.scandir(directory) as entries:
            return {
                entry.name.rsplit(".", 1)[0]
                for entry in entries
                if entry.is_file() and entry.name.rsplit(".", 1)[-1].lower() in suffixes
            }

    @staticmethod
    def _link_label_files(
        source: Path, destination: Path, allowed_stems: set[str] | None = None
    ) -> None:
        """Link labels, optionally retaining only a reproducible stem manifest.

        Args:
            source: Directory containing YOLO ``.txt`` labels.
            destination: Workspace directory that receives the label links.
            allowed_stems: Optional set of stems to retain.
        """
        destination.mkdir(parents=True, exist_ok=True)
        for label in sorted(source.glob("*.txt")):
            if allowed_stems is not None and label.stem not in allowed_stems:
                continue
            target = destination / label.name
            if target.exists():
                continue
            try:
                target.symlink_to(label)
            except OSError:
                shutil.copy2(label, target)

    def _link_split_images(
        self,
        labels_dir: Path,
        images_dir: Path,
        split_subdir: Path | None,
        destination: Path,
        limit: int | None = None,
        skip_empty_labels: bool = False,
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
            skip_empty_labels: Whether empty label files should be skipped.

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
            if skip_empty_labels and txt_path.stat().st_size == 0:
                continue
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

        # The 79 frames not processed by LaMa must be excluded from *all* F1
        # conditions. Otherwise C1/C2 and C3 would learn from different frames.
        raw_images_dir = Path(dataset_config.get("raw_images_dir", ""))
        lama_images_dir = Path(dataset_config.get("lama_images_dir", ""))
        common_train_stems: set[str] | None = None
        calibration_val_stems: set[str] | None = None
        if (
            labels_source.is_dir()
            and raw_images_dir.is_dir()
            and lama_images_dir.is_dir()
        ):
            label_stems = {
                path.stem for path in (labels_source / "train").glob("*.txt")
            }
            common_train_stems = (
                label_stems
                & self._image_stems(raw_images_dir)
                & self._image_stems(lama_images_dir)
            )
            if not common_train_stems:
                raise RuntimeError("No common raw/LaMa training images were found.")
            if self.calibration_mode:
                # Pick the densest labels deterministically. Mosaic combines four
                # inputs, so this stresses the exact TaskAlignedAssigner pressure
                # that triggered the C2 CPU fallback without changing the article
                # dataset or producing scientific metrics.
                ranked_stems = sorted(
                    common_train_stems,
                    key=lambda stem: (
                        -sum(
                            1
                            for line in (labels_source / "train" / f"{stem}.txt")
                            .read_text(encoding="utf-8")
                            .splitlines()
                            if line.strip()
                        ),
                        stem,
                    ),
                )
                common_train_stems = set(ranked_stems[: self.calibration_images])
                nonempty_val = sorted(
                    path.stem
                    for path in (labels_source / "val").glob("*.txt")
                    if path.stat().st_size > 0
                )
                calibration_val_stems = set(nonempty_val[:1])
                if not calibration_val_stems:
                    raise RuntimeError(
                        "No non-empty validation label is available for calibration."
                    )
            manifest = workspace / "common_train_stems.txt"
            manifest.write_text("\n".join(sorted(common_train_stems)) + "\n")
            excluded = len(label_stems) - len(common_train_stems)
            if self.calibration_mode:
                print(
                    f"[{self.run_name}] Dense calibration manifest: "
                    f"{len(common_train_stems)} frames selected from "
                    f"{len(label_stems)} labelled frames",
                    flush=True,
                )
            else:
                print(
                    f"[{self.run_name}] Common F1 train manifest: "
                    f"{len(common_train_stems)} frames ({excluded} excluded)",
                    flush=True,
                )

        # Handle labels source (directory vs zip file)
        if labels_source.name and labels_source.exists():
            labels_dest.mkdir(parents=True, exist_ok=True)
            if labels_source.is_dir():
                for split in ["train", "val"]:
                    src = labels_source / split
                    dst = labels_dest / split
                    if common_train_stems is not None:
                        self._link_label_files(
                            src,
                            dst,
                            (
                                common_train_stems
                                if split == "train"
                                else calibration_val_stems
                            ),
                        )
                    elif src.exists() and not dst.exists():
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

            # A whole-directory symlink cannot express a subset, so a smoke run
            # always links image by image even when the source is already split.
            limit = self.smoke_images if self.smoke_test else None
            train_images_dir = Path(dataset_config["train_images_dir"])
            val_images_dir = Path(dataset_config["val_images_dir"])
            has_matching_split_dirs = (
                train_images_dir == val_images_dir
                and (train_images_dir / "train").is_dir()
                and (train_images_dir / "val").is_dir()
            )
            if common_train_stems is None and limit is None and has_matching_split_dirs:
                for split in ["train", "val"]:
                    src = train_images_dir / split
                    dst = images_dest / split
                    if not dst.exists():
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
                    split_images_dir = Path(dataset_config[f"{split}_images_dir"])
                    has_split_subdirs = (split_images_dir / split).is_dir()
                    linked = self._link_split_images(
                        labels_dir=labels_dest / split,
                        images_dir=split_images_dir,
                        split_subdir=(
                            split_images_dir / split if has_split_subdirs else None
                        ),
                        destination=images_dest / split,
                        limit=1 if self.calibration_mode and split == "val" else limit,
                        skip_empty_labels=self.smoke_test and split == "val",
                    )
                    print(f"[{self.run_name}]   {split}: {linked} images", flush=True)

        self._link_augmentation_delta(
            images_dest=images_dest,
            labels_dest=labels_dest,
            images_source=Path(dataset_config["augmentation_delta_images_dir"]),
            labels_source=Path(dataset_config["augmentation_delta_labels_dir"]),
            manifest_path=workspace / "augmentation_delta_manifest.json",
        )

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

    @classmethod
    def _link_augmentation_delta(
        cls,
        *,
        images_dest: Path,
        labels_dest: Path,
        images_source: Path,
        labels_source: Path,
        manifest_path: Path | None = None,
    ) -> int:
        """Link a compact synthetic train delta without duplicating base data.

        The delta must have matching image/label stems and must not shadow a
        real training frame.  Validation deliberately remains untouched.
        """
        if not images_source.name and not labels_source.name:
            return 0
        if not images_source.is_dir() or not labels_source.is_dir():
            raise FileNotFoundError(
                "Augmentation delta image and label directories are required"
            )
        image_stems = cls._image_stems(images_source)
        label_stems = {path.stem for path in labels_source.glob("*.txt")}
        if not image_stems:
            raise ValueError("Augmentation delta is empty")
        if image_stems != label_stems:
            raise ValueError("Augmentation delta image/label stems do not match")
        image_destination = images_dest / "train"
        label_destination = labels_dest / "train"
        image_destination.mkdir(parents=True, exist_ok=True)
        label_destination.mkdir(parents=True, exist_ok=True)
        collisions = image_stems & {
            path.stem for path in label_destination.glob("*.txt")
        }
        if collisions:
            raise ValueError(
                f"Augmentation delta collides with real stems: {sorted(collisions)[:3]}"
            )
        for stem in sorted(image_stems):
            label = labels_source / f"{stem}.txt"
            cls._validate_obb_label(label)
            image = next(
                images_source / f"{stem}{extension}"
                for extension in cls.IMAGE_EXTENSIONS
                if (images_source / f"{stem}{extension}").exists()
            )
            for source, destination in (
                (image, image_destination / image.name),
                (label, label_destination / f"{stem}.txt"),
            ):
                try:
                    destination.symlink_to(source)
                except OSError:
                    shutil.copy2(source, destination)
        if image_stems:
            print(f"[Base1Trainer] Linked {len(image_stems)} augmentation-delta images")
        if manifest_path is not None:
            manifest_path.write_text(
                json.dumps(
                    {
                        "image_count": len(image_stems),
                        "label_count": len(label_stems),
                        "images_source": str(images_source),
                        "labels_source": str(labels_source),
                        "validation_modified": False,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return len(image_stems)

    @staticmethod
    def _validate_obb_label(label_path: Path) -> None:
        """Reject malformed synthetic YOLO-OBB labels before a GPU run starts."""
        lines = [
            line.split()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            raise ValueError(f"Synthetic label is empty: {label_path.name}")
        for line_number, fields in enumerate(lines, start=1):
            if len(fields) != 9:
                raise ValueError(
                    f"Invalid OBB field count in {label_path.name}:{line_number}"
                )
            try:
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric OBB value in {label_path.name}:{line_number}"
                ) from exc
            if not 0 <= class_id <= 8 or any(
                not 0.0 <= value <= 1.0 for value in coordinates
            ):
                raise ValueError(
                    f"Out-of-range OBB value in {label_path.name}:{line_number}"
                )

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
            message = (
                "drive_folder_id configured but Google Drive service is unavailable "
                "(Auth/token failed)."
            )
            # Only a real run depends on Drive to survive a Kaggle session timeout.
            # Blocking the smoke run here would withhold the GPU and dataset
            # evidence it exists to produce.
            if self.smoke_test or self.calibration_mode:
                checks["details"]["drive_service"]["warning"] = (
                    f"{message} Tolerated: this operational probe has disposable artifacts."
                )
            else:
                checks["details"]["drive_service"]["error"] = message
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

    def _checkpoint_state(
        self, epoch: int, save_dir: Path, hyperparameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a durable latest-state record paired with ``last.pt``.

        The state file is intentionally overwritten under one run-specific
        remote name. It describes the exact epoch represented by the likewise
        overwritten ``last.pt`` and prevents a partial session from being
        mistaken for a completed run.

        Args:
            epoch: One-based epoch whose checkpoint was just saved.
            save_dir: Ultralytics run directory.
            hyperparameters: Effective train arguments for this run.

        Returns:
            JSON-serializable checkpoint state.
        """
        elapsed = max(0.0, time.time() - self.start_time)
        peak_ram_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        results_csv = save_dir / "results.csv"
        latest_metrics: dict[str, Any] = {}
        if results_csv.exists():
            rows = self.parse_results_csv(results_csv)
            if rows:
                latest_metrics = rows[-1]
        return {
            "run_name": self.run_name,
            "experiment_condition": self.config.get("experiment_condition"),
            "checkpoint_epoch": epoch,
            "checkpoint_kind": "latest resumable state",
            "elapsed_seconds": round(elapsed, 2),
            "peak_cpu_ram_gb": round(peak_ram_kb / (1024 * 1024), 2),
            "gpu_snapshot": sample_gpus(),
            "latest_epoch_metrics": latest_metrics,
            "hyperparameters": hyperparameters,
            "weights": {
                "last": "last.pt",
                "best": "best.pt",
            },
        }

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
        # ``save_period`` has two distinct meanings here.  Drive synchronization
        # should happen every N completed epochs, while Ultralytics must invoke
        # ``on_model_save`` every epoch so the callback can observe an exact
        # one-based epoch number.  Ultralytics' internal counter is zero-based;
        # using the same N for both otherwise shifts saves to 1, N+1, 2N+1 ...
        # and the callback rejects every intermediate checkpoint.
        drive_sync_period: int = int(
            hyperparams.get("save_period") or self.config.get("save_period", 5)
        )
        hyperparams["save_period"] = 1 if checkpoints_folder_id else drive_sync_period

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

            current_epoch = (
                getattr(trainer_obj, "epoch", 0) + 1
            )  # 0-indexed -> 1-indexed
            is_final = current_epoch >= total_epochs

            if current_epoch % drive_sync_period != 0 and not is_final:
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

                state_path = save_dir / "checkpoint_state.json"
                self.io_manager.save_json(
                    self._checkpoint_state(current_epoch, save_dir, hyperparams),
                    state_path,
                )
                self._upload_file(state_path, checkpoints_folder_id)

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
        expected = int(self.config.get("expected_gpus", torch.cuda.device_count()))
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
        help="c1 raw, c2 classic, c3 LaMa, mb raw+synthetic, mc LaMa+synthetic",
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
        "--c2-calibration-batch",
        type=int,
        choices=Base1Trainer.C2_CALIBRATION_CANDIDATES,
        default=None,
        help=(
            "Run one disposable dense C2 calibration candidate. Use "
            "run_c2_batch_calibration.py to evaluate the full ladder."
        ),
    )
    parser.add_argument(
        "--c2-batch",
        type=int,
        choices=Base1Trainer.C2_CALIBRATION_CANDIDATES,
        default=None,
        help=(
            "Apply a batch selected by the C2 calibration to the production C2 run, "
            "preserving its effective global optimizer batch."
        ),
    )
    parser.add_argument(
        "--calibration-images",
        type=int,
        default=384,
        help="Dense train images used by a disposable C2 calibration candidate",
    )
    parser.add_argument(
        "--fraction", type=float, default=None, help="Dataset fraction (0.0 to 1.0)"
    )
    parser.add_argument(
        "--drive-folder-id", default=None, help="Override results Drive folder ID"
    )
    parser.add_argument(
        "--drive-checkpoints-folder-id",
        default=None,
        help="Override checkpoints Drive folder ID",
    )
    args = parser.parse_args()
    if args.c2_batch is not None and args.c2_calibration_batch is not None:
        parser.error("--c2-batch and --c2-calibration-batch cannot be used together.")

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
    augmentation_images_dir = (
        dataset_dir / "augmentation_images" / "images" / "train"
    )
    augmentation_labels_dir = (
        dataset_dir / "augmentation_labels" / "labels" / "train"
    )

    # Use /tmp for the dataset workspace on Kaggle to avoid the 20GB output limit.
    # One workspace per condition: C1 and C3 link different images under the same
    # file names, so a shared workspace would silently mix both variants.
    workspace_root = Path("/tmp") if IS_KAGGLE else Path("/content")
    workspace_suffix = f"{args.condition}{'_smoke' if args.smoke_test else ''}"
    if args.c2_calibration_batch is not None:
        workspace_suffix = f"c2_batchcal_b{args.c2_calibration_batch}"
    dataset_workspace = workspace_root / f"dataset_{workspace_suffix}"
    data_yaml_path = dataset_workspace / "smart_dataset.yaml"

    output_dir = Path("/kaggle/working/runs") if IS_KAGGLE else Path("/content/runs")

    destinations = DRIVE_DESTINATIONS[args.condition]
    config = {
        "condition": args.condition,
        "output_dir": str(output_dir),
        "model_weights": "yolo26s-obb.pt",
        "labels_path": str(labels_path),
        "data_yaml_path": str(data_yaml_path),
        "images_dir": str(raw_images_dir),
        "lama_images_dir": str(lama_images_dir),
        "augmentation_delta_images_dir": str(augmentation_images_dir),
        "augmentation_delta_labels_dir": str(augmentation_labels_dir),
        "resized_zip_path": str(resized_zip),
        "dataset_workspace": str(dataset_workspace),
        "save_period": 5,
        "hardware_name": "Tesla_T4x2_Kaggle" if IS_KAGGLE else "Colab_GPU",
        "token_path": str(
            Path("/tmp/ia_article_drive_token.json")
            if IS_KAGGLE
            else Path("/content/token.json")
        ),
        # Results and weights are condition-specific and never share folders.
        "drive_folder_id": args.drive_folder_id or destinations["results"],
        "drive_checkpoints_folder_id": (
            args.drive_checkpoints_folder_id or destinations["checkpoints"]
        ),
        "smoke_test": args.smoke_test,
        "smoke_images": args.smoke_images,
        "smoke_epochs": args.smoke_epochs,
        "c2_calibration_mode": args.c2_calibration_batch is not None,
        "c2_calibration_batch": args.c2_calibration_batch,
        "c2_selected_batch": args.c2_batch,
        "calibration_images": args.calibration_images,
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
