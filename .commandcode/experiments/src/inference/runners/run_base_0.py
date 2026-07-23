"""Base 0 Zero-Shot inference and OBB tracking runner module."""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from src.inference.base_inference import BaseInferencePipeline
from src.utils.io_manager import IOManager
from ultralytics import YOLO

_DEFAULT_BATCH_SIZE = 16


class Base0Runner(BaseInferencePipeline):
    """Zero-Shot Inference and Tracking Pipeline for Base 0."""

    def __init__(self, config: dict[str, Any], model_path: str) -> None:
        """Initialize YOLO model, IOManager, and direct DOTA -> MTC class mapping."""
        super().__init__(config, model_path)
        self.model = YOLO(model_path)
        self.io_manager = IOManager(token_path=config.get("token_path"))
        self.batch_size: int = config.get("batch_size", _DEFAULT_BATCH_SIZE)
        self.tracker: str = config.get("tracker", "bytetrack.yaml")

        # Direct class mapping: {dota_class_id: mtc_class_id}
        self.dota_to_mtc: dict[int, int] = {}
        for idx, name in self.model.names.items():
            clean = name.lower().replace("-", " ").replace("_", " ")
            if any(kw in clean for kw in ("small vehicle", "car")):
                self.dota_to_mtc[int(idx)] = 0  # MTC: car
            elif any(kw in clean for kw in ("large vehicle", "bus", "truck")):
                self.dota_to_mtc[int(idx)] = 6  # MTC: truck

        print(f"[Base0Runner] DOTA -> MTC Class Mapping: {self.dota_to_mtc}")
        print(f"[Base0Runner] Selected Tracker: {self.tracker}")

    # ===================================================================
    # Health Check
    # ===================================================================
    def health_check(self) -> bool:
        """Validate that all required resources and paths are available before execution.

        Returns:
            True if all checks pass, False if a critical resource is missing.
        """
        ok = True
        metadata_path = Path(self.config["metadata_path"])
        images_dir = Path(self.config["images_dir"])
        output_dir = Path(self.config["output_dir"])

        print("\n" + "=" * 60)
        print("HEALTH CHECK")
        print("=" * 60)

        # Metadata file check
        if metadata_path.exists():
            csv_data = self.io_manager.load_csv(metadata_path)
            val_clips = {
                r["clip_id"]
                for r in csv_data
                if r.get("split") == "val" and r.get("clip_id")
            }
            print(
                f"  ✅ Metadata:   {metadata_path} ({len(csv_data)} rows, {len(val_clips)} val clips)"
            )
        else:
            print(f"  ❌ Metadata:   {metadata_path} NOT FOUND")
            ok = False

        # Images directory check
        if images_dir.exists():
            sample = list(images_dir.iterdir())[:5]
            print(f"  ✅ Images:     {images_dir} (sample: {[s.name for s in sample]})")
        else:
            print(f"  ❌ Images:     {images_dir} NOT FOUND")
            ok = False

        # Output directory check
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Output:     {output_dir}")

        # Model check
        print(f"  ✅ Model:      {self.model_path} ({len(self.model.names)} classes)")

        # Class mapping check
        if self.dota_to_mtc:
            print(f"  ✅ Vehicles:   {len(self.dota_to_mtc)} vehicle classes mapped")
        else:
            print("  ❌ Vehicles:   No vehicle classes mapped from model")
            ok = False

        # Google Drive Service check
        if self.io_manager.drive_service:
            print("  ✅ Drive:      Service initialized")
        else:
            print("  ⚠️  Drive:      Not available (local output only)")

        # GPU check
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✅ GPU:        {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("  ⚠️  GPU:        CUDA not available (CPU fallback)")

        print("=" * 60)
        status = "PASSED ✅" if ok else "FAILED ❌"
        print(f"  Health Check Status: {status}")
        print("=" * 60 + "\n")
        return ok

    # ===================================================================
    # Track Results Processing
    # ===================================================================
    def _process_track_results(
        self, results: list, frame_offset: int
    ) -> list[dict[str, Any]]:
        """Extract data from model.track() results including assigned track_ids.

        Args:
            results: List of Results objects returned by model.track().
            frame_offset: Index of the frame within the video clip.

        Returns:
            List of frame_data dictionaries.
        """
        batch_frames: list[dict[str, Any]] = []

        for i, r in enumerate(results):
            self.accumulate_speed(r.speed)

            frame_data: dict[str, Any] = {
                "frame_idx": frame_offset + i,
                "original_shape": list(r.orig_shape),
                "speed_ms": r.speed,
                "detections": [],
            }

            if r.obb is not None and len(r.obb):
                corners = r.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
                classes = r.obb.cls.cpu().numpy().astype(int)
                scores = r.obb.conf.cpu().numpy()

                # Extract track IDs from OBB results if assigned
                track_ids = None
                if r.obb.id is not None:
                    track_ids = r.obb.id.int().cpu().numpy()

                for j in range(len(classes)):
                    mtc_id = self.dota_to_mtc.get(int(classes[j]))
                    if mtc_id is not None:
                        tid = int(track_ids[j]) if track_ids is not None else -1
                        frame_data["detections"].append(
                            {
                                "track_id": tid,
                                "class_id": mtc_id,
                                "score": round(float(scores[j]), 6),
                                "obb_corners": corners[j].flatten().tolist(),
                            }
                        )

            batch_frames.append(frame_data)

        return batch_frames

    # ===================================================================
    # Save & Upload Helper
    # ===================================================================
    def _save_and_upload(
        self, data: Any, local_path: Path, drive_folder_id: str | None
    ) -> dict[str, Any]:
        """Save data as JSON locally and optionally upload to Google Drive."""
        self.io_manager.save_json(data, local_path)
        drive_id = None
        if drive_folder_id:
            try:
                drive_id = self.io_manager.upload_file_to_drive(
                    local_path, drive_folder_id
                )
                if drive_id:
                    print(
                        f"    ✅ Uploaded to Drive: {local_path.name} (id: {drive_id})",
                        flush=True,
                    )
                else:
                    print(
                        f"    ⚠️  Drive upload returned None for {local_path.name} (Local only)",
                        flush=True,
                    )
            except Exception as e:
                print(
                    f"    ⚠️  Drive upload error for {local_path.name}: {e} (Local only)",
                    flush=True,
                )
                drive_id = None
        return {"local": str(local_path), "drive_id": drive_id}

    # ===================================================================
    # Main Execution
    # ===================================================================
    def execute(self) -> dict[str, Any]:
        """Execute tracking pipeline across sequential frames per clip."""
        self.start_hardware_monitoring()

        metadata_path = Path(self.config["metadata_path"])
        images_dir = Path(self.config["images_dir"])
        output_dir = Path(self.config["output_dir"])
        drive_folder_id: str | None = self.config.get("drive_folder_id")
        imgsz: int = self.config.get("imgsz", 640)
        max_clips: int = self.config.get("max_clips", 0)

        # 1. Fetch validation clip IDs
        csv_data = self.io_manager.load_csv(metadata_path)
        seen: set[str] = set()
        clip_ids: list[str] = []
        for row in csv_data:
            cid = row.get("clip_id", "")
            if row.get("split") == "val" and cid and cid not in seen:
                seen.add(cid)
                clip_ids.append(cid)

        if max_clips > 0:
            clip_ids = clip_ids[:max_clips]
        print(f"[Base0Runner] Processing {len(clip_ids)} clips...", flush=True)

        generated_files: list[dict[str, Any]] = []

        # 2. Process each clip with tracking and exception resilience
        try:
            for clip_num, clip_id in enumerate(clip_ids, start=1):
                clip_dir = images_dir / clip_id

                # Resume check: skip if output JSON already exists
                clip_json_path = output_dir / f"{clip_id}_predictions.json"
                if clip_json_path.exists():
                    print(
                        f"  [{clip_num}/{len(clip_ids)}] {clip_id}: already processed, skipping.",
                        flush=True,
                    )
                    generated_files.append(
                        {"local": str(clip_json_path), "drive_id": "already_uploaded"}
                    )
                    continue

                # Detect directory structure (subfolder vs flat prefix)
                if clip_dir.is_dir():
                    frame_paths = self.io_manager.list_files_in_dir(
                        clip_dir, extension=".jpg"
                    )
                else:
                    frame_paths = self.io_manager.list_files_in_dir(
                        images_dir, pattern=f"{clip_id}*.jpg"
                    )

                if not frame_paths:
                    print(
                        f"  [{clip_num}/{len(clip_ids)}] {clip_id}: no frames found, skipping.",
                        flush=True,
                    )
                    continue

                print(
                    f"  [{clip_num}/{len(clip_ids)}] {clip_id}: {len(frame_paths)} frames...",
                    flush=True,
                )

                clip_results: dict[str, Any] = {
                    "clip_id": clip_id,
                    "inference_shape": [imgsz, imgsz],
                    "frames": [],
                }

                # Sequential frame tracking using model.track(persist=True)
                for frame_idx, frame_path in enumerate(frame_paths):
                    results = self.model.track(
                        source=str(frame_path),
                        conf=self.config.get("conf", 0.001),
                        iou=self.config.get("iou", 0.45),
                        imgsz=imgsz,
                        device=self.device,
                        tracker=self.tracker,
                        persist=True,
                        verbose=False,
                    )
                    batch_frames = self._process_track_results(
                        results, frame_offset=frame_idx
                    )
                    clip_results["frames"].extend(batch_frames)

                # Reset tracker state between clips
                if (
                    hasattr(self.model, "predictor")
                    and self.model.predictor
                    and hasattr(self.model.predictor, "trackers")
                ):
                    for t in self.model.predictor.trackers:
                        t.reset()

                # Save JSON and upload to Drive
                file_record = self._save_and_upload(
                    clip_results, clip_json_path, drive_folder_id
                )
                generated_files.append(file_record)

                del clip_results
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except (KeyboardInterrupt, Exception) as e:
            print(f"\n[Base0Runner] ⚠️  Interruption detected: {type(e).__name__}: {e}")
            print("[Base0Runner] Saving partial metrics before exiting...")

        finally:
            metrics = self.record_hardware_metrics()
            metrics_path = output_dir / "inference_metrics.json"
            metrics_record = self._save_and_upload(
                metrics, metrics_path, drive_folder_id
            )
            generated_files.append(metrics_record)

            print(
                f"[Base0Runner] Finished. {self.total_frames_processed} total frames processed."
            )
            sys.stdout.flush()
            sys.stderr.flush()

        return {
            "status": "success",
            "files": generated_files,
            "metrics": metrics,
        }


# ===================================================================
# CLI Entrypoint for Execution via !python -m src.inference.runners.run_base_0
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
        if (c / "split_metadata.csv").exists():
            return c

    for root, _, files in os.walk("/kaggle/input"):
        if "split_metadata.csv" in files:
            return Path(root)

    return Path("/kaggle/input/mtc-challenge")


if __name__ == "__main__":
    IS_KAGGLE = os.path.exists("/kaggle/working")

    dataset_dir = _find_dataset_dir()
    output_dir = (
        Path("/kaggle/working/output_base0")
        if IS_KAGGLE
        else Path("/content/output_base0")
    )

    config = {
        "device": 0,
        "conf": 0.001,
        "iou": 0.45,
        "imgsz": 640,
        "batch_size": 16,
        "tracker": "bytetrack.yaml",
        "metadata_path": str(dataset_dir / "split_metadata.csv"),
        "images_dir": str(dataset_dir / "train-001" / "train"),
        "output_dir": str(output_dir),
        "hardware_name": "Tesla_T4_Kaggle" if IS_KAGGLE else "Colab_GPU",
        "experiment_condition": "Base_0_Zero_Shot",
        "token_path": str(
            Path("/kaggle/working/token.json")
            if IS_KAGGLE
            else Path("/content/token.json")
        ),
        "drive_folder_id": "1crSt8Q48JDpB_JfR2dT0mvRwjpYp8WyQ",
        "max_clips": 0,  # 0 = process all clips
    }

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory:  {output_dir}")

    runner = Base0Runner(config=config, model_path="yolo26s-obb.pt")

    if not runner.health_check():
        print("[FATAL] Health check failed. Aborting execution.")
        sys.exit(1)

    results = runner.execute()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Status: {results['status']}")
    print(f"Generated files: {len(results['files'])}")
    for f in results["files"]:
        drive_status = f"Drive ID: {f['drive_id']}" if f["drive_id"] else "Local only"
        print(f"  📄 {Path(f['local']).name} -> {drive_status}")
    print("\nHardware Metrics:")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v}")

    # Session cleanup and termination
    print("\nTerminating session in 5 seconds...")
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(5)

    if IS_KAGGLE:
        os._exit(0)
    elif os.path.exists("/content/drive"):
        from google.colab import runtime

        runtime.unassign()
