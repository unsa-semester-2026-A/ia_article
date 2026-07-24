"""Base 1 fine-tuned OBB inference and tracking runner module."""

import argparse
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
_EXPECTED_CLASS_COUNT = 9


class Base1Runner(BaseInferencePipeline):
    """Inference and tracking pipeline for Base 1 fine-tuned weights."""

    def __init__(self, config: dict[str, Any], model_path: str) -> None:
        """Initialize the trained Base 1 model and IO dependencies."""
        super().__init__(config, model_path)
        self.model = YOLO(str(model_path))
        self.io_manager = IOManager(token_path=config.get("token_path"))
        self.batch_size: int = config.get("batch_size", _DEFAULT_BATCH_SIZE)
        self.tracker: str = config.get("tracker", "bytetrack.yaml")

        print(f"[Base1Runner] Model classes: {self.model.names}")
        print(f"[Base1Runner] Selected tracker: {self.tracker}")

    def health_check(self) -> bool:
        """Validate required resources before running Base 1 inference."""
        ok = True
        metadata_path = Path(self.config["metadata_path"])
        images_dir = Path(self.config["images_dir"])
        output_dir = Path(self.config["output_dir"])

        print("\n" + "=" * 60)
        print("BASE 1 INFERENCE HEALTH CHECK")
        print("=" * 60)

        if metadata_path.exists():
            csv_data = self.io_manager.load_csv(metadata_path)
            val_clips = {
                row["clip_id"]
                for row in csv_data
                if row.get("split") == "val" and row.get("clip_id")
            }
            print(
                f"  Metadata: {metadata_path} ({len(csv_data)} rows, {len(val_clips)} val clips)"
            )
        else:
            print(f"  Metadata: {metadata_path} NOT FOUND")
            ok = False

        if images_dir.exists():
            sample = list(images_dir.iterdir())[:5]
            print(f"  Images: {images_dir} (sample: {[item.name for item in sample]})")
        else:
            print(f"  Images: {images_dir} NOT FOUND")
            ok = False

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output: {output_dir}")

        if self.model_path.exists():
            print(f"  Model: {self.model_path} ({len(self.model.names)} classes)")
        else:
            print(f"  Model: {self.model_path} NOT FOUND")
            ok = False

        if len(self.model.names) == _EXPECTED_CLASS_COUNT:
            print("  Classes: 9 SMART classes detected")
        else:
            print(
                f"  Classes: expected 9 SMART classes, found {len(self.model.names)}"
            )
            ok = False

        if self.io_manager.drive_service:
            print("  Drive: service initialized")
        else:
            print("  Drive: not available, local output only")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("  GPU: CUDA not available, CPU fallback")

        print("=" * 60)
        print(f"  Health Check Status: {'PASSED' if ok else 'FAILED'}")
        print("=" * 60 + "\n")
        return ok

    def _process_track_results(
        self, results: list, frame_offset: int
    ) -> list[dict[str, Any]]:
        """Extract OBB detections preserving Base 1 trained class IDs 0..8."""
        batch_frames: list[dict[str, Any]] = []

        for index, result in enumerate(results):
            self.accumulate_speed(result.speed)
            frame_data: dict[str, Any] = {
                "frame_idx": frame_offset + index,
                "original_shape": list(result.orig_shape),
                "speed_ms": result.speed,
                "detections": [],
            }

            if result.obb is not None and len(result.obb):
                corners = result.obb.xyxyxyxy.cpu().numpy()
                classes = result.obb.cls.cpu().numpy().astype(int)
                scores = result.obb.conf.cpu().numpy()

                track_ids = None
                if result.obb.id is not None:
                    track_ids = result.obb.id.int().cpu().numpy()

                for detection_index in range(len(classes)):
                    class_id = int(classes[detection_index])
                    if class_id < 0 or class_id >= _EXPECTED_CLASS_COUNT:
                        continue
                    track_id = (
                        int(track_ids[detection_index]) if track_ids is not None else -1
                    )
                    frame_data["detections"].append(
                        {
                            "track_id": track_id,
                            "class_id": class_id,
                            "score": round(float(scores[detection_index]), 6),
                            "obb_corners": corners[detection_index].flatten().tolist(),
                        }
                    )

            batch_frames.append(frame_data)

        return batch_frames

    def _save_and_upload(
        self, data: Any, local_path: Path, drive_folder_id: str | None
    ) -> dict[str, Any]:
        """Save JSON locally and optionally upload to Google Drive."""
        self.io_manager.save_json(data, local_path)
        drive_id = None
        if drive_folder_id:
            try:
                drive_id = self.io_manager.upload_file_to_drive(
                    local_path, drive_folder_id
                )
                print(
                    f"    Drive upload: {local_path.name} -> {drive_id or 'local only'}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"    Drive upload error for {local_path.name}: {exc} (local only)",
                    flush=True,
                )
        return {"local": str(local_path), "drive_id": drive_id}

    def execute(self) -> dict[str, Any]:
        """Execute Base 1 tracking inference across validation clips."""
        self.start_hardware_monitoring()

        metadata_path = Path(self.config["metadata_path"])
        images_dir = Path(self.config["images_dir"])
        output_dir = Path(self.config["output_dir"])
        drive_folder_id: str | None = self.config.get("drive_folder_id")
        imgsz: int = self.config.get("imgsz", 640)
        max_clips: int = self.config.get("max_clips", 0)

        csv_data = self.io_manager.load_csv(metadata_path)
        seen: set[str] = set()
        clip_ids: list[str] = []
        for row in csv_data:
            clip_id = row.get("clip_id", "")
            if row.get("split") == "val" and clip_id and clip_id not in seen:
                seen.add(clip_id)
                clip_ids.append(clip_id)

        if max_clips > 0:
            clip_ids = clip_ids[:max_clips]
        print(f"[Base1Runner] Processing {len(clip_ids)} clips...", flush=True)

        generated_files: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}

        try:
            for clip_number, clip_id in enumerate(clip_ids, start=1):
                clip_dir = images_dir / clip_id
                clip_json_path = output_dir / f"{clip_id}_predictions.json"

                if clip_json_path.exists():
                    print(
                        f"  [{clip_number}/{len(clip_ids)}] {clip_id}: already processed, skipping.",
                        flush=True,
                    )
                    generated_files.append(
                        {"local": str(clip_json_path), "drive_id": "already_uploaded"}
                    )
                    continue

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
                        f"  [{clip_number}/{len(clip_ids)}] {clip_id}: no frames found, skipping.",
                        flush=True,
                    )
                    continue

                print(
                    f"  [{clip_number}/{len(clip_ids)}] {clip_id}: {len(frame_paths)} frames...",
                    flush=True,
                )

                clip_results: dict[str, Any] = {
                    "clip_id": clip_id,
                    "inference_shape": [imgsz, imgsz],
                    "frames": [],
                }

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
                    clip_results["frames"].extend(
                        self._process_track_results(results, frame_offset=frame_idx)
                    )

                self._reset_trackers()

                file_record = self._save_and_upload(
                    clip_results,
                    clip_json_path,
                    drive_folder_id,
                )
                generated_files.append(file_record)

                del clip_results
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except (KeyboardInterrupt, Exception) as exc:
            print(
                f"\n[Base1Runner] Interruption detected: {type(exc).__name__}: {exc}",
                flush=True,
            )
            print("[Base1Runner] Saving partial metrics before exiting...", flush=True)

        finally:
            metrics = self.record_hardware_metrics()
            metrics_path = output_dir / "inference_metrics.json"
            generated_files.append(
                self._save_and_upload(metrics, metrics_path, drive_folder_id)
            )
            print(
                f"[Base1Runner] Finished. {self.total_frames_processed} total frames processed.",
                flush=True,
            )
            sys.stdout.flush()
            sys.stderr.flush()

        return {"status": "success", "files": generated_files, "metrics": metrics}

    def _reset_trackers(self) -> None:
        """Reset tracker state between clips."""
        if (
            hasattr(self.model, "predictor")
            and self.model.predictor
            and hasattr(self.model.predictor, "trackers")
        ):
            for tracker in self.model.predictor.trackers:
                tracker.reset()


def _find_dataset_dir() -> Path:
    """Automatically discover dataset path in Kaggle or Colab environments."""
    if not os.path.exists("/kaggle/working"):
        return Path("/content/drive/MyDrive/ia_article")

    candidates = [
        Path("/kaggle/input/datasets/alvaroquispeunsa/mtc-challenge"),
        Path("/kaggle/input/mtc-challenge"),
    ]
    for candidate in candidates:
        if (candidate / "split_metadata.csv").exists():
            return candidate

    for root, _, files in os.walk("/kaggle/input"):
        if "split_metadata.csv" in files:
            return Path(root)

    return Path("/kaggle/input/mtc-challenge")


def _find_best_pt() -> Path:
    """Find a Base 1 best.pt checkpoint in common notebook locations."""
    candidates = [
        Path("/kaggle/working/best.pt"),
        Path("/content/best.pt"),
        Path(".noutil/best.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if "best.pt" in files:
                return Path(root) / "best.pt"

    return Path("best.pt")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Base 1 inference."""
    parser = argparse.ArgumentParser(description="Base 1 YOLO OBB inference runner")
    parser.add_argument("--model-path", default=None, help="Path to Base 1 best.pt")
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--drive-folder-id", default="1uxxZHpgK-gVBc3FNKEXQ1c64G7xlLGEi")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    is_kaggle = os.path.exists("/kaggle/working")

    dataset_dir = _find_dataset_dir()
    output_dir = Path(
        args.output_dir
        or ("/kaggle/working/output_base1" if is_kaggle else "/content/output_base1")
    )
    model_path = Path(args.model_path) if args.model_path else _find_best_pt()

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
        "hardware_name": "Tesla_T4_Kaggle" if is_kaggle else "Colab_GPU",
        "experiment_condition": "Base_1_Raw_Data",
        "token_path": str(
            Path("/kaggle/working/token.json")
            if is_kaggle
            else Path("/content/token.json")
        ),
        "drive_folder_id": args.drive_folder_id,
        "max_clips": args.max_clips,
    }

    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Model path:        {model_path}")

    runner = Base1Runner(config=config, model_path=str(model_path))
    if not runner.health_check():
        print("[FATAL] Health check failed. Aborting execution.")
        sys.exit(1)

    results = runner.execute()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Status: {results['status']}")
    print(f"Generated files: {len(results['files'])}")
    for file_record in results["files"]:
        drive_status = (
            f"Drive ID: {file_record['drive_id']}"
            if file_record["drive_id"]
            else "Local only"
        )
        print(f"  {Path(file_record['local']).name} -> {drive_status}")
    print("\nHardware Metrics:")
    for key, value in results["metrics"].items():
        print(f"  {key}: {value}")

    print("\nTerminating session in 5 seconds...")
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(5)

    if is_kaggle:
        os._exit(0)
    elif os.path.exists("/content/drive"):
        from google.colab import runtime

        runtime.unassign()
