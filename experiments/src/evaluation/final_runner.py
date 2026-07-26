"""Reproducible final evaluation runner for every available YOLO condition.

The runner deliberately separates *where a checkpoint comes from* from *where
its results are published*.  A missing optional checkpoint is reported and does
not prevent already available conditions from being evaluated.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from src.evaluation.metric import CLASS_IDS, RIOU_THRESHOLDS
from src.evaluation.motion_filter import Detection
from src.evaluation.pipeline import (
    INFERENCE_CONFIDENCE,
    PipelineEvaluation,
    estimate_clip_homographies,
    evaluate_dataset,
    load_ground_truth_csv,
    split_frame_id,
)
from src.utils.gpu_monitor import GpuSampler
from src.utils.io_manager import IOManager

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    """One independently reproducible model condition."""

    name: str
    result_folder_id: str
    checkpoint_folder_id: str | None = None
    checkpoint_name: str | None = None
    local_weights: str | None = None
    enabled: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "ConditionSpec":
        """Build a condition specification from its JSON representation."""
        return cls(
            name=str(raw["name"]),
            result_folder_id=str(raw["result_folder_id"]),
            checkpoint_folder_id=_optional_str(raw.get("checkpoint_folder_id")),
            checkpoint_name=_optional_str(raw.get("checkpoint_name")),
            local_weights=_optional_str(raw.get("local_weights")),
            enabled=bool(raw.get("enabled", True)),
        )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Filesystem and execution contract for a final evaluation run."""

    images_dir: Path
    validation_labels_dir: Path
    ground_truth_csv: Path
    output_dir: Path
    conditions: tuple[ConditionSpec, ...]
    token_path: Path | None = None
    batch_size: int = 16
    gpu_sample_interval_seconds: float = 2.0
    max_frames: int | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "EvaluationConfig":
        """Build a validated runner configuration from its JSON representation."""
        conditions_raw = raw.get("conditions", [])
        if not isinstance(conditions_raw, list):
            raise ValueError("conditions must be a JSON list")
        return cls(
            images_dir=Path(str(raw["images_dir"])),
            validation_labels_dir=Path(str(raw["validation_labels_dir"])),
            ground_truth_csv=Path(str(raw["ground_truth_csv"])),
            output_dir=Path(str(raw["output_dir"])),
            conditions=tuple(
                ConditionSpec.from_mapping(item)
                for item in conditions_raw
                if isinstance(item, Mapping)
            ),
            token_path=(
                Path(str(raw["token_path"])) if raw.get("token_path") else None
            ),
            batch_size=int(raw.get("batch_size", 16)),
            gpu_sample_interval_seconds=float(
                raw.get("gpu_sample_interval_seconds", 2.0)
            ),
            max_frames=(int(raw["max_frames"]) if raw.get("max_frames") else None),
        )


def _optional_str(value: object) -> str | None:
    return str(value) if value not in (None, "") else None


def default_conditions() -> tuple[ConditionSpec, ...]:
    """Return the known F1 destinations; unavailable weights are skipped safely."""
    return (
        ConditionSpec("Base 0", "1qgkdNe8_3IbeNlf7u-8NNvh9whLIR-WB", enabled=False),
        ConditionSpec(
            "Base 1",
            "1Roazcv3c72jGmGy2M0nrlx5kL75-BY3m",
            "1pn8OzJX_kctgluEZkSC6WEfbSCaPyKMa",
            "f1_c1_best.pt",
        ),
        ConditionSpec(
            "Base 2",
            "1ZX8FbFNx9wELzK41AEp2xiMJzkIrsns4",
            "1navKsrapRDxJzbLVrDIHpmDbkU4zHtVN",
            "f1_c2_best.pt",
        ),
        ConditionSpec(
            "Mejora A",
            "1Sf2SEMHhl3jQoqnGbXiuVXd_BpT6yBgf",
            "1Hi8OmTIMNzLfadjFbk79OL8yiIZewhpz",
            "f1_c3_best.pt",
        ),
        ConditionSpec("Mejora B", "1Spi8ArWMV3E9Av8mbNp6xt4xzUhWQaXv", enabled=False),
        ConditionSpec("Mejora C", "1P5CzJwUZAxETNkWm4F6_LUzChGvKRr_Q", enabled=False),
    )


def sha256_file(path: Path) -> str:
    """Calculate a stable checksum without loading a checkpoint into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_frame_ids(labels_dir: Path, max_frames: int | None = None) -> set[str]:
    """Use the immutable validation-label split as the anti-leakage manifest."""
    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"validation label directory does not exist: {labels_dir}"
        )
    frame_ids = {path.stem for path in labels_dir.glob("*.txt")}
    if not frame_ids:
        raise ValueError(f"no validation labels found in {labels_dir}")
    for frame_id in frame_ids:
        split_frame_id(frame_id)
    if max_frames is None:
        return frame_ids
    if max_frames <= 0:
        raise ValueError("max_frames must be positive when configured")
    return set(sorted(frame_ids, key=split_frame_id)[:max_frames])


def collect_validation_frame_paths(
    images_dir: Path, frame_ids: set[str]
) -> dict[str, dict[str, Path]]:
    """Build a memory-safe mapping of manifest-selected frame paths by clip."""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory does not exist: {images_dir}")
    paths: dict[str, Path] = {}
    for path in images_dir.rglob("*"):
        if path.suffix.lower() not in IMAGE_SUFFIXES or path.stem not in frame_ids:
            continue
        if path.stem in paths:
            raise ValueError(f"duplicate image stem in dataset: {path.stem}")
        paths[path.stem] = path
    missing = sorted(frame_ids.difference(paths))
    if missing:
        raise FileNotFoundError(f"missing validation images (first 10): {missing[:10]}")

    grouped: dict[str, dict[str, Path]] = {}
    for frame_id in sorted(frame_ids, key=lambda item: split_frame_id(item)):
        clip_id, _ = split_frame_id(frame_id)
        grouped.setdefault(clip_id, {})[frame_id] = paths[frame_id]
    return grouped


def load_clip_frames(frame_paths: Mapping[str, Path]) -> dict[str, np.ndarray]:
    """Read one temporal clip, keeping the full validation split out of RAM."""
    frames: dict[str, np.ndarray] = {}
    for frame_id, path in sorted(
        frame_paths.items(), key=lambda item: split_frame_id(item[0])
    ):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read validation image: {path}")
        frames[frame_id] = image
    return frames


class _TimingModel:
    """Measure Ultralytics per-result speed without coupling the pipeline to it."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.speed_samples: list[dict[str, float]] = []

    def predict(self, source: Sequence[np.ndarray], **kwargs: Any) -> Sequence[Any]:
        results = self.model.predict(source, **kwargs)
        for result in results:
            speed = getattr(result, "speed", {}) or {}
            self.speed_samples.append(
                {
                    key: float(speed.get(key, 0.0))
                    for key in ("preprocess", "inference", "postprocess")
                }
            )
        return results


def _latency_summary(
    samples: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate Ultralytics millisecond timings into paper-ready statistics."""
    summary: dict[str, dict[str, float]] = {}
    for field in ("preprocess", "inference", "postprocess"):
        values = np.asarray([sample[field] for sample in samples], dtype=np.float64)
        summary[field] = {
            "mean_ms": float(np.mean(values)) if len(values) else 0.0,
            "median_ms": float(np.median(values)) if len(values) else 0.0,
            "p95_ms": float(np.percentile(values, 95)) if len(values) else 0.0,
            "std_ms": float(np.std(values)) if len(values) else 0.0,
        }
    return summary


def _serialize_predictions(
    predictions: Mapping[str, list[Detection]],
) -> list[dict[str, object]]:
    return [
        {"frame_id": frame_id, "detections": [asdict(item) for item in detections]}
        for frame_id, detections in sorted(predictions.items())
    ]


def _write_json_gz(path: Path, value: object) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)


def _write_metric_csvs(directory: Path, evaluation: PipelineEvaluation) -> None:
    details = evaluation.metric_details
    with (directory / "metrics_by_class.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "ap_mean_riou"])
        writer.writeheader()
        for class_id in CLASS_IDS:
            writer.writerow(
                {"class_id": class_id, "ap_mean_riou": details["ap_by_class"][class_id]}
            )
    with (directory / "metrics_by_threshold.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["class_id", "riou", "ap", "tp", "fp", "fn"]
        )
        writer.writeheader()
        for class_id in CLASS_IDS:
            for threshold in RIOU_THRESHOLDS:
                counts = details["counts"][class_id][threshold]
                writer.writerow(
                    {
                        "class_id": class_id,
                        "riou": threshold,
                        "ap": details["ap_by_class_threshold"][class_id][threshold],
                        **counts,
                    }
                )


def _package_result(
    config: EvaluationConfig,
    condition: ConditionSpec,
    raw_predictions: Mapping[str, list[Detection]],
    homographies: Mapping[str, Mapping[str, np.ndarray | None]],
    evaluation: PipelineEvaluation,
    hardware: Mapping[str, object],
    weight_path: Path,
    weight_sha256: str,
) -> Path:
    """Build one self-contained ZIP and leave no loose final artifacts behind."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    condition_slug = condition.name.lower().replace(" ", "_")
    run_root = config.output_dir / f"{condition_slug}_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    git_revision = _git_revision()
    report = {
        "condition": condition.name,
        "metric_name": "Macro AP-rIoU",
        "macro_ap_riou": evaluation.macro_score,
        "macro_ap_at_050": float(
            np.mean(
                [
                    evaluation.metric_details["ap_by_class_threshold"][class_id][0.5]
                    for class_id in CLASS_IDS
                ]
            )
        ),
        "macro_ap_at_080": float(
            np.mean(
                [
                    evaluation.metric_details["ap_by_class_threshold"][class_id][0.8]
                    for class_id in CLASS_IDS
                ]
            )
        ),
        "motion_by_clip": {
            clip: asdict(value) for clip, value in evaluation.motion_by_clip.items()
        },
        "metric_details": evaluation.metric_details,
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": timestamp,
        "condition": asdict(condition),
        "weight": {
            "local_name": weight_path.name,
            "sha256": weight_sha256,
            "bytes": weight_path.stat().st_size,
        },
        "code_revision": git_revision,
        "input": {
            "images_dir": str(config.images_dir),
            "validation_labels_dir": str(config.validation_labels_dir),
            "ground_truth_csv": str(config.ground_truth_csv),
            "frame_count": len(raw_predictions),
            "is_smoke_run": config.max_frames is not None,
        },
        "inference": {
            "confidence": INFERENCE_CONFIDENCE,
            "batch_size": config.batch_size,
            "homographies": "shared_from_frames_without_model_masks",
        },
    }
    (run_root / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_root / "hardware.json").write_text(
        json.dumps(hardware, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_json_gz(
        run_root / "predictions_raw.json.gz", _serialize_predictions(raw_predictions)
    )
    _write_json_gz(
        run_root / "predictions_filtered.json.gz",
        _serialize_predictions(evaluation.filtered_predictions),
    )
    _write_json_gz(
        run_root / "homographies.json.gz",
        {
            clip: {
                frame: None if value is None else np.asarray(value).tolist()
                for frame, value in frames.items()
            }
            for clip, frames in homographies.items()
        },
    )
    _write_metric_csvs(run_root, evaluation)
    archive = (
        config.output_dir
        / f"final_evaluation_{condition_slug}_{weight_sha256[:8]}_{git_revision[:8]}_{timestamp}.zip"
    )
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as handle:
        for path in sorted(run_root.iterdir()):
            handle.write(path, arcname=path.name)
    shutil.rmtree(run_root)
    return archive


def _git_revision() -> str:
    """Return the checked-out revision when the runner is launched from the repo."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _resolve_weight(
    condition: ConditionSpec, io_manager: IOManager | None, directory: Path
) -> Path | None:
    if not condition.enabled:
        return None
    if condition.local_weights:
        path = Path(condition.local_weights)
        if not path.is_file():
            raise FileNotFoundError(f"local checkpoint does not exist: {path}")
        return path
    if (
        not io_manager
        or not condition.checkpoint_folder_id
        or not condition.checkpoint_name
    ):
        return None
    return io_manager.download_file_from_drive(
        condition.checkpoint_name,
        condition.checkpoint_folder_id,
        directory / condition.checkpoint_name,
    )


def run_final_evaluation(
    config: EvaluationConfig,
    model_loader: Callable[[Path], Any] | None = None,
) -> dict[str, object]:
    """Evaluate every resolvable condition and upload one ZIP per success."""
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = validation_frame_ids(config.validation_labels_dir, config.max_frames)
    frame_paths_by_clip = collect_validation_frame_paths(config.images_dir, frame_ids)
    ground_truths = load_ground_truth_csv(config.ground_truth_csv, frame_ids)
    homographies_by_clip = {
        clip: estimate_clip_homographies(load_clip_frames(paths))
        for clip, paths in frame_paths_by_clip.items()
    }
    io_manager = (
        IOManager(config.token_path)
        if any(item.checkpoint_folder_id for item in config.conditions if item.enabled)
        else None
    )
    if model_loader is None:
        from ultralytics import YOLO

        def model_loader(path: Path) -> Any:
            """Load one Ultralytics checkpoint lazily in cloud runtimes."""
            return YOLO(str(path))

    completed: dict[str, object] = {}
    skipped: dict[str, str] = {}
    for condition in config.conditions:
        weight_path = _resolve_weight(
            condition, io_manager, config.output_dir / "weights"
        )
        if weight_path is None:
            skipped[condition.name] = "disabled or checkpoint not configured/available"
            continue
        timing_model = _TimingModel(model_loader(weight_path))
        sampler = GpuSampler(config.gpu_sample_interval_seconds)
        sampler.start()
        started = time.perf_counter()
        raw_by_clip: dict[str, dict[str, list[Detection]]] = {}
        try:
            from src.evaluation.pipeline import infer_clip

            for clip_id, clip_paths in sorted(frame_paths_by_clip.items()):
                clip_frames = load_clip_frames(clip_paths)
                raw_by_clip[clip_id], _ = infer_clip(
                    timing_model,
                    clip_frames,
                    config.batch_size,
                    homographies=homographies_by_clip[clip_id],
                )
        finally:
            elapsed = time.perf_counter() - started
            gpu_sampling = sampler.stop()
        evaluation = evaluate_dataset(raw_by_clip, homographies_by_clip, ground_truths)
        raw_by_frame = {
            frame: detections
            for clips in raw_by_clip.values()
            for frame, detections in clips.items()
        }
        hardware = {
            "platform": platform.platform(),
            "python": sys.version,
            "elapsed_seconds": elapsed,
            "end_to_end_fps": len(raw_by_frame) / elapsed if elapsed else 0.0,
            "peak_cpu_ram_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 * 1024),
            "gpu_sampling": gpu_sampling,
            "latency": _latency_summary(timing_model.speed_samples),
        }
        checksum = sha256_file(weight_path)
        archive = _package_result(
            config,
            condition,
            raw_by_frame,
            homographies_by_clip,
            evaluation,
            hardware,
            weight_path,
            checksum,
        )
        drive_id = None
        if io_manager:
            drive_id = io_manager.upload_file_to_drive(
                archive,
                condition.result_folder_id,
                mime_type="application/zip",
                remote_name=archive.name,
            )
            if drive_id is None:
                raise RuntimeError(
                    f"could not upload final archive for {condition.name}"
                )
        completed[condition.name] = {
            "archive": str(archive),
            "drive_id": drive_id,
            "macro_ap_riou": evaluation.macro_score,
        }
    return {"completed": completed, "skipped": skipped}


def main() -> int:
    """Run from Kaggle, Colab, or local after writing a JSON configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    print(
        json.dumps(run_final_evaluation(EvaluationConfig.from_mapping(raw)), indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
