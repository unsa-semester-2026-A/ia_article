"""End-to-end inference adaptation, motion filtering, and evaluation pipeline."""

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray
from src.evaluation.metric import (
    OBB,
    DetailedResults,
    GroundTruthsByClassFrame,
    PredictionsByClass,
    compute_macro_ap_riou,
)
from src.evaluation.motion_filter import (
    Detection,
    HomographiesByFrame,
    MotionFilterDiagnostics,
    PredictionsByFrame,
    estimate_homography,
    filter_static_predictions,
)

INFERENCE_CONFIDENCE = 0.001
INTERNAL_TO_OFFICIAL_CLASS_IDS: dict[int, int] = {
    internal_id: internal_id + 1 for internal_id in range(9)
}

PredictionsByClip: TypeAlias = dict[str, PredictionsByFrame]
HomographiesByClip: TypeAlias = dict[str, HomographiesByFrame]
ConditionsPredictions: TypeAlias = dict[str, PredictionsByClip]
Image: TypeAlias = NDArray[np.uint8]


class TensorLike(Protocol):
    """Minimal tensor surface used from Torch without importing Torch."""

    def cpu(self) -> "TensorLike":
        """Return a CPU-backed tensor."""
        ...

    def numpy(self) -> NDArray[np.generic]:
        """Convert the tensor to NumPy."""
        ...


class OBBResultLike(Protocol):
    """Minimal Ultralytics OBB collection used by the adapter."""

    xywhr: TensorLike
    conf: TensorLike
    cls: TensorLike


class YoloResultLike(Protocol):
    """Minimal Ultralytics result surface used by the adapter."""

    obb: OBBResultLike | None


class YoloModelLike(Protocol):
    """Minimal Ultralytics model prediction surface used by clip inference."""

    def predict(
        self,
        source: Sequence[Image],
        *,
        conf: float,
        batch: int,
        verbose: bool,
        classes: Sequence[int] | None = None,
    ) -> Sequence[YoloResultLike]:
        """Run OBB inference over an ordered image batch."""
        ...


@dataclass(frozen=True, slots=True)
class PipelineEvaluation:
    """Final score and diagnostics for one complete validation dataset."""

    macro_score: float
    filtered_predictions: PredictionsByFrame
    motion_by_clip: dict[str, MotionFilterDiagnostics]
    metric_details: DetailedResults


def split_frame_id(frame_id: str) -> tuple[str, int]:
    """Split a SMART frame ID into clip identifier and numeric frame index."""
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("frame_id must be a non-empty string")
    clip_id, separator, frame_index = frame_id.rpartition("_")
    if not separator or not clip_id or not frame_index.isdigit():
        raise ValueError("frame_id must end with an underscore and numeric index")
    return clip_id, int(frame_index)


def adapt_yolo_obb_arrays(
    frame_id: str,
    xywhr: NDArray[np.float64],
    scores: NDArray[np.float64],
    class_ids: NDArray[np.int64],
    class_id_map: Mapping[int, int] = INTERNAL_TO_OFFICIAL_CLASS_IDS,
) -> list[Detection]:
    """Convert pixel-space Ultralytics OBB arrays into filter detections.

    Ultralytics represents the angle in radians and uses zero-based class IDs;
    the public SMART contract uses degrees and official IDs 1 through 9.

    Args:
        frame_id: Source frame identifier, validated for traceability.
        xywhr: Array shaped ``(N, 5)`` with angle in radians.
        scores: Confidence array shaped ``(N,)``.
        class_ids: Internal class array shaped ``(N,)``.
        class_id_map: Explicit internal-to-official class mapping.

    Returns:
        Raw detections suitable for ``motion_filter.py``.

    Raises:
        ValueError: If shapes, classes, scores, or OBB values are invalid.
    """
    split_frame_id(frame_id)
    boxes = np.asarray(xywhr, dtype=np.float64)
    confidences = np.asarray(scores, dtype=np.float64).reshape(-1)
    internal_classes = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError("xywhr must have shape (N, 5)")
    if len(boxes) != len(confidences) or len(boxes) != len(internal_classes):
        raise ValueError("xywhr, scores, and class_ids must have equal lengths")
    if not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(confidences)):
        raise ValueError("YOLO OBB arrays must contain finite values")

    detections: list[Detection] = []
    for box, score, internal_class in zip(
        boxes, confidences, internal_classes, strict=True
    ):
        normalized_internal_id = int(internal_class)
        if normalized_internal_id not in class_id_map:
            raise ValueError(f"unmapped internal class ID: {normalized_internal_id}")
        official_class_id = int(class_id_map[normalized_internal_id])
        normalized_score = float(score)
        cx, cy, width, height, angle_rad = (float(value) for value in box)
        if official_class_id not in range(1, 10):
            raise ValueError("mapped class IDs must be official IDs 1 through 9")
        if not 0.0 <= normalized_score <= 1.0:
            raise ValueError("YOLO confidence must be within [0, 1]")
        # Ultralytics may occasionally emit a non-positive OBB dimension after
        # geometric clipping/NMS. Such a box cannot overlap a ground-truth OBB
        # and is therefore a non-detection, rather than malformed evaluation
        # input.
        if width <= 0.0 or height <= 0.0:
            continue
        detections.append(
            Detection(
                official_class_id,
                normalized_score,
                cx,
                cy,
                width,
                height,
                math.degrees(angle_rad),
            )
        )
    return detections


def adapt_ultralytics_result(
    frame_id: str,
    result: YoloResultLike,
    class_id_map: Mapping[int, int] = INTERNAL_TO_OFFICIAL_CLASS_IDS,
) -> list[Detection]:
    """Convert one Ultralytics ``Results.obb`` object without Torch coupling."""
    if result.obb is None:
        return []
    boxes = np.asarray(result.obb.xywhr.cpu().numpy(), dtype=np.float64)
    scores = np.asarray(result.obb.conf.cpu().numpy(), dtype=np.float64)
    classes = np.asarray(result.obb.cls.cpu().numpy(), dtype=np.int64)
    return adapt_yolo_obb_arrays(frame_id, boxes, scores, classes, class_id_map)


def infer_clip(
    model: YoloModelLike,
    frames_by_id: Mapping[str, Image],
    batch_size: int = 16,
    confidence: float = INFERENCE_CONFIDENCE,
    class_id_map: Mapping[int, int] = INTERNAL_TO_OFFICIAL_CLASS_IDS,
    allowed_model_class_ids: Sequence[int] | None = None,
    homographies: HomographiesByFrame | None = None,
) -> tuple[PredictionsByFrame, HomographiesByFrame]:
    """Run batch OBB inference and return model-independent homographies.

    The camera transform must be identical for every experimental condition.
    It is therefore estimated from the input frames only, rather than from a
    condition-specific detection mask.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and within [0, 1]")
    ordered_frame_ids = sorted(frames_by_id, key=lambda item: split_frame_id(item)[1])
    images = [
        np.asarray(frames_by_id[frame_id], dtype=np.uint8)
        for frame_id in ordered_frame_ids
    ]
    if any(image.ndim != 3 or image.shape[2] != 3 for image in images):
        raise ValueError("inference frames must be BGR images shaped (H, W, 3)")
    if not images:
        return {}, {}

    prediction_kwargs: dict[str, object] = {
        "conf": confidence,
        "batch": batch_size,
        "verbose": False,
    }
    if allowed_model_class_ids is not None:
        allowed = sorted({int(class_id) for class_id in allowed_model_class_ids})
        if not allowed:
            raise ValueError("allowed_model_class_ids must not be empty")
        prediction_kwargs["classes"] = allowed
    results = list(model.predict(images, **prediction_kwargs))
    if len(results) != len(images):
        raise ValueError("YOLO must return exactly one result per input frame")

    predictions: PredictionsByFrame = {}
    for frame_id, image, result in zip(ordered_frame_ids, images, results, strict=True):
        detections = adapt_ultralytics_result(frame_id, result, class_id_map)
        predictions[frame_id] = detections
    return (
        predictions,
        homographies
        if homographies is not None
        else estimate_clip_homographies(frames_by_id),
    )


def estimate_clip_homographies(
    frames_by_id: Mapping[str, Image],
) -> HomographiesByFrame:
    """Estimate one shared sequence of camera transforms for a clip.

    No model predictions enter this calculation. This deliberately trades a
    detector-specific vehicle mask for a fair and reproducible transform that
    can be reused across Base and Improvement conditions.
    """
    ordered_frame_ids = sorted(frames_by_id, key=lambda item: split_frame_id(item)[1])
    homographies: HomographiesByFrame = {}
    previous_gray: Image | None = None
    for frame_id in ordered_frame_ids:
        image = np.asarray(frames_by_id[frame_id], dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("inference frames must be BGR images shaped (H, W, 3)")
        current_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if previous_gray is not None:
            homography, _ = estimate_homography(previous_gray, current_gray, [])
            homographies[frame_id] = homography
        previous_gray = current_gray
    return homographies


def group_predictions_by_clip(
    predictions_by_frame: PredictionsByFrame,
) -> PredictionsByClip:
    """Group a dataset prediction mapping into temporally independent clips."""
    grouped: PredictionsByClip = {}
    for frame_id, detections in predictions_by_frame.items():
        clip_id, _ = split_frame_id(frame_id)
        grouped.setdefault(clip_id, {})[frame_id] = list(detections)
    return grouped


def predictions_for_metric(
    predictions_by_frame: PredictionsByFrame,
) -> PredictionsByClass:
    """Convert filtered detections into the public metric input format."""
    predictions: PredictionsByClass = {}
    for frame_id, detections in predictions_by_frame.items():
        for detection in detections:
            predictions.setdefault(detection.class_id, []).append(
                (
                    frame_id,
                    detection.score,
                    detection.cx,
                    detection.cy,
                    detection.width,
                    detection.height,
                    detection.angle_deg,
                )
            )
    return predictions


def parse_ground_truth_target(target: str) -> list[tuple[int, OBB]]:
    """Parse one SMART CSV Target cell into official class and OBB tuples."""
    normalized = target.strip()
    if not normalized or normalized.lower() == "none":
        return []
    parsed: list[tuple[int, OBB]] = []
    for annotation in normalized.split(";"):
        fields = annotation.strip().split()
        if len(fields) != 6:
            raise ValueError("each ground-truth annotation must contain six fields")
        class_id = int(fields[0])
        values = tuple(float(value) for value in fields[1:])
        if class_id not in range(1, 10):
            raise ValueError("ground-truth class IDs must be from 1 through 9")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ground-truth OBB values must be finite")
        cx, cy, width, height, angle_deg = values
        if width <= 0.0 or height <= 0.0:
            raise ValueError("ground-truth dimensions must be greater than zero")
        parsed.append((class_id, (cx, cy, width, height, angle_deg)))
    return parsed


def load_ground_truth_csv(
    csv_path: str | Path,
    frame_ids: set[str] | None = None,
) -> GroundTruthsByClassFrame:
    """Load selected SMART ``Id,Target`` rows into the metric GT contract."""
    ground_truths: GroundTruthsByClassFrame = {}
    with Path(csv_path).open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not {"Id", "Target"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("ground-truth CSV must contain Id and Target columns")
        for row in reader:
            frame_id = row["Id"]
            split_frame_id(frame_id)
            if frame_ids is not None and frame_id not in frame_ids:
                continue
            for class_id, obb in parse_ground_truth_target(row["Target"]):
                ground_truths.setdefault(class_id, {}).setdefault(frame_id, []).append(
                    obb
                )
    return ground_truths


def evaluate_dataset(
    predictions_by_clip: PredictionsByClip,
    homographies_by_clip: HomographiesByClip,
    ground_truths: GroundTruthsByClassFrame,
) -> PipelineEvaluation:
    """Apply the same motion filter per clip and compute one global metric."""
    filtered_dataset: PredictionsByFrame = {}
    motion_by_clip: dict[str, MotionFilterDiagnostics] = {}
    for clip_id in sorted(predictions_by_clip):
        if clip_id not in homographies_by_clip:
            raise ValueError(f"missing homographies for clip: {clip_id}")
        filtered_clip, diagnostics = filter_static_predictions(
            predictions_by_clip[clip_id],
            homographies_by_clip[clip_id],
        )
        overlap = set(filtered_dataset).intersection(filtered_clip)
        if overlap:
            raise ValueError(f"duplicate frame IDs across clips: {sorted(overlap)}")
        filtered_dataset.update(filtered_clip)
        motion_by_clip[clip_id] = diagnostics

    macro_score, metric_details = compute_macro_ap_riou(
        predictions_for_metric(filtered_dataset),
        ground_truths,
    )
    return PipelineEvaluation(
        macro_score,
        filtered_dataset,
        motion_by_clip,
        metric_details,
    )


def evaluate_conditions(
    conditions: ConditionsPredictions,
    homographies_by_clip: HomographiesByClip,
    ground_truths: GroundTruthsByClassFrame,
) -> dict[str, PipelineEvaluation]:
    """Evaluate any number of model conditions through one identical filter."""
    return {
        condition_name: evaluate_dataset(
            predictions_by_clip,
            homographies_by_clip,
            ground_truths,
        )
        for condition_name, predictions_by_clip in sorted(conditions.items())
    }


def build_synthetic_pipeline_case() -> tuple[
    PredictionsByClip, HomographiesByClip, GroundTruthsByClassFrame
]:
    """Create the deterministic clip validated in the motion-filter notebook."""
    clip_id = "clip_demo"
    frame_ids = [f"{clip_id}_{index:04d}" for index in range(12)]
    camera_step = np.asarray(
        ((1.0, 0.0, 2.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    predictions: PredictionsByFrame = {}
    ground_truths: GroundTruthsByClassFrame = {2: {}}
    for index, frame_id in enumerate(frame_ids):
        static = Detection(1, 0.95, 100.0 + 2.0 * index, 120.0, 40.0, 20.0, 0.0)
        moving = Detection(2, 0.90, 250.0 + 5.0 * index, 180.0, 40.0, 20.0, 0.0)
        detections = [static, moving]
        if index < 6:
            detections.append(
                Detection(3, 0.85, 380.0 + 2.0 * index, 240.0, 40.0, 20.0, 0.0)
            )
        predictions[frame_id] = detections
        ground_truths[2][frame_id] = [
            (moving.cx, moving.cy, moving.width, moving.height, moving.angle_deg)
        ]
    homographies = {frame_id: camera_step for frame_id in frame_ids[1:]}
    return {clip_id: predictions}, {clip_id: homographies}, ground_truths


def run_synthetic_pipeline() -> PipelineEvaluation:
    """Execute the deterministic end-to-end integration without external data."""
    predictions, homographies, ground_truths = build_synthetic_pipeline_case()
    return evaluate_dataset(predictions, homographies, ground_truths)


def write_evaluation_report(
    report_path: str | Path,
    evaluations: Mapping[str, PipelineEvaluation],
) -> Path:
    """Write small JSON diagnostics suitable for local disk or Drive upload."""
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        condition: {
            "macro_score": evaluation.macro_score,
            "motion_by_clip": {
                clip_id: asdict(diagnostics)
                for clip_id, diagnostics in evaluation.motion_by_clip.items()
            },
            "metric_details": evaluation.metric_details,
        }
        for condition, evaluation in evaluations.items()
    }
    destination.write_text(
        json.dumps(serializable, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination
