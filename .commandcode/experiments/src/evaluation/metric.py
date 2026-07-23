"""Exact Macro AP-rIoU metric for oriented vehicle detection.

Scientific rationale:
    DOTA motivates oriented boxes for aerial objects with arbitrary headings,
    scales, and dense spatial distributions. Yang et al. show that localization
    of rotated boxes is strongly coupled to angle and aspect ratio: a small
    angular error can seriously reduce high-precision detection quality for an
    elongated object. These findings support polygon-based rIoU and the angular
    deviation tests in this module.

    The greedy matching rules, seven rIoU thresholds, nine-class macro average,
    and COCO-style 101-point interpolation are challenge-specific requirements
    defined in ``02_metric.md``; they are not attributed to the two papers.

References:
    Ding et al. (2022), "Object Detection in Aerial Images: A Large-Scale
    Benchmark and Challenges", doi:10.1109/TPAMI.2021.3117983.
    Yang et al. (2021), "Learning High-Precision Bounding Box for Rotated
    Object Detection via Kullback-Leibler Divergence", arXiv:2106.01883.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

CLASS_IDS: tuple[int, ...] = tuple(range(1, 10))
RIOU_THRESHOLDS: tuple[float, ...] = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
)

# Parametric oriented bounding box: (cx, cy, width, height, angle_deg).
OBB: TypeAlias = tuple[float, float, float, float, float]

# Model prediction: (frame_id, score, cx, cy, width, height, angle_deg).
Prediction: TypeAlias = tuple[str, float, float, float, float, float, float]

PredictionsByClass: TypeAlias = dict[int, list[Prediction]]
GroundTruthsByClassFrame: TypeAlias = dict[int, dict[str, list[OBB]]]
Polygon: TypeAlias = NDArray[np.float32]
DetailedResults: TypeAlias = dict[str, object]


@dataclass(frozen=True, slots=True)
class _PreparedBox:
    """Validated polygon and area used during repeated matching."""

    polygon: Polygon
    area: float


@dataclass(frozen=True, slots=True)
class _PreparedPrediction:
    """Validated prediction sorted by model confidence."""

    frame_id: str
    score: float
    box: _PreparedBox


def normalize_angle(angle_deg: float) -> float:
    """Normalize a finite angle to the half-open range [0, 360).

    Args:
        angle_deg: Angle expressed in degrees.

    Returns:
        The equivalent angle in the range [0, 360).

    Raises:
        ValueError: If the angle is not finite.
    """
    normalized_input = float(angle_deg)
    if not math.isfinite(normalized_input):
        raise ValueError("angle_deg must be finite")
    return normalized_input % 360.0


def validate_obb(obb: OBB) -> OBB:
    """Validate and normalize an oriented bounding box.

    Args:
        obb: Box encoded as ``(cx, cy, width, height, angle_deg)``.

    Returns:
        A float-valued OBB whose angle is normalized to [0, 360).

    Raises:
        ValueError: If a value is non-finite or a dimension is not positive.
    """
    cx, cy, width, height, angle_deg = (float(value) for value in obb)
    values = (cx, cy, width, height, angle_deg)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all OBB values must be finite")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("OBB width and height must be greater than zero")
    return (cx, cy, width, height, normalize_angle(angle_deg))


def obb_to_polygon(obb: OBB) -> Polygon:
    """Convert a parametric OBB into four counter-clockwise vertices.

    Args:
        obb: Box encoded as ``(cx, cy, width, height, angle_deg)``.

    Returns:
        A ``(4, 2)`` float32 array of polygon vertices in pixel coordinates.
    """
    return _validated_obb_to_polygon(validate_obb(obb))


def _validated_obb_to_polygon(obb: OBB) -> Polygon:
    """Convert an already validated OBB without repeating validation."""
    cx, cy, width, height, angle_deg = obb
    theta = math.radians(angle_deg)
    rotation = np.asarray(
        (
            (math.cos(theta), -math.sin(theta)),
            (math.sin(theta), math.cos(theta)),
        ),
        dtype=np.float64,
    )
    local_vertices = np.asarray(
        (
            (-width / 2.0, -height / 2.0),
            (width / 2.0, -height / 2.0),
            (width / 2.0, height / 2.0),
            (-width / 2.0, height / 2.0),
        ),
        dtype=np.float64,
    )
    vertices = local_vertices @ rotation.T + np.asarray((cx, cy))
    return vertices.astype(np.float32)


def _prepare_box(obb: OBB) -> _PreparedBox:
    """Validate an OBB and cache its polygon and exact parametric area."""
    _, _, width, height, _ = validated = validate_obb(obb)
    return _PreparedBox(_validated_obb_to_polygon(validated), width * height)


def _polygon_iou(box_a: _PreparedBox, box_b: _PreparedBox) -> float:
    """Compute rIoU between two validated convex polygons."""
    intersection_area, _ = cv2.intersectConvexConvex(box_a.polygon, box_b.polygon)
    intersection = min(max(float(intersection_area), 0.0), min(box_a.area, box_b.area))
    union = box_a.area + box_b.area - intersection
    if union <= 0.0:
        return 0.0
    return min(max(intersection / union, 0.0), 1.0)


def rotated_iou(box_a: OBB, box_b: OBB) -> float:
    """Compute rotated intersection over union for two valid OBBs.

    Args:
        box_a: First oriented bounding box.
        box_b: Second oriented bounding box.

    Returns:
        Rotated IoU in the closed range [0, 1].
    """
    return _polygon_iou(_prepare_box(box_a), _prepare_box(box_b))


def precision_recall(
    tp: Sequence[int], fp: Sequence[int], total_gt: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute cumulative Precision and Recall from ranked TP/FP flags.

    Args:
        tp: Binary true-positive flags ordered by descending score.
        fp: Binary false-positive flags ordered by descending score.
        total_gt: Number of ground-truth objects for the evaluated class.

    Returns:
        Arrays containing cumulative Precision and Recall.

    Raises:
        ValueError: If flags are inconsistent or ``total_gt`` is negative.
    """
    if len(tp) != len(fp):
        raise ValueError("tp and fp must have equal lengths")
    if total_gt < 0:
        raise ValueError("total_gt cannot be negative")

    tp_array = np.asarray(tp, dtype=np.float64)
    fp_array = np.asarray(fp, dtype=np.float64)
    if not np.all(np.isin(tp_array, (0.0, 1.0))):
        raise ValueError("tp flags must be binary")
    if not np.all(np.isin(fp_array, (0.0, 1.0))):
        raise ValueError("fp flags must be binary")
    if not np.all(tp_array + fp_array == 1.0):
        raise ValueError("each prediction must be exactly one of TP or FP")

    cumulative_tp = np.cumsum(tp_array)
    cumulative_fp = np.cumsum(fp_array)
    denominators = cumulative_tp + cumulative_fp
    precision = np.divide(
        cumulative_tp,
        denominators,
        out=np.zeros_like(cumulative_tp),
        where=denominators > 0.0,
    )
    recall = (
        cumulative_tp / float(total_gt)
        if total_gt > 0
        else np.zeros_like(cumulative_tp)
    )
    return precision, recall


def average_precision_101(tp: Sequence[int], fp: Sequence[int], total_gt: int) -> float:
    """Compute COCO-style AP using 101 interpolated recall levels.

    Args:
        tp: Binary true-positive flags ordered by descending score.
        fp: Binary false-positive flags ordered by descending score.
        total_gt: Number of ground-truth objects for the evaluated class.

    Returns:
        Interpolated AP in the closed range [0, 1]. A class without GT has
        AP equal to zero.
    """
    precision, recall = precision_recall(tp, fp, total_gt)
    if total_gt == 0:
        return 0.0
    recall_levels = np.linspace(0.0, 1.0, 101)
    interpolated = np.zeros_like(recall_levels)
    for index, recall_level in enumerate(recall_levels):
        eligible = precision[recall >= recall_level]
        if eligible.size > 0:
            interpolated[index] = np.max(eligible)
    return min(max(float(np.mean(interpolated)), 0.0), 1.0)


def _prepare_predictions(
    predictions: Sequence[Prediction],
) -> list[_PreparedPrediction]:
    """Validate predictions and sort them stably by descending score."""
    prepared: list[_PreparedPrediction] = []
    for prediction in predictions:
        frame_id, score, cx, cy, width, height, angle_deg = prediction
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("prediction frame_id must be a non-empty string")
        normalized_score = float(score)
        if not math.isfinite(normalized_score) or not 0.0 <= normalized_score <= 1.0:
            raise ValueError("prediction scores must be finite and within [0, 1]")
        box = _prepare_box((cx, cy, width, height, angle_deg))
        prepared.append(_PreparedPrediction(frame_id, normalized_score, box))
    return sorted(prepared, key=lambda item: item.score, reverse=True)


def _prepare_ground_truths(
    ground_truths: dict[str, list[OBB]],
) -> dict[str, list[_PreparedBox]]:
    """Validate and group ground-truth boxes by frame."""
    prepared: dict[str, list[_PreparedBox]] = {}
    for frame_id, boxes in ground_truths.items():
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("ground-truth frame_id must be a non-empty string")
        prepared[frame_id] = [_prepare_box(box) for box in boxes]
    return prepared


def _precompute_ious(
    predictions: Sequence[_PreparedPrediction],
    ground_truths: dict[str, list[_PreparedBox]],
) -> list[list[float]]:
    """Compute candidate rIoUs once so all seven thresholds can reuse them."""
    return [
        [
            _polygon_iou(prediction.box, ground_truth)
            for ground_truth in ground_truths.get(prediction.frame_id, [])
        ]
        for prediction in predictions
    ]


def _match_at_threshold(
    predictions: Sequence[_PreparedPrediction],
    ground_truths: dict[str, list[_PreparedBox]],
    ious: Sequence[Sequence[float]],
    threshold: float,
) -> tuple[list[int], list[int], dict[str, int]]:
    """Greedily match one class at one rIoU threshold."""
    used_by_frame: dict[str, set[int]] = {}
    tp: list[int] = []
    fp: list[int] = []

    for prediction, candidate_ious in zip(predictions, ious, strict=True):
        used_indices = used_by_frame.setdefault(prediction.frame_id, set())
        best_index: int | None = None
        best_iou = -1.0
        for gt_index, candidate_iou in enumerate(candidate_ious):
            if gt_index not in used_indices and candidate_iou > best_iou:
                best_index = gt_index
                best_iou = candidate_iou

        if best_index is not None and best_iou >= threshold:
            used_indices.add(best_index)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    true_positives = sum(tp)
    false_positives = sum(fp)
    total_gt = sum(len(boxes) for boxes in ground_truths.values())
    counts = {
        "tp": true_positives,
        "fp": false_positives,
        "fn": total_gt - true_positives,
    }
    return tp, fp, counts


def _validate_class_ids(
    predictions: PredictionsByClass, ground_truths: GroundTruthsByClassFrame
) -> None:
    """Reject class IDs outside the nine official categories."""
    unknown = (set(predictions) | set(ground_truths)).difference(CLASS_IDS)
    if unknown:
        raise ValueError(f"unsupported class IDs: {sorted(unknown)}")


def compute_macro_ap_riou(
    predictions: PredictionsByClass,
    ground_truths: GroundTruthsByClassFrame,
) -> tuple[float, DetailedResults]:
    """Compute exact Macro AP-rIoU over nine classes and seven thresholds.

    Predictions are matched greedily by descending confidence against unused
    ground truths from the same class and frame. AP uses COCO interpolation at
    101 recall levels. Classes without ground truth receive AP equal to zero.

    Args:
        predictions: Predictions grouped by official class ID (1 through 9).
        ground_truths: Ground-truth OBBs grouped by class ID and frame ID.

    Returns:
        The Macro AP-rIoU score and detailed AP/count diagnostics.

    Raises:
        ValueError: If a class ID, score, frame ID, or OBB is invalid.
    """
    _validate_class_ids(predictions, ground_truths)
    ap_by_class: dict[int, float] = {}
    ap_by_class_threshold: dict[int, dict[float, float]] = {}
    counts: dict[int, dict[float, dict[str, int]]] = {}

    for class_id in CLASS_IDS:
        class_predictions = _prepare_predictions(predictions.get(class_id, []))
        class_ground_truths = _prepare_ground_truths(ground_truths.get(class_id, {}))
        ious = _precompute_ious(class_predictions, class_ground_truths)
        total_gt = sum(len(boxes) for boxes in class_ground_truths.values())
        threshold_aps: list[float] = []
        threshold_counts: dict[float, dict[str, int]] = {}
        threshold_scores: dict[float, float] = {}

        for threshold in RIOU_THRESHOLDS:
            tp, fp, diagnostic_counts = _match_at_threshold(
                class_predictions,
                class_ground_truths,
                ious,
                threshold,
            )
            ap = average_precision_101(tp, fp, total_gt)
            threshold_aps.append(ap)
            threshold_scores[threshold] = ap
            threshold_counts[threshold] = diagnostic_counts

        ap_by_class[class_id] = float(np.mean(threshold_aps))
        ap_by_class_threshold[class_id] = threshold_scores
        counts[class_id] = threshold_counts

    macro_score = float(np.mean([ap_by_class[class_id] for class_id in CLASS_IDS]))
    details: DetailedResults = {
        "ap_by_class": ap_by_class,
        "ap_by_class_threshold": ap_by_class_threshold,
        "counts": counts,
    }
    return min(max(macro_score, 0.0), 1.0), details
