"""Tests for the exact Macro AP-rIoU evaluation module."""

import math
import time
from typing import cast

import cv2
import numpy as np
import pytest
from src.evaluation.metric import (
    CLASS_IDS,
    OBB,
    RIOU_THRESHOLDS,
    GroundTruthsByClassFrame,
    Prediction,
    PredictionsByClass,
    average_precision_101,
    compute_macro_ap_riou,
    normalize_angle,
    obb_to_polygon,
    precision_recall,
    rotated_iou,
    validate_obb,
)


def _prediction(frame_id: str, score: float, obb: OBB) -> Prediction:
    """Build a prediction tuple from its frame, score, and OBB."""
    cx, cy, width, height, angle_deg = obb
    return (frame_id, score, cx, cy, width, height, angle_deg)


def _perfect_nine_class_data() -> tuple[PredictionsByClass, GroundTruthsByClassFrame]:
    """Build one perfect prediction and GT for every official class."""
    predictions: PredictionsByClass = {}
    ground_truths: GroundTruthsByClassFrame = {}
    for class_id in CLASS_IDS:
        box = (100.0 + class_id, 120.0, 80.0, 40.0, 10.0)
        predictions[class_id] = [_prediction("frame_perfect", 1.0, box)]
        ground_truths[class_id] = {"frame_perfect": [box]}
    return predictions, ground_truths


def test_official_evaluation_contract() -> None:
    """Expose nine classes, seven thresholds, and a callable metric."""
    assert CLASS_IDS == tuple(range(1, 10))
    assert RIOU_THRESHOLDS == (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    assert callable(compute_macro_ap_riou)


@pytest.mark.parametrize(
    "invalid_box",
    [
        (0.0, 0.0, 0.0, 2.0, 0.0),
        (0.0, 0.0, 2.0, -1.0, 0.0),
        (math.nan, 0.0, 2.0, 1.0, 0.0),
        (0.0, 0.0, 2.0, 1.0, math.inf),
    ],
)
def test_validate_obb_rejects_invalid_geometry(invalid_box: OBB) -> None:
    """Reject non-positive dimensions and non-finite OBB values."""
    with pytest.raises(ValueError):
        validate_obb(invalid_box)


def test_angle_normalization_and_polygon_area() -> None:
    """Normalize equivalent angles and preserve center and polygon area."""
    negative_box: OBB = (200.0, 150.0, 120.0, 60.0, -15.0)
    equivalent_box: OBB = (200.0, 150.0, 120.0, 60.0, 345.0)
    polygon = obb_to_polygon(negative_box)

    assert normalize_angle(-15.0) == 345.0
    assert np.allclose(polygon, obb_to_polygon(equivalent_box))
    assert np.allclose(np.mean(polygon, axis=0), (200.0, 150.0))
    assert cv2.contourArea(polygon) == pytest.approx(120.0 * 60.0)


def test_rotated_iou_extreme_cases_and_symmetry() -> None:
    """Handle identical, separate, symmetric, and equivalent-angle boxes."""
    box_a: OBB = (160.0, 150.0, 120.0, 80.0, 0.0)
    box_b: OBB = (200.0, 150.0, 120.0, 80.0, 0.0)
    separated: OBB = (340.0, 150.0, 120.0, 80.0, 0.0)

    assert rotated_iou(box_a, box_a) == pytest.approx(1.0)
    assert rotated_iou(box_a, separated) == pytest.approx(0.0)
    assert rotated_iou(box_a, box_b) == pytest.approx(0.5)
    assert rotated_iou(box_a, box_b) == pytest.approx(rotated_iou(box_b, box_a))
    assert rotated_iou(
        (200.0, 150.0, 120.0, 60.0, -15.0),
        (200.0, 150.0, 120.0, 60.0, 345.0),
    ) == pytest.approx(1.0)


def test_angular_deviation_reduces_riou_progressively() -> None:
    """Reduce overlap monotonically for an elongated box rotated to 45 degrees."""
    reference: OBB = (200.0, 150.0, 160.0, 40.0, 0.0)
    scores = [
        rotated_iou(reference, (200.0, 150.0, 160.0, 40.0, angle))
        for angle in (10.0, 20.0, 30.0, 45.0)
    ]

    assert all(left > right for left, right in zip(scores, scores[1:]))
    assert scores[-1] < 0.50


def test_precision_recall_and_ap_101() -> None:
    """Compute cumulative curves and exact 101-point interpolation cases."""
    precision, recall = precision_recall([1, 0, 1, 0], [0, 1, 0, 1], total_gt=2)

    assert np.allclose(precision, (1.0, 0.5, 2.0 / 3.0, 0.5))
    assert np.allclose(recall, (0.5, 0.5, 1.0, 1.0))
    assert average_precision_101([1], [0], total_gt=1) == pytest.approx(1.0)
    assert average_precision_101([], [], total_gt=1) == pytest.approx(0.0)
    assert average_precision_101([], [], total_gt=0) == pytest.approx(0.0)
    assert average_precision_101([1, 0], [0, 1], total_gt=1) == pytest.approx(1.0)
    assert average_precision_101([0, 1], [1, 0], total_gt=1) == pytest.approx(0.5)


def test_perfect_predictions_for_all_classes_score_one() -> None:
    """Return Macro AP one when all nine official classes are perfect."""
    predictions, ground_truths = _perfect_nine_class_data()
    score, details = compute_macro_ap_riou(predictions, ground_truths)
    ap_by_class = cast(dict[int, float], details["ap_by_class"])

    assert score == pytest.approx(1.0)
    assert all(ap_by_class[class_id] == pytest.approx(1.0) for class_id in CLASS_IDS)


def test_two_perfect_classes_and_empty_predictions() -> None:
    """Average all nine classes and return zero for empty predictions."""
    predictions, ground_truths = _perfect_nine_class_data()
    two_class_predictions = {class_id: predictions[class_id] for class_id in (1, 2)}
    two_class_ground_truths = {class_id: ground_truths[class_id] for class_id in (1, 2)}

    two_class_score, _ = compute_macro_ap_riou(
        two_class_predictions, two_class_ground_truths
    )
    empty_score, _ = compute_macro_ap_riou({}, ground_truths)

    assert two_class_score == pytest.approx(2.0 / 9.0)
    assert empty_score == pytest.approx(0.0)


def test_duplicate_counts_and_ap_ordering() -> None:
    """Count a duplicate FP and distinguish late from early false positives."""
    box: OBB = (160.0, 150.0, 120.0, 80.0, 0.0)
    ground_truths: GroundTruthsByClassFrame = {1: {"frame_001": [box]}}
    late_duplicate: PredictionsByClass = {
        1: [
            _prediction("frame_001", 0.95, box),
            _prediction("frame_001", 0.80, box),
        ]
    }
    early_fp: PredictionsByClass = {
        1: [
            _prediction("frame_001", 0.99, (340.0, 150.0, 120.0, 80.0, 0.0)),
            _prediction("frame_001", 0.90, box),
        ]
    }

    duplicate_score, duplicate_details = compute_macro_ap_riou(
        late_duplicate, ground_truths
    )
    early_fp_score, early_fp_details = compute_macro_ap_riou(early_fp, ground_truths)
    duplicate_ap = cast(dict[int, float], duplicate_details["ap_by_class"])
    early_ap = cast(dict[int, float], early_fp_details["ap_by_class"])
    counts = cast(dict[int, dict[float, dict[str, int]]], duplicate_details["counts"])

    assert duplicate_score == pytest.approx(1.0 / 9.0)
    assert duplicate_ap[1] == pytest.approx(1.0)
    assert early_fp_score == pytest.approx(0.5 / 9.0)
    assert early_ap[1] == pytest.approx(0.5)
    assert all(
        counts[1][threshold] == {"tp": 1, "fp": 1, "fn": 0}
        for threshold in RIOU_THRESHOLDS
    )


def test_matching_is_restricted_to_the_same_frame() -> None:
    """Count a geometrically perfect prediction in another frame as FP."""
    box: OBB = (160.0, 150.0, 120.0, 80.0, 0.0)
    predictions: PredictionsByClass = {1: [_prediction("frame_002", 0.9, box)]}
    ground_truths: GroundTruthsByClassFrame = {1: {"frame_001": [box]}}
    _, details = compute_macro_ap_riou(predictions, ground_truths)
    counts = cast(dict[int, dict[float, dict[str, int]]], details["counts"])

    assert counts[1][0.50] == {"tp": 0, "fp": 1, "fn": 1}


def test_equal_scores_keep_input_order() -> None:
    """Use Python's stable score ordering for deterministic matching."""
    box: OBB = (160.0, 150.0, 120.0, 80.0, 0.0)
    predictions: PredictionsByClass = {
        1: [
            _prediction("frame_001", 0.9, (340.0, 150.0, 120.0, 80.0, 0.0)),
            _prediction("frame_001", 0.9, box),
        ]
    }
    ground_truths: GroundTruthsByClassFrame = {1: {"frame_001": [box]}}
    score, _ = compute_macro_ap_riou(predictions, ground_truths)

    assert score == pytest.approx(0.5 / 9.0)


@pytest.mark.parametrize("invalid_score", [-0.1, 1.1, math.nan, math.inf])
def test_invalid_prediction_scores_are_rejected(invalid_score: float) -> None:
    """Reject scores outside the finite closed interval [0, 1]."""
    box: OBB = (160.0, 150.0, 120.0, 80.0, 0.0)
    predictions: PredictionsByClass = {
        1: [_prediction("frame_001", invalid_score, box)]
    }

    with pytest.raises(ValueError):
        compute_macro_ap_riou(predictions, {})


def test_unknown_class_is_rejected() -> None:
    """Reject predictions outside the nine official class IDs."""
    box: OBB = (160.0, 150.0, 120.0, 80.0, 0.0)
    predictions: PredictionsByClass = {10: [_prediction("frame_001", 0.9, box)]}

    with pytest.raises(ValueError, match="unsupported class IDs"):
        compute_macro_ap_riou(predictions, {})


def test_metric_benchmark_50k_predictions_10k_gt() -> None:
    """Evaluate 50k predictions and 10k GT in under 30 seconds on one CPU thread."""
    rng = np.random.default_rng(2026)
    predictions: PredictionsByClass = {class_id: [] for class_id in CLASS_IDS}
    ground_truths: GroundTruthsByClassFrame = {class_id: {} for class_id in CLASS_IDS}

    for frame_index in range(1_000):
        frame_id = f"benchmark_{frame_index:04d}"
        for box_index in range(10):
            class_id = CLASS_IDS[box_index % len(CLASS_IDS)]
            box: OBB = (
                float(rng.uniform(100.0, 1_900.0)),
                float(rng.uniform(100.0, 900.0)),
                float(rng.uniform(30.0, 150.0)),
                float(rng.uniform(20.0, 80.0)),
                float(rng.uniform(0.0, 360.0)),
            )
            ground_truths[class_id].setdefault(frame_id, []).append(box)

        for prediction_index in range(50):
            class_id = CLASS_IDS[prediction_index % len(CLASS_IDS)]
            box = (
                float(rng.uniform(100.0, 1_900.0)),
                float(rng.uniform(100.0, 900.0)),
                float(rng.uniform(30.0, 150.0)),
                float(rng.uniform(20.0, 80.0)),
                float(rng.uniform(0.0, 360.0)),
            )
            predictions[class_id].append(
                _prediction(frame_id, float(rng.uniform(0.0, 1.0)), box)
            )

    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    started_at = time.perf_counter()
    try:
        score, _ = compute_macro_ap_riou(predictions, ground_truths)
    finally:
        elapsed_seconds = time.perf_counter() - started_at
        cv2.setNumThreads(previous_threads)

    print(f"Macro AP-rIoU benchmark: {elapsed_seconds:.3f} seconds")
    assert sum(map(len, predictions.values())) == 50_000
    assert (
        sum(
            len(boxes) for frames in ground_truths.values() for boxes in frames.values()
        )
        == 10_000
    )
    assert 0.0 <= score <= 1.0
    assert elapsed_seconds < 30.0, f"benchmark took {elapsed_seconds:.3f} seconds"
