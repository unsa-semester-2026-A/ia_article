"""Macro AP-rIoU metric for oriented vehicle detection.

This module follows the public interface defined in
``.alvaro/01_procedimiento/plan/02_metric.md``.  The geometric and metric
calculations are introduced incrementally in the subsequent implementation
stages of issue #9.
"""

from typing import TypeAlias

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
DetailedResults: TypeAlias = dict[str, object]


def compute_macro_ap_riou(
    predictions: PredictionsByClass,
    ground_truths: GroundTruthsByClassFrame,
) -> tuple[float, DetailedResults]:
    """Compute Macro AP-rIoU over the official classes and thresholds.

    Args:
        predictions: Predictions grouped by official class ID (1 through 9).
        ground_truths: Ground-truth OBBs grouped by class ID and frame ID.

    Returns:
        The macro score and a dictionary with detailed diagnostic results.

    Raises:
        NotImplementedError: Until the remaining metric stages are completed.
    """
    del predictions, ground_truths
    raise NotImplementedError("Metric calculations begin in stage A1")
