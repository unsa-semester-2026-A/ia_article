"""Evaluation utilities for the SMART Challenge 2026."""

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

__all__ = [
    "CLASS_IDS",
    "RIOU_THRESHOLDS",
    "GroundTruthsByClassFrame",
    "OBB",
    "Prediction",
    "PredictionsByClass",
    "average_precision_101",
    "compute_macro_ap_riou",
    "normalize_angle",
    "obb_to_polygon",
    "precision_recall",
    "rotated_iou",
    "validate_obb",
]
