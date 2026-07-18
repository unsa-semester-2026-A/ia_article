"""Evaluation utilities for the SMART Challenge 2026."""

from src.evaluation.metric import (
    CLASS_IDS,
    OBB,
    RIOU_THRESHOLDS,
    GroundTruthsByClassFrame,
    Prediction,
    PredictionsByClass,
    compute_macro_ap_riou,
)

__all__ = [
    "CLASS_IDS",
    "RIOU_THRESHOLDS",
    "GroundTruthsByClassFrame",
    "OBB",
    "Prediction",
    "PredictionsByClass",
    "compute_macro_ap_riou",
]
