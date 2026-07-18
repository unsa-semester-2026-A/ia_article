"""Contract tests for the Macro AP-rIoU evaluation module."""

from src.evaluation.metric import (
    CLASS_IDS,
    RIOU_THRESHOLDS,
    GroundTruthsByClassFrame,
    PredictionsByClass,
    compute_macro_ap_riou,
)


def test_official_evaluation_contract() -> None:
    """Expose the nine classes, seven thresholds, and metric entry point."""
    predictions: PredictionsByClass = {
        1: [("v_example_0001", 0.95, 450.0, 300.0, 80.0, 40.0, 25.0)]
    }
    ground_truths: GroundTruthsByClassFrame = {
        1: {"v_example_0001": [(452.0, 301.0, 78.0, 42.0, 24.0)]}
    }

    assert CLASS_IDS == tuple(range(1, 10))
    assert RIOU_THRESHOLDS == (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    assert len(predictions[1][0]) == 7
    assert len(ground_truths[1]["v_example_0001"][0]) == 5
    assert callable(compute_macro_ap_riou)
