"""Tests for inference adaptation and filter-to-metric integration."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from src.evaluation.motion_filter import Detection, MotionFilterDiagnostics
from src.evaluation.pipeline import (
    INFERENCE_CONFIDENCE,
    INTERNAL_TO_OFFICIAL_CLASS_IDS,
    ConditionsPredictions,
    OBBResultLike,
    PipelineEvaluation,
    TensorLike,
    YoloResultLike,
    adapt_ultralytics_result,
    adapt_yolo_obb_arrays,
    build_synthetic_pipeline_case,
    evaluate_conditions,
    evaluate_dataset,
    group_predictions_by_clip,
    infer_clip,
    load_ground_truth_csv,
    parse_ground_truth_target,
    predictions_for_metric,
    run_synthetic_pipeline,
    split_frame_id,
    write_evaluation_report,
)


class _FakeTensor(TensorLike):
    """CPU NumPy wrapper emulating the tensor operations used by the adapter."""

    def __init__(self, values: NDArray[np.generic]) -> None:
        self._values = values

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> NDArray[np.generic]:
        return self._values


class _FakeOBB(OBBResultLike):
    """Minimal fake OBB collection."""

    def __init__(
        self,
        xywhr: NDArray[np.float64],
        scores: NDArray[np.float64],
        classes: NDArray[np.int64],
    ) -> None:
        self.xywhr = _FakeTensor(xywhr)
        self.conf = _FakeTensor(scores)
        self.cls = _FakeTensor(classes)


class _FakeResult(YoloResultLike):
    """Minimal fake Ultralytics result."""

    def __init__(self, obb: OBBResultLike | None) -> None:
        self.obb = obb


class _FakeModel:
    """Deterministic model returning one OBB result per frame."""

    def predict(
        self,
        source: list[NDArray[np.uint8]],
        *,
        conf: float,
        batch: int,
        verbose: bool,
    ) -> list[_FakeResult]:
        assert conf == INFERENCE_CONFIDENCE
        assert batch == 2
        assert verbose is False
        return [
            _FakeResult(
                _FakeOBB(
                    np.asarray(((10.0 + index, 20.0, 8.0, 4.0, math.pi / 2),)),
                    np.asarray((0.75,)),
                    np.asarray((0,), dtype=np.int64),
                )
            )
            for index, _ in enumerate(source)
        ]


def test_split_and_group_frame_predictions() -> None:
    """Recover clip IDs and preserve every source detection."""
    assert split_frame_id("v_demo_0042") == ("v_demo", 42)
    detection = Detection(1, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0)
    grouped = group_predictions_by_clip({"v_a_0000": [detection], "v_b_0000": []})
    assert set(grouped) == {"v_a", "v_b"}
    assert grouped["v_a"]["v_a_0000"] == [detection]


def test_yolo_adapter_maps_classes_pixels_scores_and_degrees() -> None:
    """Convert internal 0..8 IDs and radian angles into the public contract."""
    detections = adapt_yolo_obb_arrays(
        "clip_0000",
        np.asarray(((100.0, 50.0, 20.0, 10.0, math.pi / 2),)),
        np.asarray((0.8,)),
        np.asarray((0,), dtype=np.int64),
    )
    assert INTERNAL_TO_OFFICIAL_CLASS_IDS == {index: index + 1 for index in range(9)}
    assert detections == [Detection(1, 0.8, 100.0, 50.0, 20.0, 10.0, 90.0)]


def test_ultralytics_result_adapter_handles_obb_and_empty_results() -> None:
    """Read the same xywhr, conf, and cls attributes exposed by Results.obb."""
    result = _FakeResult(
        _FakeOBB(
            np.asarray(((1.0, 2.0, 3.0, 4.0, math.pi),)),
            np.asarray((0.5,)),
            np.asarray((8,), dtype=np.int64),
        )
    )
    assert adapt_ultralytics_result("clip_0000", result) == [
        Detection(9, 0.5, 1.0, 2.0, 3.0, 4.0, 180.0)
    ]
    assert adapt_ultralytics_result("clip_0000", _FakeResult(None)) == []


def test_infer_clip_batches_results_and_builds_homographies() -> None:
    """Run batch inference but perform CPU temporal work in frame order."""
    frames = {
        "clip_0001": np.zeros((64, 64, 3), dtype=np.uint8),
        "clip_0000": np.zeros((64, 64, 3), dtype=np.uint8),
    }
    predictions, homographies = infer_clip(_FakeModel(), frames, batch_size=2)

    assert list(predictions) == ["clip_0000", "clip_0001"]
    assert predictions["clip_0000"][0].angle_deg == pytest.approx(90.0)
    assert set(homographies) == {"clip_0001"}
    assert np.allclose(homographies["clip_0001"], np.eye(3))


def test_ground_truth_parser_and_csv_loader(tmp_path: Path) -> None:
    """Parse official Id/Target rows, including frames marked none."""
    assert parse_ground_truth_target("none") == []
    parsed = parse_ground_truth_target("1 10 20 30 40 -15;9 1 2 3 4 5")
    assert parsed[0] == (1, (10.0, 20.0, 30.0, 40.0, -15.0))
    csv_path = tmp_path / "ground_truth.csv"
    csv_path.write_text(
        "Id,Target\nclip_0000,1 10 20 30 40 -15\nclip_0001,none\n",
        encoding="utf-8",
    )
    assert load_ground_truth_csv(csv_path, {"clip_0000"}) == {
        1: {"clip_0000": [(10.0, 20.0, 30.0, 40.0, -15.0)]}
    }


def test_filtered_predictions_convert_to_metric_contract() -> None:
    """Keep frame, score, class, and pixel-space OBB fields unchanged."""
    detection = Detection(4, 0.7, 10.0, 20.0, 30.0, 40.0, 50.0)
    assert predictions_for_metric({"clip_0000": [detection]}) == {
        4: [("clip_0000", 0.7, 10.0, 20.0, 30.0, 40.0, 50.0)]
    }


def test_end_to_end_filter_then_metric_is_deterministic_and_preserves_gt() -> None:
    """Remove static predictions, preserve GT, and reproduce the global score."""
    predictions, homographies, ground_truths = build_synthetic_pipeline_case()
    original_ground_truths = copy.deepcopy(ground_truths)

    first = evaluate_dataset(predictions, homographies, ground_truths)
    second = evaluate_dataset(predictions, homographies, ground_truths)

    assert ground_truths == original_ground_truths
    assert first.macro_score == pytest.approx(1.0 / 9.0)
    assert second.macro_score == first.macro_score
    assert first.filtered_predictions == second.filtered_predictions
    assert first.motion_by_clip == {
        "clip_demo": MotionFilterDiagnostics(3, (0,), (1, 2), 12)
    }


def test_six_conditions_use_the_same_filter_and_report_writer(tmp_path: Path) -> None:
    """Evaluate six conditions through one function and serialize small reports."""
    predictions, homographies, ground_truths = build_synthetic_pipeline_case()
    condition_names = ("Base 0", "Base 1", "Base 2", "Mejora A", "Mejora B", "Mejora C")
    conditions: ConditionsPredictions = {
        name: copy.deepcopy(predictions) for name in condition_names
    }
    results = evaluate_conditions(conditions, homographies, ground_truths)

    assert list(results) == sorted(condition_names)
    assert all(
        isinstance(result, PipelineEvaluation)
        and result.macro_score == pytest.approx(1.0 / 9.0)
        for result in results.values()
    )
    report_path = write_evaluation_report(tmp_path / "report.json", results)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) == set(condition_names)


def test_synthetic_pipeline_public_smoke_test() -> None:
    """Expose a no-credential end-to-end execution for every platform."""
    result = run_synthetic_pipeline()
    assert result.macro_score == pytest.approx(1.0 / 9.0)


@pytest.mark.parametrize(
    "invalid_frame_id",
    ["", "clip", "clip_frame"],
)
def test_invalid_frame_ids_are_rejected(invalid_frame_id: str) -> None:
    """Reject identifiers that cannot be grouped temporally by clip."""
    with pytest.raises(ValueError):
        split_frame_id(invalid_frame_id)


def test_invalid_yolo_shapes_and_classes_are_rejected() -> None:
    """Reject inconsistent result lengths and classes without explicit mapping."""
    with pytest.raises(ValueError):
        adapt_yolo_obb_arrays(
            "clip_0000",
            np.zeros((1, 5)),
            np.zeros(2),
            np.zeros(1, dtype=np.int64),
        )
    with pytest.raises(ValueError, match="unmapped"):
        adapt_yolo_obb_arrays(
            "clip_0000",
            np.asarray(((1.0, 2.0, 3.0, 4.0, 0.0),)),
            np.asarray((0.5,)),
            np.asarray((9,), dtype=np.int64),
        )
