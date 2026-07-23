"""Tests for saved inference JSON adapters."""

import json

import pytest
from src.evaluation.inference_json import (
    load_inference_clip_json,
    load_inference_predictions_dir,
    obb_corners_to_xywhr_deg,
)
from src.evaluation.motion_filter import Detection


def test_obb_corners_to_xywhr_deg_axis_aligned() -> None:
    """Convert flat OBB corners into the metric OBB representation."""
    obb = obb_corners_to_xywhr_deg([0.0, 0.0, 4.0, 0.0, 4.0, 2.0, 0.0, 2.0])
    assert obb == pytest.approx((2.0, 1.0, 4.0, 2.0, 0.0))


def test_load_inference_clip_json_builds_official_detections(tmp_path) -> None:
    """Load Base 0-style JSON and convert class IDs and corners."""
    json_path = tmp_path / "clip_a_predictions.json"
    json_path.write_text(
        json.dumps(
            {
                "clip_id": "clip_a",
                "inference_shape": [640, 640],
                "frames": [
                    {
                        "frame_idx": 3,
                        "original_shape": [1080, 1920],
                        "speed_ms": {},
                        "detections": [
                            {
                                "track_id": 7,
                                "class_id": 0,
                                "score": 0.75,
                                "obb_corners": [
                                    0.0,
                                    0.0,
                                    4.0,
                                    0.0,
                                    4.0,
                                    2.0,
                                    0.0,
                                    2.0,
                                ],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    clip_id, predictions = load_inference_clip_json(json_path)

    assert clip_id == "clip_a"
    assert predictions == {
        "clip_a_0003": [Detection(1, 0.75, 2.0, 1.0, 4.0, 2.0, 0.0)]
    }


def test_load_inference_predictions_dir_limits_clips(tmp_path) -> None:
    """Load a bounded subset of prediction JSONs for smoke tests."""
    for clip_id in ("clip_a", "clip_b"):
        (tmp_path / f"{clip_id}_predictions.json").write_text(
            json.dumps({"clip_id": clip_id, "inference_shape": [640, 640], "frames": []}),
            encoding="utf-8",
        )

    predictions = load_inference_predictions_dir(tmp_path, max_clips=1)

    assert list(predictions) == ["clip_a"]


def test_load_inference_clip_json_supports_frame_width(tmp_path) -> None:
    """Use the dataset frame zero-padding width configured by callers."""
    json_path = tmp_path / "clip_a_predictions.json"
    json_path.write_text(
        json.dumps(
            {
                "clip_id": "clip_a",
                "inference_shape": [640, 640],
                "frames": [{"frame_idx": 8, "detections": []}],
            }
        ),
        encoding="utf-8",
    )

    _, predictions = load_inference_clip_json(json_path, frame_idx_width=6)

    assert list(predictions) == ["clip_a_000008"]


def test_invalid_obb_corners_are_rejected() -> None:
    """Reject malformed raw prediction geometry early."""
    with pytest.raises(ValueError, match="exactly 8"):
        obb_corners_to_xywhr_deg([0.0, 1.0])
