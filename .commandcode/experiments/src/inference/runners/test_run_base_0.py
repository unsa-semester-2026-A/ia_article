"""Unit tests for Base0Runner module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.inference.runners.run_base_0 import Base0Runner


@pytest.fixture
@patch("src.inference.runners.run_base_0.YOLO")
@patch("src.inference.runners.run_base_0.IOManager")
def runner(mock_io, mock_yolo):
    """Instantiate Base0Runner with mocks for YOLO and IOManager."""
    mock_model = MagicMock()
    mock_model.names = {
        0: "plane",
        9: "small-vehicle",
        10: "large-vehicle",
        14: "swimming-pool",
    }
    mock_yolo.return_value = mock_model

    config = {
        "metadata_path": "dummy_meta.csv",
        "images_dir": "dummy_img",
        "output_dir": "dummy_out",
        "device": "cpu",
        "batch_size": 2,
        "max_clips": 5,
    }
    return Base0Runner(config=config, model_path="dummy.pt")


# ==========================================
# Direct Class Mapping Tests (Black Box)
# ==========================================
def test_dota_to_mtc_mapping(runner):
    """Black Box: Verify direct mapping dictionary maps vehicle classes correctly."""
    assert runner.dota_to_mtc == {9: 0, 10: 6}


def test_non_vehicle_classes_excluded(runner):
    """Black Box: Ensure non-vehicle classes (plane, swimming-pool) are excluded."""
    assert 0 not in runner.dota_to_mtc  # plane
    assert 14 not in runner.dota_to_mtc  # swimming-pool


# ==========================================
# Track Results Processing Tests (White Box)
# ==========================================
def test_process_track_accumulates_speed(runner):
    """White Box: Verify speed metrics accumulate correctly."""
    mock_r1 = MagicMock()
    mock_r1.speed = {"preprocess": 1.0, "inference": 10.0, "postprocess": 2.0}
    mock_r1.orig_shape = (1080, 1920)
    mock_r1.obb = None

    mock_r2 = MagicMock()
    mock_r2.speed = {"preprocess": 1.5, "inference": 12.0, "postprocess": 2.5}
    mock_r2.orig_shape = (1080, 1920)
    mock_r2.obb = None

    frames = runner._process_track_results([mock_r1, mock_r2], frame_offset=0)

    assert len(frames) == 2
    assert runner.total_frames_processed == 2
    assert runner._time_sums["inference"] == 22.0


def test_process_track_extracts_track_ids_and_corners(runner):
    """White Box: Verify track_id extraction from r.obb.id."""
    mock_r = MagicMock()
    mock_r.speed = {"preprocess": 1.0, "inference": 10.0, "postprocess": 2.0}
    mock_r.orig_shape = (1080, 1920)

    mock_obb = MagicMock()
    mock_obb.__len__ = lambda self: 2
    corners = np.array(
        [
            [[10, 20], [50, 20], [50, 80], [10, 80]],
            [[100, 200], [150, 200], [150, 280], [100, 280]],
        ],
        dtype=np.float32,
    )
    mock_obb.xyxyxyxy.cpu().numpy.return_value = corners
    mock_obb.cls.cpu().numpy.return_value = np.array([9, 0])
    mock_obb.conf.cpu().numpy.return_value = np.array([0.95, 0.80])
    mock_obb.id.int().cpu().numpy.return_value = np.array([42, 101])
    mock_r.obb = mock_obb

    frames = runner._process_track_results([mock_r], frame_offset=5)

    assert frames[0]["frame_idx"] == 5
    assert len(frames[0]["detections"]) == 1
    det = frames[0]["detections"][0]
    assert det["track_id"] == 42
    assert det["class_id"] == 0
    assert det["score"] == pytest.approx(0.95, abs=0.01)
    assert len(det["obb_corners"]) == 8


# ==========================================
# Pipeline Execution Tests (White Box)
# ==========================================
@patch("src.inference.runners.run_base_0.gc")
def test_execute_pipeline_tracking(mock_gc, runner):
    """White Box: Verify model.track is invoked per frame."""
    runner.io_manager.load_csv.return_value = [
        {"clip_id": "clip1", "split": "val"},
        {"clip_id": "clip2", "split": "val"},
        {"clip_id": "clip3", "split": "train"},
    ]

    runner.io_manager.list_files_in_dir.return_value = [
        Path("f0.jpg"),
        Path("f1.jpg"),
    ]

    mock_result = MagicMock()
    mock_result.speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}
    mock_result.orig_shape = (1080, 1920)
    mock_result.obb = None
    runner.model.track.return_value = [mock_result]

    runner.start_hardware_monitoring = MagicMock()
    runner.record_hardware_metrics = MagicMock(return_value={"mock": True})

    res = runner.execute()

    # 2 val clips × 2 frames each = 4 tracking calls
    assert runner.model.track.call_count == 4
    assert mock_gc.collect.call_count == 2
    assert res["status"] == "success"


@patch("src.inference.runners.run_base_0.gc")
def test_execute_skips_empty_clips(mock_gc, runner):
    """Black Box: Empty clips are gracefully skipped."""
    runner.io_manager.load_csv.return_value = [
        {"clip_id": "empty_clip", "split": "val"},
    ]
    runner.io_manager.list_files_in_dir.return_value = []

    runner.start_hardware_monitoring = MagicMock()
    runner.record_hardware_metrics = MagicMock(return_value={})

    res = runner.execute()

    runner.model.track.assert_not_called()
    assert res["status"] == "success"


@patch("src.inference.runners.run_base_0.gc")
def test_execute_resumes_existing_clips(mock_gc, runner, tmp_path):
    """White Box: Already processed JSON files are skipped for resumption."""
    runner.config["output_dir"] = str(tmp_path)

    runner.io_manager.load_csv.return_value = [
        {"clip_id": "done_clip", "split": "val"},
        {"clip_id": "new_clip", "split": "val"},
    ]

    (tmp_path / "done_clip_predictions.json").write_text("{}")
    runner.io_manager.list_files_in_dir.return_value = [Path("f0.jpg")]

    mock_result = MagicMock()
    mock_result.speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}
    mock_result.orig_shape = (1080, 1920)
    mock_result.obb = None
    runner.model.track.return_value = [mock_result]

    runner.start_hardware_monitoring = MagicMock()
    runner.record_hardware_metrics = MagicMock(return_value={})

    res = runner.execute()

    # Only new_clip processed (1 frame = 1 tracking call)
    assert runner.model.track.call_count == 1
    assert res["status"] == "success"


@patch("src.inference.runners.run_base_0.gc")
def test_execute_saves_metrics_on_error(mock_gc, runner):
    """White Box: Finally block saves metrics even on exception."""
    runner.io_manager.load_csv.return_value = [
        {"clip_id": "clip1", "split": "val"},
    ]
    runner.io_manager.list_files_in_dir.return_value = [Path("f0.jpg")]

    runner.model.track.side_effect = RuntimeError("Simulated OOM")

    runner.start_hardware_monitoring = MagicMock()
    runner.record_hardware_metrics = MagicMock(return_value={"partial": True})

    res = runner.execute()

    runner.record_hardware_metrics.assert_called_once()
    assert runner.io_manager.save_json.call_count >= 1
    assert res["status"] == "success"
