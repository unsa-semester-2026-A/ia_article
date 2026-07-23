"""Adapters from saved inference JSON files into evaluation data structures."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from src.evaluation.motion_filter import Detection
from src.evaluation.pipeline import PredictionsByClip, PredictionsByFrame


def obb_corners_to_xywhr_deg(corners: list[float]) -> tuple[float, float, float, float, float]:
    """Convert flat four-corner OBB coordinates into center, size, and angle."""
    if len(corners) != 8:
        raise ValueError("obb_corners must contain exactly 8 numbers")
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    if not np.all(np.isfinite(points)):
        raise ValueError("obb_corners must contain finite values")

    center = points.mean(axis=0)
    edge_01 = points[1] - points[0]
    edge_12 = points[2] - points[1]
    width = float(np.linalg.norm(edge_01))
    height = float(np.linalg.norm(edge_12))
    if width <= 0.0 or height <= 0.0:
        raise ValueError("OBB width and height must be greater than zero")

    angle_deg = math.degrees(math.atan2(float(edge_01[1]), float(edge_01[0]))) % 360.0
    return (float(center[0]), float(center[1]), width, height, angle_deg)


def load_inference_clip_json(
    json_path: str | Path,
    *,
    class_id_offset: int = 1,
    class_id_map: dict[int, int] | None = None,
    frame_idx_width: int = 4,
) -> tuple[str, PredictionsByFrame]:
    """Load one ``*_predictions.json`` file produced by an inference runner.

    Args:
        json_path: Path to a clip prediction JSON.
        class_id_offset: Value added to saved zero-based model class IDs to match
            the official metric contract. Use ``1`` for MTC 0..8 -> 1..9.
            Ignored when ``class_id_map`` is provided.
        class_id_map: Optional explicit saved-to-official class ID mapping.
        frame_idx_width: Zero-padding width used by dataset frame IDs.

    Returns:
        Tuple with clip ID and predictions keyed by reconstructed frame ID.
    """
    if frame_idx_width <= 0:
        raise ValueError("frame_idx_width must be positive")
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    clip_id = _require_str(data, "clip_id")
    frames = _require_list(data, "frames")

    predictions: PredictionsByFrame = {}
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("each frame entry must be an object")
        frame_idx = _require_int(frame, "frame_idx")
        frame_id = f"{clip_id}_{frame_idx:0{frame_idx_width}d}"
        detections = _require_list(frame, "detections")
        predictions[frame_id] = [
            _json_detection_to_detection(
                det,
                class_id_offset=class_id_offset,
                class_id_map=class_id_map,
            )
            for det in detections
        ]

    return clip_id, predictions


def load_inference_predictions_dir(
    predictions_dir: str | Path,
    *,
    max_clips: int = 0,
    class_id_offset: int = 1,
    class_id_map: dict[int, int] | None = None,
    frame_idx_width: int = 4,
) -> PredictionsByClip:
    """Load all clip prediction JSONs in a directory."""
    directory = Path(predictions_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"predictions directory not found: {directory}")
    if max_clips < 0:
        raise ValueError("max_clips must be non-negative")

    json_paths = sorted(directory.glob("*_predictions.json"))
    if max_clips > 0:
        json_paths = json_paths[:max_clips]
    if not json_paths:
        raise FileNotFoundError(f"no *_predictions.json files found in {directory}")

    predictions_by_clip: PredictionsByClip = {}
    for json_path in json_paths:
        clip_id, predictions = load_inference_clip_json(
            json_path,
            class_id_offset=class_id_offset,
            class_id_map=class_id_map,
            frame_idx_width=frame_idx_width,
        )
        if clip_id in predictions_by_clip:
            raise ValueError(f"duplicate clip_id found: {clip_id}")
        predictions_by_clip[clip_id] = predictions
    return predictions_by_clip


def _json_detection_to_detection(
    detection: Any,
    *,
    class_id_offset: int,
    class_id_map: dict[int, int] | None,
) -> Detection:
    if not isinstance(detection, dict):
        raise ValueError("each detection entry must be an object")
    saved_class_id = _require_int(detection, "class_id")
    class_id = (
        class_id_map[saved_class_id]
        if class_id_map is not None
        else saved_class_id + class_id_offset
    )
    score = _require_float(detection, "score")
    cx, cy, width, height, angle_deg = obb_corners_to_xywhr_deg(
        _require_list(detection, "obb_corners")
    )
    return Detection(class_id, score, cx, cy, width, height, angle_deg)


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _require_float(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{key} must be finite")
    return normalized


def _require_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value
