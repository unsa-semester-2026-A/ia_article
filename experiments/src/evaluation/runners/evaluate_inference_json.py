"""Evaluate saved inference JSON predictions with homography motion filtering."""

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from src.evaluation.inference_json import load_inference_predictions_dir
from src.evaluation.metric import obb_to_polygon
from src.evaluation.motion_filter import Detection, estimate_homography
from src.evaluation.pipeline import (
    HomographiesByClip,
    PipelineEvaluation,
    PredictionsByClip,
    evaluate_dataset,
    load_ground_truth_csv,
    predictions_for_metric,
    write_evaluation_report,
)
from src.evaluation.metric import compute_macro_ap_riou

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def collect_frame_ids(predictions_by_clip: PredictionsByClip) -> set[str]:
    """Return all frame IDs present in loaded predictions."""
    return {
        frame_id
        for predictions_by_frame in predictions_by_clip.values()
        for frame_id in predictions_by_frame
    }


def find_frame_image(images_dir: str | Path, clip_id: str, frame_id: str) -> Path:
    """Find a frame image in either flat or per-clip dataset layouts."""
    root = Path(images_dir)
    candidates: list[Path] = []
    for extension in _IMAGE_EXTENSIONS:
        candidates.append(root / f"{frame_id}{extension}")
        candidates.append(root / clip_id / f"{frame_id}{extension}")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"image not found for frame_id={frame_id}")


def detection_to_polygon(detection: Detection) -> np.ndarray:
    """Convert an evaluation detection into a 4-point polygon."""
    return obb_to_polygon(
        (
            detection.cx,
            detection.cy,
            detection.width,
            detection.height,
            detection.angle_deg,
        )
    )


def compute_homographies_from_images(
    predictions_by_clip: PredictionsByClip,
    images_dir: str | Path,
) -> HomographiesByClip:
    """Compute previous-to-current homographies for every evaluated clip."""
    homographies_by_clip: HomographiesByClip = {}
    for clip_id, predictions_by_frame in sorted(predictions_by_clip.items()):
        previous_gray: np.ndarray | None = None
        previous_polygons: list[np.ndarray] = []
        homographies_by_frame = {}

        for frame_id in sorted(predictions_by_frame):
            image_path = find_frame_image(images_dir, clip_id, frame_id)
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"could not read image: {image_path}")

            if previous_gray is not None:
                homography, _ = estimate_homography(
                    previous_gray,
                    image,
                    previous_polygons,
                )
                homographies_by_frame[frame_id] = homography

            previous_gray = image
            previous_polygons = [
                detection_to_polygon(detection)
                for detection in predictions_by_frame[frame_id]
            ]

        homographies_by_clip[clip_id] = homographies_by_frame
    return homographies_by_clip


def evaluate_inference_json_dir(
    *,
    predictions_dir: str | Path,
    images_dir: str | Path,
    ground_truth_csv: str | Path,
    output_json: str | Path,
    condition_name: str,
    inference_metrics_json: str | Path | None = None,
    max_clips: int = 0,
    class_id_offset: int = 1,
    class_id_map: dict[int, int] | None = None,
    frame_idx_width: int = 4,
    skip_motion_filter: bool = False,
) -> Path:
    """Load saved inference JSONs, filter static detections, and write metrics."""
    predictions_by_clip = load_inference_predictions_dir(
        predictions_dir,
        max_clips=max_clips,
        class_id_offset=class_id_offset,
        class_id_map=class_id_map,
        frame_idx_width=frame_idx_width,
    )
    frame_ids = collect_frame_ids(predictions_by_clip)
    ground_truths = load_ground_truth_csv(ground_truth_csv, frame_ids)
    if skip_motion_filter:
        unfiltered_predictions = {
            frame_id: detections
            for predictions_by_frame in predictions_by_clip.values()
            for frame_id, detections in predictions_by_frame.items()
        }
        macro_score, metric_details = compute_macro_ap_riou(
            predictions_for_metric(unfiltered_predictions),
            ground_truths,
        )
        evaluation = PipelineEvaluation(
            macro_score,
            unfiltered_predictions,
            {},
            metric_details,
        )
    else:
        homographies_by_clip = compute_homographies_from_images(
            predictions_by_clip,
            images_dir,
        )
        evaluation = evaluate_dataset(
            predictions_by_clip,
            homographies_by_clip,
            ground_truths,
        )
    inference_metrics = load_inference_metrics(inference_metrics_json)
    return write_evaluation_report(
        output_json,
        {condition_name: evaluation},
        inference_metrics=inference_metrics,
    )


def load_inference_metrics(
    inference_metrics_json: str | Path | None,
) -> dict[str, Any] | None:
    """Load optional inference speed and hardware metrics."""
    if inference_metrics_json is None:
        return None
    path = Path(inference_metrics_json)
    if not path.is_file():
        raise FileNotFoundError(f"inference metrics JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("inference metrics JSON must contain an object")
    return data


def parse_class_id_map(raw_value: str | None) -> dict[int, int] | None:
    """Parse comma-separated saved:official class mappings."""
    if not raw_value:
        return None
    mapping: dict[int, int] = {}
    for pair in raw_value.split(","):
        saved, separator, official = pair.strip().partition(":")
        if not separator:
            raise ValueError("--class-id-map entries must use saved:official syntax")
        saved_id = int(saved)
        official_id = int(official)
        if official_id not in range(1, 10):
            raise ValueError("official class IDs must be from 1 through 9")
        mapping[saved_id] = official_id
    return mapping


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate saved *_predictions.json inference outputs."
    )
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--ground-truth-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--condition-name", default="Base 0")
    parser.add_argument(
        "--inference-metrics-json",
        help="Optional inference_metrics.json file with FPS and VRAM diagnostics.",
    )
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--class-id-offset", type=int, default=1)
    parser.add_argument(
        "--class-id-map",
        help="Explicit saved-to-official class map, for example 0:1,6:7.",
    )
    parser.add_argument("--frame-idx-width", type=int, default=4)
    parser.add_argument(
        "--skip-motion-filter",
        action="store_true",
        help="Evaluate raw predictions without homography motion filtering.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    report_path = evaluate_inference_json_dir(
        predictions_dir=args.predictions_dir,
        images_dir=args.images_dir,
        ground_truth_csv=args.ground_truth_csv,
        output_json=args.output_json,
        condition_name=args.condition_name,
        inference_metrics_json=args.inference_metrics_json,
        max_clips=args.max_clips,
        class_id_offset=args.class_id_offset,
        class_id_map=parse_class_id_map(args.class_id_map),
        frame_idx_width=args.frame_idx_width,
        skip_motion_filter=args.skip_motion_filter,
    )
    print(f"Evaluation report written to: {report_path}")


if __name__ == "__main__":
    main()
