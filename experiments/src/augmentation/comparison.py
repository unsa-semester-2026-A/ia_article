"""Build a small, auditable Raw-versus-LaMa IC-Light comparison batch."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from src.augmentation.pipeline import OBB, crop_rgba, load_yolo_obb
from src.augmentation.render import warp_crop_to_slot
from src.augmentation.smoke import run_smoke_batch

SYNTHETIC_CLASS_NAMES = {
    1: "combi",
    2: "microbus",
    4: "omnibus",
    5: "articulado",
    7: "mototaxi",
}


def _image_path(directory: Path, stem: str) -> Path | None:
    """Return a readable image path for one frame stem."""
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def _select_sources(
    labels_dir: Path, raw_images_dir: Path, class_ids: tuple[int, ...]
) -> dict[int, tuple[Path, OBB]]:
    """Select the first readable crop for every requested class.

    The smoke only needs coverage of the five planned vehicle forms. Unlike the
    production crop-ranking workflow, it stops immediately once that coverage
    is found and never scans remaining label files.
    """
    selected: dict[int, tuple[Path, OBB]] = {}
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = _image_path(raw_images_dir, label_path.stem)
        if image_path is None:
            continue
        for box in load_yolo_obb(label_path):
            if box.class_id not in class_ids:
                continue
            selected.setdefault(box.class_id, (image_path, box))
        if len(selected) == len(class_ids):
            break
    missing = sorted(set(class_ids) - set(selected))
    if missing:
        raise RuntimeError(f"No readable crop source was found for classes {missing}")
    return selected


def _select_target_frames(
    labels_dir: Path,
    raw_images_dir: Path,
    lama_images_dir: Path,
    objects_per_frame: tuple[int, ...],
) -> list[tuple[str, Path, Path, list[OBB]]]:
    """Choose distinct paired Raw/LaMa frames with enough target OBBs."""
    selected = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        if len(selected) == len(objects_per_frame):
            break
        required = objects_per_frame[len(selected)]
        raw = _image_path(raw_images_dir, label_path.stem)
        lama = _image_path(lama_images_dir, label_path.stem)
        boxes = load_yolo_obb(label_path)
        if raw is not None and lama is not None and len(boxes) >= required:
            selected.append((label_path.stem, raw, lama, boxes[:required]))
    if len(selected) != len(objects_per_frame):
        required = objects_per_frame[len(selected)]
        raise RuntimeError(
            f"Could not find a paired Raw/LaMa frame with {required} OBBs"
        )
    return selected


def _merge_layers(layers: list[np.ndarray]) -> np.ndarray:
    """Merge non-overlapping full-frame BGRA vehicle layers."""
    merged = np.zeros((360, 640, 4), dtype=np.uint8)
    for layer in layers:
        alpha = layer[:, :, 3] > 0
        merged[alpha] = layer[alpha]
    return merged


def prepare_three_frame_comparison(
    *,
    labels_dir: Path,
    raw_images_dir: Path,
    lama_images_dir: Path,
    output_dir: Path,
    class_ids: tuple[int, ...] = tuple(SYNTHETIC_CLASS_NAMES),
    frame_count: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create comparable LaMa-rendered examples covering all target classes.

    The target OBBs are smoke-test locations, not accepted production static
    slots. A full production run must instead use the stationary-slot manifest.
    """
    if len(class_ids) != 5:
        raise ValueError(
            "The comparison is defined for the five planned synthetic classes"
        )
    source_by_class = _select_sources(labels_dir, raw_images_dir, class_ids)
    if frame_count < len(class_ids):
        raise ValueError("frame_count must cover every planned synthetic class")
    target_frames = _select_target_frames(
        labels_dir,
        raw_images_dir,
        lama_images_dir,
        objects_per_frame=(1,) * frame_count,
    )
    crops_dir = output_dir / "source_crops"
    comparison_dir = output_dir / "comparisons"
    jobs: list[dict[str, Any]] = []
    assignments = tuple(
        (class_ids[index % len(class_ids)],) for index in range(frame_count)
    )
    frames: list[dict[str, Any]] = []
    for frame_index, ((frame_id, raw, lama, target_boxes), assigned) in enumerate(
        zip(target_frames, assignments, strict=True)
    ):
        layers = []
        class_rows = []
        for class_id, target_box in zip(assigned, target_boxes, strict=True):
            source_image, source_box = source_by_class[class_id]
            crop_path = crop_rgba(
                source_image,
                source_box,
                crops_dir / f"class_{class_id}_{source_image.stem}.png",
            )
            layers.append(
                warp_crop_to_slot(
                    crop_path, [list(point) for point in target_box.points]
                )
            )
            class_rows.append(
                {
                    "class_id": class_id,
                    "class_name": SYNTHETIC_CLASS_NAMES[class_id],
                    "source_frame_id": source_image.stem,
                    "target_obb": [list(point) for point in target_box.points],
                }
            )
        foreground = _merge_layers(layers)
        frame_dir = comparison_dir / f"frame_{frame_index:02d}_{frame_id}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw, frame_dir / "raw_original.jpg")
        shutil.copy2(lama, frame_dir / "lama_background.jpg")
        if not cv2.imwrite(str(frame_dir / "inserted_vehicles.png"), foreground):
            raise OSError(f"Could not write foreground input for {frame_id}")
        frame_metadata = {
            "frame_index": frame_index,
            "frame_id": frame_id,
            "classes": class_rows,
            "raw_original": str(frame_dir / "raw_original.jpg"),
            "lama_background": str(frame_dir / "lama_background.jpg"),
            "inserted_vehicles": str(frame_dir / "inserted_vehicles.png"),
            "direct_overlay": str(frame_dir / "direct_overlay.jpg"),
            "iclight_full": str(frame_dir / "iclight_full.jpg"),
            "iclight_soft_composite": str(frame_dir / "iclight_soft_composite.jpg"),
        }
        frames.append(frame_metadata)
        # The smoke renders only the LaMa condition. The paired unmodified Raw
        # and LaMa inputs above are preserved for visual comparison, avoiding an
        # unnecessary second diffusion invocation for each frame.
        jobs.append(
            {
                "id": f"comparison/frame_{frame_index:02d}_{frame_id}/iclight_relight",
                "foreground_bgra": foreground,
                "background_path": lama,
                "seed": 20260726 + frame_index,
                "frame_id": frame_id,
                "background_variant": "lama",
                "class_ids": list(assigned),
                "class_names": [
                    SYNTHETIC_CLASS_NAMES[class_id] for class_id in assigned
                ],
                "direct_overlay_path": str(frame_dir / "direct_overlay.jpg"),
                "generated_output_path": str(frame_dir / "iclight_full.jpg"),
                "output_path": str(frame_dir / "iclight_soft_composite.jpg"),
            }
        )
    manifest = {
        "scope": "ten-example smoke comparison; target OBBs are not production slots",
        "class_coverage": {str(key): SYNTHETIC_CLASS_NAMES[key] for key in class_ids},
        "frames": frames,
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return jobs, manifest


def run_three_frame_comparison(
    client: Any,
    *,
    labels_dir: Path,
    raw_images_dir: Path,
    lama_images_dir: Path,
    output_dir: Path,
    working_size: tuple[int, int] = (576, 320),
    steps: int = 20,
) -> dict[str, Any]:
    """Prepare and render the ten-example Raw/LaMa comparison smoke batch."""
    jobs, manifest = prepare_three_frame_comparison(
        labels_dir=labels_dir,
        raw_images_dir=raw_images_dir,
        lama_images_dir=lama_images_dir,
        output_dir=output_dir,
        frame_count=10,
    )
    report = run_smoke_batch(
        client, jobs, output_dir, working_size=working_size, steps=steps
    )
    report["comparison_manifest"] = str(output_dir / "comparison_manifest.json")
    report["class_coverage"] = manifest["class_coverage"]
    (output_dir / "iclight_smoke_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
