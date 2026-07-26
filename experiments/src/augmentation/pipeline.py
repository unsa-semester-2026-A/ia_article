"""Candidate selection, IC-Light job preparation, validation and archive packaging.

The module deliberately keeps model inference behind injectable callables. This
makes all deterministic data rules testable on CPU while Colab supplies IC-Light
and the frozen Base1 detector at rendering time.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
FRAME_WIDTH = 640
FRAME_HEIGHT = 360


@dataclass(frozen=True)
class OBB:
    """One class-labelled oriented bounding box in pixel coordinates."""

    class_id: int
    points: tuple[tuple[float, float], ...]

    @property
    def area(self) -> float:
        """Return polygon area in pixels squared."""
        return float(abs(cv2.contourArea(np.asarray(self.points, dtype=np.float32))))

    @property
    def centroid(self) -> tuple[float, float]:
        """Return the arithmetic centre of the four corners."""
        points = np.asarray(self.points, dtype=np.float32)
        return float(points[:, 0].mean()), float(points[:, 1].mean())


@dataclass(frozen=True)
class TrackObservation:
    """One OBB observation linked to a temporal source track."""

    track_id: str
    frame_id: str
    clip_id: str
    obb: OBB


@dataclass(frozen=True)
class CropCandidate:
    """Best usable real crop selected for one source track."""

    track_id: str
    frame_id: str
    clip_id: str
    class_id: int
    obb: OBB
    crop_path: str


@dataclass(frozen=True)
class StaticSlot:
    """One pseudo-labelled parked-vehicle location eligible for insertion."""

    slot_id: str
    frame_id: str
    clip_id: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class AugmentationConfig:
    """Policy fixed by the calibration pilot before final rendering."""

    tau: float = 0.02
    reuse_cap: int = 10
    budget_fraction: float = 1.0
    seed: int = 42
    max_objects_per_image: int = 3

    def __post_init__(self) -> None:
        """Reject policy values that violate the pre-registered study bounds."""
        if self.tau not in {0.01, 0.02, 0.04}:
            raise ValueError("tau must be one of 0.01, 0.02 or 0.04")
        if self.reuse_cap not in {1, 2, 5, 10}:
            raise ValueError("reuse_cap must be one of 1, 2, 5 or 10")
        if self.budget_fraction not in {0.25, 0.5, 1.0}:
            raise ValueError("budget_fraction must be 0.25, 0.5 or 1.0")


def split_frame_id(frame_id: str) -> tuple[str, int]:
    """Split ``v_hash_0000`` into its clip id and numeric frame index."""
    clip_id, separator, index = frame_id.rpartition("_")
    if not separator or not clip_id or not index.isdigit():
        raise ValueError(f"Invalid SMART frame id: {frame_id}")
    return clip_id, int(index)


def load_yolo_obb(
    path: Path, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT
) -> list[OBB]:
    """Load one normalized YOLO-OBB label file into pixel polygons."""
    boxes: list[OBB] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if not parts:
            continue
        if len(parts) != 9:
            raise ValueError(f"Invalid YOLO-OBB line in {path}: {raw_line}")
        values = [float(value) for value in parts[1:]]
        points = tuple(
            (values[index] * width, values[index + 1] * height)
            for index in range(0, 8, 2)
        )
        boxes.append(OBB(class_id=int(parts[0]), points=points))
    return boxes


def obb_to_yolo_line(
    obb: OBB, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT
) -> str:
    """Encode a pixel OBB as one normalized YOLO-OBB line."""
    values = [str(obb.class_id)]
    for x, y in obb.points:
        values.extend(
            (f"{np.clip(x / width, 0, 1):.6f}", f"{np.clip(y / height, 0, 1):.6f}")
        )
    return " ".join(values)


def polygon_overlap(first: OBB, second: OBB) -> float:
    """Return the intersection area of two convex OBB polygons."""
    first_poly = np.asarray(first.points, dtype=np.float32)
    second_poly = np.asarray(second.points, dtype=np.float32)
    area, _ = cv2.intersectConvexConvex(first_poly, second_poly)
    return float(max(area, 0.0))


def is_inside_frame(
    obb: OBB, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT
) -> bool:
    """Return whether every OBB corner stays inside the image."""
    return all(0 <= x < width and 0 <= y < height for x, y in obb.points)


def crop_rgba(image_path: Path, obb: OBB, output_path: Path) -> Path:
    """Extract an alpha-masked crop using the OBB polygon.

    Args:
        image_path: Source BGR image.
        obb: Object polygon to retain.
        output_path: PNG destination for the RGBA crop.

    Returns:
        Written crop path.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load crop source: {image_path}")
    polygon = np.asarray(obb.points, dtype=np.int32)
    x, y, width, height = cv2.boundingRect(polygon)
    alpha = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(alpha, polygon, 255)
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), rgba[y : y + height, x : x + width]):
        raise OSError(f"Could not write crop: {output_path}")
    return output_path


def quota_by_class(
    real_counts: dict[int, int],
    track_counts: dict[int, int],
    config: AugmentationConfig,
) -> dict[int, int]:
    """Compute the policy-constrained synthetic count per class."""
    total = sum(real_counts.values())
    target = math.ceil(config.tau * total)
    quotas: dict[int, int] = {}
    for class_id, real_count in real_counts.items():
        eligible = total and real_count / total < config.tau
        requested = min(
            real_count,
            max(0, target - real_count),
            config.reuse_cap * track_counts.get(class_id, 0),
        )
        quotas[class_id] = (
            math.floor(requested * config.budget_fraction) if eligible else 0
        )
    return quotas


class SyntheticDatasetBuilder:
    """Build deterministic crop/slot manifests and self-contained train archives."""

    def __init__(self, config: AugmentationConfig) -> None:
        """Create a builder with one immutable selected policy."""
        self.config = config

    def train_clip_ids(self, metadata_path: Path) -> set[str]:
        """Load the clip-level train split and reject malformed metadata."""
        import pandas as pd

        metadata = pd.read_csv(metadata_path)
        required = {"clip_id", "split"}
        if not required.issubset(metadata.columns):
            raise ValueError(
                f"Metadata must contain {required}; found {set(metadata.columns)}"
            )
        return set(metadata.loc[metadata["split"] == "train", "clip_id"])

    def collect_slots(
        self, static_json_path: Path, train_clips: set[str], labels_dir: Path
    ) -> list[StaticSlot]:
        """Load safe stationary slots with no overlap against real train boxes."""
        static_map = json.loads(static_json_path.read_text(encoding="utf-8"))
        slots: list[StaticSlot] = []
        for frame_id, raw_slots in static_map.items():
            clip_id, _ = split_frame_id(frame_id)
            if clip_id not in train_clips:
                continue
            labels_path = labels_dir / f"{frame_id}.txt"
            real_boxes = load_yolo_obb(labels_path) if labels_path.exists() else []
            for index, raw_slot in enumerate(raw_slots):
                points = self._points_from_parametric_slot(raw_slot)
                slot_box = OBB(class_id=-1, points=points)
                if is_inside_frame(slot_box) and not any(
                    polygon_overlap(slot_box, box) > 0 for box in real_boxes
                ):
                    slots.append(
                        StaticSlot(f"{frame_id}:{index}", frame_id, clip_id, points)
                    )
        return slots

    def extract_track_crops(
        self, images_dir: Path, labels_dir: Path, train_clips: set[str], crops_dir: Path
    ) -> tuple[list[CropCandidate], dict[int, int]]:
        """Link same-class observations by clip and keep one high-quality crop/track."""
        observations: dict[str, list[tuple[str, OBB]]] = {}
        real_counts: dict[int, int] = {}
        active: dict[tuple[str, int], list[tuple[str, int, tuple[float, float]]]] = {}
        next_id: dict[tuple[str, int], int] = {}
        for label_path in sorted(labels_dir.glob("*.txt")):
            frame_id = label_path.stem
            clip_id, frame_index = split_frame_id(frame_id)
            if clip_id not in train_clips:
                continue
            for box in load_yolo_obb(label_path):
                real_counts[box.class_id] = real_counts.get(box.class_id, 0) + 1
                key = (clip_id, box.class_id)
                centre = box.centroid
                candidates = [
                    item for item in active.get(key, []) if frame_index - item[1] <= 2
                ]
                match = min(
                    candidates,
                    key=lambda item: np.hypot(
                        centre[0] - item[2][0], centre[1] - item[2][1]
                    ),
                    default=None,
                )
                if (
                    match
                    and np.hypot(centre[0] - match[2][0], centre[1] - match[2][1]) <= 35
                ):
                    track_id = match[0]
                else:
                    number = next_id.get(key, 0)
                    next_id[key] = number + 1
                    track_id = f"{clip_id}:c{box.class_id}:t{number:04d}"
                active[key] = [
                    item for item in active.get(key, []) if item[0] != track_id
                ] + [(track_id, frame_index, centre)]
                observations.setdefault(track_id, []).append((frame_id, box))
        selected: list[CropCandidate] = []
        for track_id, items in sorted(observations.items()):
            frame_id, box = max(items, key=lambda item: item[1].area)
            image_path = next(
                (
                    images_dir / f"{frame_id}{ext}"
                    for ext in IMAGE_EXTENSIONS
                    if (images_dir / f"{frame_id}{ext}").exists()
                ),
                None,
            )
            if image_path is None or not is_inside_frame(box):
                continue
            all_boxes = load_yolo_obb(labels_dir / f"{frame_id}.txt")
            if any(
                other != box and polygon_overlap(box, other) > 0 for other in all_boxes
            ):
                continue
            clip_id, _ = split_frame_id(frame_id)
            crop_path = crops_dir / f"{track_id.replace(':', '_')}.png"
            crop_rgba(image_path, box, crop_path)
            selected.append(
                CropCandidate(
                    track_id, frame_id, clip_id, box.class_id, box, str(crop_path)
                )
            )
        return selected, real_counts

    @staticmethod
    def _points_from_parametric_slot(
        raw_slot: dict[str, float],
    ) -> tuple[tuple[float, float], ...]:
        """Convert pseudo-label xywhr fields to a four-corner polygon."""
        cx, cy, width, height, angle = (
            raw_slot[key] for key in ("cx", "cy", "w", "h", "angle")
        )
        radians = math.radians(float(angle))
        rotation = np.array(
            [
                [math.cos(radians), -math.sin(radians)],
                [math.sin(radians), math.cos(radians)],
            ]
        )
        local = np.array(
            [
                [-width / 2, -height / 2],
                [width / 2, -height / 2],
                [width / 2, height / 2],
                [-width / 2, height / 2],
            ],
            dtype=np.float32,
        )
        points = local @ rotation.T + np.array([cx, cy])
        return tuple((float(x), float(y)) for x, y in points)

    def package_full_dataset(
        self,
        base_images: Path,
        base_labels: Path,
        synthetic_images: Path,
        synthetic_labels: Path,
        manifest_path: Path,
        output_zip: Path,
        variant: str,
    ) -> dict[str, object]:
        """Create one self-contained training ZIP and its checksum report."""
        image_files = self._image_files(base_images) + self._image_files(
            synthetic_images
        )
        label_files = sorted(base_labels.glob("*.txt")) + sorted(
            synthetic_labels.glob("*.txt")
        )
        image_stems = {path.stem for path in image_files}
        label_stems = {path.stem for path in label_files}
        if image_stems != label_stems:
            raise ValueError(
                f"Image/label stem mismatch: images-only={image_stems - label_stems}, labels-only={label_stems - image_stems}"
            )
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            output_zip, "w", compression=zipfile.ZIP_STORED
        ) as archive:
            for path in image_files:
                archive.write(path, f"images/train/{path.name}")
            for path in label_files:
                archive.write(path, f"labels/train/{path.name}")
            archive.write(manifest_path, "manifest.csv")
        report = {
            "variant": variant,
            "zip": output_zip.name,
            "sha256": self._sha256(output_zip),
            "images": len(image_files),
            "labels": len(label_files),
            "synthetic_images": len(self._image_files(synthetic_images)),
        }
        report_path = output_zip.with_suffix(".quality_report.json")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    @staticmethod
    def _image_files(directory: Path) -> list[Path]:
        """Return direct image children sorted by filename."""
        return sorted(
            path for path in directory.iterdir() if path.suffix in IMAGE_EXTENSIONS
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        """Return a streaming SHA-256 checksum."""
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def write_manifest(rows: Iterable[dict[str, object]], path: Path) -> Path:
    """Write a deterministic CSV manifest with stable field ordering."""
    materialized = list(rows)
    fields = sorted({field for row in materialized for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)
    return path
