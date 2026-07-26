"""Semantic vehicle-mask extraction for copy-paste augmentation.

The production augmentation path must never use an oriented bounding box as an
alpha mask.  This module isolates the vehicle with SAM and exposes small,
deterministic quality checks that can be tested without downloading a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from src.augmentation.pipeline import OBB


@dataclass(frozen=True)
class MaskCandidate:
    """One semantic-mask candidate returned by a segmentation backend."""

    mask: np.ndarray
    score: float


@dataclass(frozen=True)
class MaskDecision:
    """Accepted semantic mask or a stable reason for rejecting it."""

    mask: np.ndarray | None
    score: float | None
    reason: str | None

    @property
    def accepted(self) -> bool:
        """Return whether this decision contains an accepted mask."""
        return self.mask is not None


class Masker(Protocol):
    """Protocol used by the pipeline so unit tests need no GPU model."""

    def predict(
        self, image: np.ndarray, box_xyxy: tuple[int, int, int, int]
    ) -> list[MaskCandidate]:
        """Return candidate masks in the coordinate system of ``image``."""


def obb_bounds(
    obb: OBB, width: int, height: int, padding: int = 4
) -> tuple[int, int, int, int]:
    """Return a clipped axis-aligned prompt rectangle for an oriented box."""
    points = np.asarray(obb.points, dtype=np.float32)
    x1 = max(0, int(np.floor(points[:, 0].min())) - padding)
    y1 = max(0, int(np.floor(points[:, 1].min())) - padding)
    x2 = min(width, int(np.ceil(points[:, 0].max())) + padding)
    y2 = min(height, int(np.ceil(points[:, 1].max())) + padding)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("OBB prompt has no drawable area")
    return x1, y1, x2, y2


def choose_clean_mask(
    candidates: list[MaskCandidate],
    prompt_box: tuple[int, int, int, int],
    *,
    excluded_mask: np.ndarray | None = None,
) -> MaskDecision:
    """Choose the strongest single-component mask compatible with the prompt.

    A confidence value alone is not enough for vehicle extraction.  Candidates
    that spill into another labelled object, are empty, or split into multiple
    substantial components are rejected before they can become a crop.
    """
    if not candidates:
        return MaskDecision(None, None, "sam_returned_no_masks")
    x1, y1, x2, y2 = prompt_box
    prompt_area = max(1, (x2 - x1) * (y2 - y1))
    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    for candidate in ranked:
        mask = (np.asarray(candidate.mask) > 0).astype(np.uint8)
        if mask.ndim != 2 or not mask.any():
            continue
        if excluded_mask is not None and np.any((mask > 0) & (excluded_mask > 0)):
            continue
        components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        areas = stats[1:, cv2.CC_STAT_AREA] if components > 1 else np.array([])
        if not len(areas):
            continue
        largest = int(areas.max())
        if largest / int(mask.sum()) < 0.95:
            continue
        if largest > prompt_area * 1.25:
            continue
        clean = np.where(labels == int(np.argmax(areas) + 1), 255, 0).astype(np.uint8)
        return MaskDecision(clean, float(candidate.score), None)
    return MaskDecision(None, None, "no_candidate_passed_geometry_checks")


def write_rgba_crop(image: np.ndarray, mask: np.ndarray, output_path: Path) -> Path:
    """Write the tight RGBA crop defined by a semantic mask."""
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Cannot write an empty semantic mask")
    x, y, width, height = cv2.boundingRect(points)
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), rgba[y : y + height, x : x + width]):
        raise OSError(f"Could not write semantic RGBA crop: {output_path}")
    return output_path


class SamBoxMasker:
    """Lazy Ultralytics SAM adapter using an OBB-derived box prompt.

    The import and checkpoint download happen only in Kaggle/Colab production;
    local unit tests inject a lightweight ``Masker`` implementation instead.
    """

    def __init__(self, model_name: str = "sam_b.pt") -> None:
        """Create a SAM adapter without loading weights until first use."""
        self.model_name = model_name
        self._model: object | None = None

    def _load(self) -> object:
        """Load the Ultralytics SAM model lazily."""
        if self._model is None:
            try:
                from ultralytics import SAM
            except ImportError as exc:  # pragma: no cover - cloud dependency
                raise RuntimeError("Ultralytics with SAM support is required") from exc
            self._model = SAM(self.model_name)
        return self._model

    def predict(
        self, image: np.ndarray, box_xyxy: tuple[int, int, int, int]
    ) -> list[MaskCandidate]:
        """Run SAM on a padded, enlarged ROI and restore full-frame masks."""
        model = self._load()
        x1, y1, x2, y2 = box_xyxy
        margin = max(x2 - x1, y2 - y1)
        roi_x1, roi_y1 = max(0, x1 - margin), max(0, y1 - margin)
        roi_x2, roi_y2 = (
            min(image.shape[1], x2 + margin),
            min(image.shape[0], y2 + margin),
        )
        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
        scale = 512 / max(roi.shape[:2])
        resized = cv2.resize(
            roi,
            (round(roi.shape[1] * scale), round(roi.shape[0] * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
        prompt = [
            (x1 - roi_x1) * scale,
            (y1 - roi_y1) * scale,
            (x2 - roi_x1) * scale,
            (y2 - roi_y1) * scale,
        ]
        # Ultralytics SAM expects one xyxy prompt as a flat four-value list.
        # See https://docs.ultralytics.com/models/sam/#sam-prediction-example
        results = model(resized, bboxes=prompt, verbose=False)
        if not results or results[0].masks is None:
            return []
        masks = results[0].masks.data.detach().cpu().numpy()
        # Prompted SAM returns masks but does not promise detector confidence
        # boxes. Geometry gates below are the reproducible acceptance criterion.
        scores = [1.0] * len(masks)
        candidates = []
        for index, mask in enumerate(masks):
            restored_roi = cv2.resize(
                mask.astype(np.uint8),
                (roi.shape[1], roi.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            restored = np.zeros(image.shape[:2], dtype=np.uint8)
            restored[roi_y1:roi_y2, roi_x1:roi_x2] = restored_roi
            candidates.append(
                MaskCandidate(restored, float(scores[min(index, len(scores) - 1)]))
            )
        return candidates
