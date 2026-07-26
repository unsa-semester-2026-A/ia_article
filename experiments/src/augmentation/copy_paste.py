"""Pixel-preserving semantic-mask copy-paste rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CounterfactualScore:
    """Detector evidence that an inserted object, not its background, was found."""

    composite_score: float
    background_score: float

    @property
    def delta(self) -> float:
        """Return the detection gain attributable to the inserted object."""
        return self.composite_score - self.background_score


def feather_alpha(alpha: np.ndarray, radius_px: int = 1) -> np.ndarray:
    """Return a minimally softened alpha mask without expanding its support."""
    if alpha.ndim != 2:
        raise ValueError("Alpha mask must be two-dimensional")
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    if radius_px == 0:
        return alpha.copy()
    blurred = cv2.GaussianBlur(alpha, (0, 0), sigmaX=radius_px)
    support = cv2.dilate(
        alpha,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1,) * 2),
    )
    return np.where(support > 0, blurred, 0).astype(np.uint8)


def composite_semantic_foreground(
    background: np.ndarray,
    foreground_bgra: np.ndarray,
    *,
    feather_px: int = 1,
) -> np.ndarray:
    """Composite only semantic foreground pixels onto an unchanged background."""
    if background.shape[:2] != foreground_bgra.shape[:2]:
        raise ValueError("Background and foreground must share frame dimensions")
    alpha = feather_alpha(foreground_bgra[:, :, 3], feather_px)
    weight = alpha.astype(np.float32)[..., None] / 255.0
    return (
        foreground_bgra[:, :, :3].astype(np.float32) * weight
        + background.astype(np.float32) * (1.0 - weight)
    ).astype(np.uint8)


def background_is_unchanged(
    background: np.ndarray,
    composite: np.ndarray,
    alpha: np.ndarray,
    *,
    feather_px: int = 1,
) -> bool:
    """Check exact background pass-through outside the compositing support."""
    support = feather_alpha(alpha, feather_px) > 0
    return bool(np.array_equal(background[~support], composite[~support]))


def lab_ring_distance(
    source: np.ndarray,
    source_alpha: np.ndarray,
    target: np.ndarray,
    target_alpha: np.ndarray,
    *,
    ring_px: int = 4,
) -> float:
    """Measure local source/target appearance mismatch around semantic masks."""
    if ring_px < 1:
        raise ValueError("ring_px must be positive")

    def _ring(alpha: np.ndarray) -> np.ndarray:
        expanded = cv2.dilate(
            alpha,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1,) * 2),
        )
        return (expanded > 0) & ~(alpha > 0)

    source_ring, target_ring = _ring(source_alpha), _ring(target_alpha)
    if not source_ring.any() or not target_ring.any():
        return float("inf")
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)[source_ring]
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)[target_ring]
    return float(
        np.linalg.norm(np.median(source_lab, axis=0) - np.median(target_lab, axis=0))
    )
