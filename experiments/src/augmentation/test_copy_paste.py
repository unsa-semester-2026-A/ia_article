"""Tests for background-preserving semantic composition."""

import numpy as np
from src.augmentation.copy_paste import (
    background_is_unchanged,
    composite_semantic_foreground,
    lab_ring_distance,
)


def test_composition_changes_no_pixels_outside_semantic_support() -> None:
    """Copy-paste preserves all road pixels outside the feathered object."""
    background = np.full((30, 30, 3), 30, dtype=np.uint8)
    foreground = np.zeros((30, 30, 4), dtype=np.uint8)
    foreground[12:18, 12:18, :3] = (80, 120, 160)
    foreground[12:18, 12:18, 3] = 255

    composite = composite_semantic_foreground(background, foreground)

    assert background_is_unchanged(background, composite, foreground[:, :, 3])
    assert not np.array_equal(composite[15, 15], background[15, 15])


def test_lab_ring_distance_prefers_matching_road_conditions() -> None:
    """Appearance matching is based on surrounding pixels, not detector output."""
    source = np.full((20, 20, 3), 50, dtype=np.uint8)
    target_near = np.full((20, 20, 3), 55, dtype=np.uint8)
    target_far = np.full((20, 20, 3), 190, dtype=np.uint8)
    alpha = np.zeros((20, 20), dtype=np.uint8)
    alpha[8:12, 8:12] = 255

    assert lab_ring_distance(source, alpha, target_near, alpha) < lab_ring_distance(
        source, alpha, target_far, alpha
    )
