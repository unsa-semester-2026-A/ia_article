"""Tests for semantic mask selection and RGBA crop extraction."""

from pathlib import Path

import cv2
import numpy as np
from src.augmentation.masking import (
    MaskCandidate,
    choose_clean_mask,
    write_rgba_crop,
)


def test_choose_clean_mask_rejects_spill_into_excluded_object() -> None:
    """A mask touching a neighbouring labelled vehicle is never accepted."""
    clean = np.zeros((20, 20), dtype=np.uint8)
    clean[6:12, 6:12] = 1
    spilled = clean.copy()
    spilled[2:5, 2:5] = 1
    excluded = np.zeros((20, 20), dtype=np.uint8)
    excluded[2:5, 2:5] = 1

    decision = choose_clean_mask(
        [MaskCandidate(spilled, 0.99), MaskCandidate(clean, 0.5)],
        (4, 4, 14, 14),
        excluded_mask=excluded,
    )

    assert decision.accepted
    assert decision.score == 0.5
    assert decision.mask is not None
    assert decision.mask[3, 3] == 0


def test_write_rgba_crop_preserves_only_semantic_alpha(tmp_path: Path) -> None:
    """The written crop uses the semantic mask rather than an OBB rectangle."""
    image = np.full((12, 16, 3), 80, dtype=np.uint8)
    mask = np.zeros((12, 16), dtype=np.uint8)
    cv2.circle(mask, (8, 6), 3, 255, -1)

    output = write_rgba_crop(image, mask, tmp_path / "crop.png")
    crop = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)

    assert crop is not None and crop.shape[2] == 4
    assert np.any(crop[:, :, 3] == 0)
    assert np.any(crop[:, :, 3] == 255)
