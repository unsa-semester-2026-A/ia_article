"""Unit tests for the homography and exclusion mask utility module."""

import numpy as np
import pytest
from src.utils.homography import (
    create_exclusion_mask_from_predictions,
    estimate_interframe_homography,
)


def test_create_exclusion_mask_empty_predictions() -> None:
    """Verify that an empty or None prediction list returns a solid white mask."""
    shape = (100, 100)

    # Test with None
    mask_none = create_exclusion_mask_from_predictions(shape, None)
    assert mask_none.shape == shape
    assert mask_none.dtype == np.uint8
    assert np.all(mask_none == 255)

    # Test with empty array
    mask_empty = create_exclusion_mask_from_predictions(shape, np.array([]))
    assert np.all(mask_empty == 255)


def test_create_exclusion_mask_draws_polygons() -> None:
    """Verify that vehicle boxes are drawn as black (0) polygons on the white mask."""
    shape = (100, 100)
    # Box: cx=50, cy=50, w=20, h=10, angle_rad=0.0
    predictions = np.array([[50.0, 50.0, 20.0, 10.0, 0.0]])

    mask = create_exclusion_mask_from_predictions(shape, predictions)
    assert mask.shape == shape

    # The center of the box (50, 50) should be painted black (0)
    assert mask[50, 50] == 0
    # Corners of the image should remain white (255)
    assert mask[0, 0] == 255
    assert mask[99, 99] == 255


def test_estimate_homography_fallback_on_flat_images() -> None:
    """Verify that the estimation returns the Identity matrix when features are insufficient."""
    # Create two featureless flat black images
    prev_gray = np.zeros((100, 100), dtype=np.uint8)
    curr_gray = np.zeros((100, 100), dtype=np.uint8)

    H = estimate_interframe_homography(prev_gray, curr_gray)

    # Should fall back to the 3x3 identity matrix
    assert H.shape == (3, 3)
    assert np.allclose(H, np.eye(3))


def test_estimate_homography_happy_path() -> None:
    """Verify that the homography is successfully estimated for shifted textured images."""
    import cv2

    # Generate a random textured pattern to ensure ORB finds keypoints
    rng = np.random.default_rng(2026)
    pattern = rng.integers(0, 255, size=(400, 400), dtype=np.uint8)

    # Blur to create continuous structures
    prev_gray = cv2.GaussianBlur(pattern, (5, 5), 0)

    # Translate image by 5 pixels along X and 3 pixels along Y
    translation_matrix = np.float32([[1, 0, 5], [0, 1, 3]])
    curr_gray = cv2.warpAffine(prev_gray, translation_matrix, (400, 400))

    # Estimate homography
    H = estimate_interframe_homography(prev_gray, curr_gray)

    # H should be approximately equivalent to translation:
    # [[1, 0, 5], [0, 1, 3], [0, 0, 1]]
    assert H.shape == (3, 3)
    assert H[2, 0] == pytest.approx(0.0, abs=1e-4)
    assert H[2, 1] == pytest.approx(0.0, abs=1e-4)
    assert H[2, 2] == pytest.approx(1.0, abs=1e-4)

    # Check translation components
    assert H[0, 2] == pytest.approx(5.0, abs=0.5)
    assert H[1, 2] == pytest.approx(3.0, abs=0.5)


def test_estimate_homography_low_matches_mocked() -> None:
    """Verify fallback to Identity matrix when matches are low (mocked)."""
    from unittest.mock import MagicMock, patch

    prev_gray = np.zeros((100, 100), dtype=np.uint8)
    curr_gray = np.zeros((100, 100), dtype=np.uint8)

    # Create keypoint lists with .pt attribute
    kps_prev = [MagicMock(pt=(10.0, 10.0)) for _ in range(20)]
    kps_curr = [MagicMock(pt=(10.0, 10.0)) for _ in range(20)]

    with patch("cv2.ORB_create") as mock_orb_create:
        mock_orb = MagicMock()
        mock_orb.detectAndCompute.side_effect = [
            (kps_prev, np.ones((20, 32), dtype=np.uint8)),
            (kps_curr, np.ones((20, 32), dtype=np.uint8)),
        ]
        mock_orb_create.return_value = mock_orb

        with patch("cv2.BFMatcher") as mock_bf_create:
            mock_bf = MagicMock()
            # Return list of 5 matches with numeric distance to support sorting comparison
            mock_bf.match.return_value = [
                MagicMock(distance=float(i)) for i in range(5)
            ]
            mock_bf_create.return_value = mock_bf

            H = estimate_interframe_homography(prev_gray, curr_gray)
            assert np.allclose(H, np.eye(3))


def test_estimate_homography_ransac_failure_mocked() -> None:
    """Verify fallback to Identity matrix when RANSAC fails (mocked)."""
    from unittest.mock import MagicMock, patch

    prev_gray = np.zeros((100, 100), dtype=np.uint8)
    curr_gray = np.zeros((100, 100), dtype=np.uint8)

    kps_prev = [MagicMock(pt=(10.0, 10.0)) for _ in range(20)]
    kps_curr = [MagicMock(pt=(10.0, 10.0)) for _ in range(20)]

    with patch("cv2.ORB_create") as mock_orb_create:
        mock_orb = MagicMock()
        mock_orb.detectAndCompute.side_effect = [
            (kps_prev, np.ones((20, 32), dtype=np.uint8)),
            (kps_curr, np.ones((20, 32), dtype=np.uint8)),
        ]
        mock_orb_create.return_value = mock_orb

        with patch("cv2.BFMatcher") as mock_bf_create:
            mock_bf = MagicMock()
            # Return 15 matches (>= 10) with numeric distance to support sorting comparison
            mock_bf.match.return_value = [
                MagicMock(queryIdx=i, trainIdx=i, distance=float(i)) for i in range(15)
            ]
            mock_bf_create.return_value = mock_bf

            # Mock findHomography to return (None, None)
            with patch("cv2.findHomography", return_value=(None, None)):
                H = estimate_interframe_homography(prev_gray, curr_gray)
                assert np.allclose(H, np.eye(3))
