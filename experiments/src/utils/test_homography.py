"""Unit tests for the homography and exclusion mask utility module."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from src.utils.homography import (
    create_exclusion_mask_from_polygons,
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


def test_create_exclusion_mask_from_predictions_3d_shape_and_multiple_boxes() -> None:
    """Verify mask generation from 3D shape tuples and multiple rotated OBB boxes."""
    shape = (100, 100, 3)
    predictions = np.array(
        [
            [30.0, 30.0, 10.0, 10.0, 0.0],
            [70.0, 70.0, 10.0, 10.0, np.pi / 4],
        ]
    )

    mask = create_exclusion_mask_from_predictions(shape, predictions)
    assert mask.shape == (100, 100)
    assert mask[30, 30] == 0
    assert mask[70, 70] == 0
    assert mask[0, 0] == 255


def test_create_exclusion_mask_from_polygons_none_and_empty() -> None:
    """Verify polygon exclusion mask creation with None or empty sequence."""
    shape = (100, 100)
    mask_none = create_exclusion_mask_from_polygons(shape, None)
    assert mask_none.shape == shape
    assert np.all(mask_none == 255)

    mask_empty = create_exclusion_mask_from_polygons(shape, [])
    assert np.all(mask_empty == 255)


def test_create_exclusion_mask_from_polygons_valid() -> None:
    """Verify polygon exclusion mask creation with valid 4-vertex polygons."""
    shape = (100, 100)
    poly_list = [
        [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)],
        np.array([[50.0, 50.0], [70.0, 50.0], [70.0, 70.0], [50.0, 70.0]]),
    ]

    mask = create_exclusion_mask_from_polygons(shape, poly_list)
    assert mask.shape == shape
    assert mask[20, 20] == 0
    assert mask[60, 60] == 0
    assert mask[0, 0] == 255


@pytest.mark.parametrize(
    "invalid_poly",
    [
        [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0)],  # Triangle (3 vertices)
        [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (15.0, 25.0), (10.0, 20.0)],  # 5
        np.array([1.0, 2.0, 3.0, 4.0]),  # 1D array
    ],
)
def test_create_exclusion_mask_from_polygons_invalid_shapes(
    invalid_poly: object,
) -> None:
    """Reject polygons that do not have shape (4, 2)."""
    with pytest.raises(ValueError, match="four finite vertices"):
        create_exclusion_mask_from_polygons((100, 100), [invalid_poly])  # type: ignore[list-item]


@pytest.mark.parametrize(
    "non_finite_vertex",
    [
        np.array([[10.0, np.nan], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]),
        np.array([[10.0, 10.0], [20.0, np.inf], [20.0, 20.0], [10.0, 20.0]]),
    ],
)
def test_create_exclusion_mask_from_polygons_non_finite(
    non_finite_vertex: np.ndarray,
) -> None:
    """Reject polygons with non-finite vertex coordinates."""
    with pytest.raises(ValueError, match="four finite vertices"):
        create_exclusion_mask_from_polygons((100, 100), [non_finite_vertex])


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


def test_estimate_homography_happy_path_return_status() -> None:
    """Verify that return_status=True returns (matrix, True) on successful estimation."""
    rng = np.random.default_rng(2026)
    pattern = rng.integers(0, 255, size=(400, 400), dtype=np.uint8)
    prev_gray = cv2.GaussianBlur(pattern, (5, 5), 0)
    translation_matrix = np.float32([[1, 0, 5], [0, 1, 3]])
    curr_gray = cv2.warpAffine(prev_gray, translation_matrix, (400, 400))

    H, success = estimate_interframe_homography(
        prev_gray, curr_gray, return_status=True
    )
    assert success is True
    assert H.shape == (3, 3)


def test_estimate_interframe_homography_invalid_image_dimensions() -> None:
    """Reject images with non-2D shapes or mismatched frame dimensions."""
    img2d = np.zeros((100, 100), dtype=np.uint8)
    img3d = np.zeros((100, 100, 3), dtype=np.uint8)
    img_diff_shape = np.zeros((100, 50), dtype=np.uint8)

    with pytest.raises(ValueError, match="equally sized grayscale arrays"):
        estimate_interframe_homography(img3d, img3d)
    with pytest.raises(ValueError, match="equally sized grayscale arrays"):
        estimate_interframe_homography(img2d, img_diff_shape)


def test_estimate_interframe_homography_non_finite_images() -> None:
    """Reject image inputs containing NaN or infinite values."""
    img_valid = np.zeros((100, 100), dtype=np.float32)
    img_nan = np.full((100, 100), np.nan, dtype=np.float32)
    img_inf = np.full((100, 100), np.inf, dtype=np.float32)

    with pytest.raises(ValueError, match="finite values"):
        estimate_interframe_homography(img_nan, img_valid)
    with pytest.raises(ValueError, match="finite values"):
        estimate_interframe_homography(img_valid, img_inf)


@pytest.mark.parametrize(
    ("nfeatures", "min_kps", "min_matches"),
    [
        (0, 15, 10),
        (-5, 15, 10),
        (2500, 0, 10),
        (2500, -1, 10),
        (2500, 15, 3),  # min_matches must be >= 4
    ],
)
def test_estimate_interframe_homography_invalid_parameters(
    nfeatures: int, min_kps: int, min_matches: int
) -> None:
    """Reject invalid ORB feature counts, keypoint minimums, or match bounds."""
    img = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="invalid ORB or matching parameters"):
        estimate_interframe_homography(
            img,
            img,
            nfeatures=nfeatures,
            min_keypoints=min_kps,
            min_matches=min_matches,
        )


def test_estimate_interframe_homography_mask_options() -> None:
    """Verify mask options: explicit prev_mask, previous_polygons, and default no-mask."""
    rng = np.random.default_rng(2026)
    pattern = rng.integers(0, 255, size=(400, 400), dtype=np.uint8)
    prev_gray = cv2.GaussianBlur(pattern, (5, 5), 0)
    translation_matrix = np.float32([[1, 0, 5], [0, 1, 3]])
    curr_gray = cv2.warpAffine(prev_gray, translation_matrix, (400, 400))

    # Explicit mask
    mask = np.ones((400, 400), dtype=np.uint8) * 255
    H_mask, ok1 = estimate_interframe_homography(
        prev_gray, curr_gray, prev_mask=mask, return_status=True
    )
    assert ok1 is True
    assert H_mask.shape == (3, 3)

    # Polygons mask option
    polys = [[(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)]]
    H_poly, ok2 = estimate_interframe_homography(
        prev_gray, curr_gray, previous_polygons=polys, return_status=True
    )
    assert ok2 is True
    assert H_poly.shape == (3, 3)


def test_estimate_interframe_homography_insufficient_keypoints_mocked() -> None:
    """Verify fallback to Identity matrix when keypoint count is below minimum."""
    prev_gray = np.zeros((100, 100), dtype=np.uint8)
    curr_gray = np.zeros((100, 100), dtype=np.uint8)

    kps_prev = [MagicMock(pt=(10.0, 10.0)) for _ in range(5)]  # 5 < min_keypoints (15)
    kps_curr = [MagicMock(pt=(10.0, 10.0)) for _ in range(5)]

    with patch("cv2.ORB_create") as mock_orb_create:
        mock_orb = MagicMock()
        mock_orb.detectAndCompute.side_effect = [
            (kps_prev, np.ones((5, 32), dtype=np.uint8)),
            (kps_curr, np.ones((5, 32), dtype=np.uint8)),
        ]
        mock_orb_create.return_value = mock_orb

        H, ok = estimate_interframe_homography(prev_gray, curr_gray, return_status=True)
        assert ok is False
        assert np.allclose(H, np.eye(3))


def test_estimate_interframe_homography_insufficient_descriptors_mocked() -> None:
    """Verify fallback when detectAndCompute returns None for descriptors."""
    prev_gray = np.zeros((100, 100), dtype=np.uint8)
    curr_gray = np.zeros((100, 100), dtype=np.uint8)

    kps = [MagicMock(pt=(10.0, 10.0)) for _ in range(20)]

    with patch("cv2.ORB_create") as mock_orb_create:
        mock_orb = MagicMock()
        mock_orb.detectAndCompute.side_effect = [
            (kps, None),
            (kps, np.ones((20, 32), dtype=np.uint8)),
        ]
        mock_orb_create.return_value = mock_orb

        H, ok = estimate_interframe_homography(prev_gray, curr_gray, return_status=True)
        assert ok is False
        assert np.allclose(H, np.eye(3))


def test_estimate_homography_low_matches_mocked() -> None:
    """Verify fallback to Identity matrix when matches are low (mocked)."""
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

            H, ok = estimate_interframe_homography(
                prev_gray, curr_gray, return_status=True
            )
            assert ok is False
            assert np.allclose(H, np.eye(3))


def test_estimate_homography_ransac_failure_mocked() -> None:
    """Verify fallback to Identity matrix when RANSAC fails (mocked)."""
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


def test_estimate_interframe_homography_ransac_low_inliers_mocked() -> None:
    """Verify fallback when findHomography returns too few inliers (< min_matches)."""
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
            mock_bf.match.return_value = [
                MagicMock(queryIdx=i, trainIdx=i, distance=float(i)) for i in range(15)
            ]
            mock_bf_create.return_value = mock_bf

            # Inlier mask has only 3 inliers (< min_matches=10)
            inlier_mask = np.zeros((15, 1), dtype=np.uint8)
            inlier_mask[:3] = 1
            matrix = np.eye(3, dtype=np.float64)

            with patch("cv2.findHomography", return_value=(matrix, inlier_mask)):
                H, ok = estimate_interframe_homography(
                    prev_gray, curr_gray, return_status=True
                )
                assert ok is False
                assert np.allclose(H, np.eye(3))


def test_estimate_interframe_homography_ransac_non_finite_matrix_mocked() -> None:
    """Verify fallback when findHomography produces a non-finite homography matrix."""
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
            mock_bf.match.return_value = [
                MagicMock(queryIdx=i, trainIdx=i, distance=float(i)) for i in range(15)
            ]
            mock_bf_create.return_value = mock_bf

            non_finite_matrix = np.full((3, 3), np.nan)
            inlier_mask = np.ones((15, 1), dtype=np.uint8)

            with patch(
                "cv2.findHomography", return_value=(non_finite_matrix, inlier_mask)
            ):
                H, ok = estimate_interframe_homography(
                    prev_gray, curr_gray, return_status=True
                )
                assert ok is False
                assert np.allclose(H, np.eye(3))
