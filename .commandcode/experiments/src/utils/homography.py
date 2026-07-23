"""Homography estimation and dynamic masking utilities for vehicle motion tracking."""

from collections.abc import Sequence

import cv2
import numpy as np

Polygon = Sequence[tuple[float, float]] | np.ndarray
Homography = np.ndarray


def create_exclusion_mask_from_predictions(
    img_shape: tuple, obb_predictions: np.ndarray | None
) -> np.ndarray:
    """Generates a binary mask where predicted vehicle boxes are 0 (black) and background is 255 (white).

    Args:
        img_shape: Tuple representing (height, width) of the image.
        obb_predictions: Numpy array of shape (N, 5) representing [cx, cy, w, h, angle_rad] from model.

    Returns:
        np.ndarray: Binary mask of shape img_shape[:2] with uint8 dtype.
    """
    mask = np.ones(img_shape[:2], dtype=np.uint8) * 255

    if obb_predictions is None or len(obb_predictions) == 0:
        return mask

    for box in obb_predictions:
        cx, cy, w, h, angle_rad = box

        cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)

        dx = np.array([-w / 2, w / 2, w / 2, -w / 2])
        dy = np.array([-h / 2, -h / 2, h / 2, h / 2])

        x_rot = dx * cos_t - dy * sin_t + cx
        y_rot = dx * sin_t + dy * cos_t + cy

        corners = np.stack((x_rot, y_rot), axis=1).astype(np.int32)

        cv2.fillPoly(mask, [corners], 0)

    return mask


def create_exclusion_mask_from_polygons(
    img_shape: tuple[int, ...], polygons: Sequence[Polygon] | None = None
) -> np.ndarray:
    """Generates a binary background mask where vehicle polygons are 0 (black) and background is 255 (white).

    Args:
        img_shape: Shape tuple (height, width) of the image frame.
        polygons: Optional sequence of 4-vertex foreground polygons to exclude.

    Returns:
        np.ndarray: Binary mask of shape img_shape[:2] with uint8 dtype.

    Raises:
        ValueError: If any polygon does not have exactly 4 finite vertices.
    """
    mask = np.full(img_shape[:2], 255, dtype=np.uint8)
    if polygons is not None:
        for polygon in polygons:
            vertices = np.asarray(polygon, dtype=np.float32)
            if vertices.shape != (4, 2) or not np.all(np.isfinite(vertices)):
                raise ValueError(
                    "each foreground polygon must have four finite vertices"
                )
            cv2.fillPoly(mask, [vertices.astype(np.int32)], 0)
    return mask


def estimate_interframe_homography(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_mask: np.ndarray | None = None,
    previous_polygons: Sequence[Polygon] | None = None,
    nfeatures: int = 2_500,
    min_keypoints: int = 15,
    min_matches: int = 10,
    return_status: bool = False,
) -> Homography | tuple[Homography, bool]:
    """Estimates the 3x3 homography matrix between previous and current grayscale frames.

    Prevents keypoint contamination inside vehicle regions by using an exclusion mask
    or explicit foreground polygons.

    Args:
        prev_gray: Grayscale image array at frame t.
        curr_gray: Grayscale image array at frame t+1.
        prev_mask: Binary mask for frame t (0 for vehicles, 255 for background).
        previous_polygons: Sequence of OBB polygons (4, 2) to mask on frame t.
        nfeatures: Maximum ORB feature count.
        min_keypoints: Minimum keypoints required in each frame.
        min_matches: Minimum descriptor matches and RANSAC inliers required.
        return_status: If True, returns (Homography, success_flag) tuple.

    Returns:
        3x3 Homography matrix (or tuple of matrix and boolean success flag).

    Raises:
        ValueError: If image dimensions/values or numeric parameters are invalid.
    """
    previous = np.asarray(prev_gray)
    current = np.asarray(curr_gray)
    if previous.ndim != 2 or current.ndim != 2 or previous.shape != current.shape:
        raise ValueError("homography images must be equally sized grayscale arrays")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        raise ValueError("homography images must contain finite values")
    if nfeatures <= 0 or min_keypoints <= 0 or min_matches < 4:
        raise ValueError("invalid ORB or matching parameters")

    previous_u8 = np.clip(previous, 0, 255).astype(np.uint8, copy=False)
    current_u8 = np.clip(current, 0, 255).astype(np.uint8, copy=False)

    if prev_mask is not None:
        background_mask = prev_mask
    elif previous_polygons is not None:
        background_mask = create_exclusion_mask_from_polygons(
            previous_u8.shape, previous_polygons
        )
    else:
        background_mask = np.full(previous_u8.shape, 255, dtype=np.uint8)

    orb = cv2.ORB_create(nfeatures=nfeatures)
    previous_keypoints, previous_descriptors = orb.detectAndCompute(
        previous_u8, background_mask
    )
    current_keypoints, current_descriptors = orb.detectAndCompute(current_u8, None)

    identity = np.eye(3, dtype=np.float64)

    if (
        previous_descriptors is None
        or current_descriptors is None
        or len(previous_keypoints) < min_keypoints
        or len(current_keypoints) < min_keypoints
    ):
        return (identity, False) if return_status else identity

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(
        matcher.match(previous_descriptors, current_descriptors),
        key=lambda match: match.distance,
    )
    if len(matches) < min_matches:
        return (identity, False) if return_status else identity

    source_points = np.float32(
        [previous_keypoints[match.queryIdx].pt for match in matches]
    ).reshape(-1, 1, 2)
    destination_points = np.float32(
        [current_keypoints[match.trainIdx].pt for match in matches]
    ).reshape(-1, 1, 2)

    matrix, inlier_mask = cv2.findHomography(
        source_points,
        destination_points,
        cv2.RANSAC,
        5.0,
    )
    if matrix is None or inlier_mask is None or int(np.sum(inlier_mask)) < min_matches:
        return (identity, False) if return_status else identity

    normalized = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(normalized)):
        return (identity, False) if return_status else identity

    return (normalized, True) if return_status else normalized

