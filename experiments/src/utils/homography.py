"""Homography estimation and dynamic masking utilities for vehicle motion tracking."""

import cv2
import numpy as np


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
    # Start with a solid white mask (all pixels active)
    mask = np.ones(img_shape[:2], dtype=np.uint8) * 255

    if obb_predictions is None or len(obb_predictions) == 0:
        return mask

    for box in obb_predictions:
        cx, cy, w, h, angle_rad = box

        # Calculate corners from center, width, height, and angle in radians
        cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)

        # Local offsets for the 4 corners relative to the center
        dx = np.array([-w / 2, w / 2, w / 2, -w / 2])
        dy = np.array([-h / 2, -h / 2, h / 2, h / 2])

        # Rotate and translate
        x_rot = dx * cos_t - dy * sin_t + cx
        y_rot = dx * sin_t + dy * cos_t + cy

        # Build polygon array in integer pixel coordinates
        corners = np.stack((x_rot, y_rot), axis=1).astype(np.int32)

        # Paint the vehicle polygon black (0) on the mask
        cv2.fillPoly(mask, [corners], 0)

    return mask


def estimate_interframe_homography(
    prev_gray: np.ndarray, curr_gray: np.ndarray, prev_mask: np.ndarray | None = None
) -> np.ndarray:
    """Estimates the 3x3 homography matrix between prev and curr grayscale frames.

    Prevents keypoint contamination inside vehicle regions by using an optional mask.
    If matching or RANSAC estimation fails, returns a 3x3 Identity matrix.

    Args:
        prev_gray: Grayscale image at frame t.
        curr_gray: Grayscale image at frame t+1.
        prev_mask: Binary mask for frame t (0 for vehicles, 255 for background).

    Returns:
        np.ndarray: 3x3 homography matrix of float64 type.
    """
    # Initialize ORB detector with a limit of 1500 features
    orb = cv2.ORB_create(nfeatures=1500)

    # Extract features applying the exclusion mask on the previous frame
    kp1, des1 = orb.detectAndCompute(prev_gray, mask=prev_mask)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    # Fallback if keypoints are insufficient
    if des1 is None or des2 is None or len(kp1) < 15 or len(kp2) < 15:
        return np.eye(3, dtype=np.float64)

    # Match descriptors using BF Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Fallback if matching keypoints count is too low
    if len(matches) < 10:
        return np.eye(3, dtype=np.float64)

    # Extract coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate homography matrix with RANSAC
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Fallback if RANSAC fails or has too few inliers
    if H is None or np.sum(status) < 10:
        return np.eye(3, dtype=np.float64)

    return H.astype(np.float64)
