"""Geometry-preserving IC-Light rendering helpers for one synthetic job."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from src.augmentation.iclight import ICLightClient, ICLightRequest


def letterbox(
    image: np.ndarray, size: int = 512
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Letterbox an image and return the exact reversible crop geometry."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4
    )
    canvas = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    left, top = (size - resized_width) // 2, (size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas, (left, top, resized_width, resized_height)


def undo_letterbox(
    image: np.ndarray,
    geometry: tuple[int, int, int, int],
    output_size: tuple[int, int] = (640, 360),
) -> np.ndarray:
    """Remove recorded padding and restore native frame dimensions."""
    left, top, width, height = geometry
    cropped = image[top : top + height, left : left + width]
    return cv2.resize(cropped, output_size, interpolation=cv2.INTER_LANCZOS4)


def warp_crop_to_slot(
    crop_path: Path,
    target_points: list[list[float]],
    frame_size: tuple[int, int] = (640, 360),
) -> np.ndarray:
    """Warp an alpha crop to a target OBB and return a full-frame BGRA layer."""
    crop = cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED)
    if crop is None or crop.ndim != 3 or crop.shape[2] != 4:
        raise ValueError(f"Crop must be a readable RGBA PNG: {crop_path}")
    contour_points = cv2.findNonZero(crop[:, :, 3])
    if contour_points is None:
        raise ValueError(f"Crop has no foreground alpha: {crop_path}")
    source = cv2.boxPoints(cv2.minAreaRect(contour_points)).astype(np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    if target.shape != (4, 2):
        raise ValueError("target_points must contain four [x, y] corners")
    transform = cv2.getPerspectiveTransform(
        _ordered_quad(source), _ordered_quad(target)
    )
    width, height = frame_size
    return cv2.warpPerspective(
        crop,
        transform,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _ordered_quad(points: np.ndarray) -> np.ndarray:
    """Order corners clockwise from upper-left for a stable homography."""
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    ordered = points[np.argsort(angles)]
    return np.roll(ordered, -np.argmin(ordered.sum(axis=1)), axis=0).astype(np.float32)


def relight_variant(
    client: ICLightClient,
    foreground_bgra: np.ndarray,
    background_path: Path,
    output_path: Path,
    seed: int,
    *,
    working_size: int | tuple[int, int] = 512,
    steps: int = 20,
) -> Path:
    """Relight one composited foreground against one Raw or LaMa background.

    ``working_size`` is either a square edge or an explicit ``(width, height)``
    pair. The latter permits an input close to the camera aspect ratio without
    introducing black letterbox bands into the diffusion conditioning. IC-Light
    requires both dimensions to be divisible by 64.
    """
    working_width, working_height = (
        (working_size, working_size)
        if isinstance(working_size, int)
        else working_size
    )
    if (
        working_width < 64
        or working_height < 64
        or working_width % 64
        or working_height % 64
    ):
        raise ValueError("working dimensions must be positive multiples of 64")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"Background not found: {background_path}")
    if foreground_bgra.ndim != 3 or foreground_bgra.shape[2] != 4:
        raise ValueError("foreground_bgra must be a four-channel BGRA image")
    # ``process_relight`` calls IC-Light's upstream RMBG model itself. Supply
    # an ordinary RGB image with a neutral white backing, rather than a black
    # transparent canvas: black pixels are otherwise interpreted as image
    # content and can dominate the generated scene when the vehicle is small.
    alpha = foreground_bgra[:, :, 3:4].astype(np.float32) / 255.0
    vehicle = foreground_bgra[:, :, :3].astype(np.float32)
    foreground_bgr = (vehicle * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    # The official callback centre-crops its inputs to the requested size.
    # Supplying both inputs at that size avoids accidental black padding. The
    # 576x320 quality-smoke size differs from 16:9 by only 1.25%.
    background_working = cv2.resize(
        background,
        (working_width, working_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    foreground_working = cv2.resize(
        foreground_bgr,
        (working_width, working_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    temp_dir = output_path.parent / ".iclight_inputs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    foreground_path = temp_dir / f"{output_path.stem}_fg.png"
    background_input_path = temp_dir / f"{output_path.stem}_bg.png"
    cv2.imwrite(str(foreground_path), foreground_working)
    cv2.imwrite(str(background_input_path), background_working)
    generated = client.relight(
        ICLightRequest(
            foreground_path,
            background_input_path,
            seed,
            steps=steps,
            image_size=working_width,
            image_height=working_height,
        )
    )
    result = cv2.imread(str(generated), cv2.IMREAD_COLOR)
    if result is None:
        raise RuntimeError(f"IC-Light returned unreadable output: {generated}")
    restored = cv2.resize(result, (640, 360), interpolation=cv2.INTER_LANCZOS4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), restored, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"Could not write IC-Light output: {output_path}")
    return output_path
