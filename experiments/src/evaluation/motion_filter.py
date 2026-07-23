"""Temporal filtering of static vehicle predictions with camera compensation."""

import math
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from src.utils.homography import estimate_interframe_homography

MAX_ASSOCIATION_DISTANCE_PX = 30.0
MIN_STATIC_TRACK_FRAMES = 10
MAX_STATIC_DISPLACEMENT_PX = 8.0

Centroid: TypeAlias = tuple[float, float]
Homography: TypeAlias = NDArray[np.float64]
Image: TypeAlias = NDArray[np.uint8]
Polygon: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Detection:
    """One raw OBB prediction from a single frame.

    Attributes:
        class_id: Official vehicle class identifier from 1 through 9.
        score: Model confidence in the closed range [0, 1].
        cx: Horizontal OBB center in pixels.
        cy: Vertical OBB center in pixels.
        width: OBB width in pixels.
        height: OBB height in pixels.
        angle_deg: OBB rotation angle in degrees.
    """

    class_id: int
    score: float
    cx: float
    cy: float
    width: float
    height: float
    angle_deg: float


PredictionsByFrame: TypeAlias = dict[str, list[Detection]]

# Each key identifies the current frame. Its matrix maps coordinates from the
# immediately previous frame into the current frame. The first frame may be
# omitted because it has no preceding transition.
HomographiesByFrame: TypeAlias = dict[str, Homography | None]


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """Reference from a temporal track to an original frame detection."""

    frame_id: str
    detection_index: int
    detection: Detection


@dataclass(frozen=True, slots=True)
class Track:
    """Ordered observations associated with one physical vehicle."""

    track_id: int
    class_id: int
    observations: tuple[TrackObservation, ...]


@dataclass(frozen=True, slots=True)
class Association:
    """One accepted track-to-detection assignment in the current frame."""

    track_id: int
    detection_index: int
    distance_px: float


@dataclass(frozen=True, slots=True)
class MotionFilterDiagnostics:
    """Summary needed to audit which predictions the filter removed."""

    total_tracks: int
    static_track_ids: tuple[int, ...]
    retained_track_ids: tuple[int, ...]
    removed_predictions: int


def _identity_homography() -> Homography:
    """Return a fresh 3x3 identity homography."""
    return np.eye(3, dtype=np.float64)


def _normalize_homography(homography: Homography | None) -> Homography:
    """Validate a homography or replace it with identity and warn once."""
    if homography is None:
        warnings.warn(
            "missing homography; using identity transformation",
            RuntimeWarning,
            stacklevel=3,
        )
        return _identity_homography()

    try:
        matrix = np.asarray(homography, dtype=np.float64)
    except (TypeError, ValueError):
        warnings.warn(
            "invalid homography; using identity transformation",
            RuntimeWarning,
            stacklevel=3,
        )
        return _identity_homography()
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or abs(float(np.linalg.det(matrix))) < 1e-12
    ):
        warnings.warn(
            "invalid homography; using identity transformation",
            RuntimeWarning,
            stacklevel=3,
        )
        return _identity_homography()
    return matrix


def _project_with_matrix(centroid: Centroid, matrix: Homography) -> Centroid:
    """Project a finite centroid using an already validated matrix."""
    cx, cy = (float(value) for value in centroid)
    if not math.isfinite(cx) or not math.isfinite(cy):
        raise ValueError("centroid coordinates must be finite")

    projected = matrix @ np.asarray((cx, cy, 1.0), dtype=np.float64)
    denominator = float(projected[2])
    if not np.all(np.isfinite(projected)) or abs(denominator) < 1e-12:
        warnings.warn(
            "invalid homogeneous projection; preserving original centroid",
            RuntimeWarning,
            stacklevel=3,
        )
        return (cx, cy)
    return (float(projected[0] / denominator), float(projected[1] / denominator))


def _validate_detection(detection: Detection) -> None:
    """Reject malformed predictions before temporal association."""
    if isinstance(detection.class_id, bool) or detection.class_id not in range(1, 10):
        raise ValueError("class_id must be an official class from 1 through 9")
    values = (
        detection.score,
        detection.cx,
        detection.cy,
        detection.width,
        detection.height,
        detection.angle_deg,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("all detection values must be finite")
    if not 0.0 <= detection.score <= 1.0:
        raise ValueError("score must be within [0, 1]")
    if detection.width <= 0.0 or detection.height <= 0.0:
        raise ValueError("detection width and height must be greater than zero")


def _validate_tracking_parameters(
    max_distance_px: float,
    min_frames: int,
    max_displacement_px: float,
) -> None:
    """Validate public tracking and classification thresholds."""
    if not math.isfinite(max_distance_px) or max_distance_px <= 0.0:
        raise ValueError("max_distance_px must be finite and greater than zero")
    if (
        isinstance(min_frames, bool)
        or not isinstance(min_frames, int)
        or min_frames <= 0
    ):
        raise ValueError("min_frames must be a positive integer")
    if not math.isfinite(max_displacement_px) or max_displacement_px < 0.0:
        raise ValueError("max_displacement_px must be finite and non-negative")


def _frame_sort_key(frame_id: str) -> tuple[str, int, str]:
    """Sort standard frame IDs naturally by their final numeric component."""
    match = re.match(r"^(.*?)(\d+)$", frame_id)
    if match is None:
        return (frame_id, -1, frame_id)
    return (match.group(1), int(match.group(2)), frame_id)


def _validate_track(track: Track) -> None:
    """Validate track identity, ordering, and referenced detections."""
    if track.track_id < 0:
        raise ValueError("track_id cannot be negative")
    if track.class_id not in range(1, 10):
        raise ValueError("track class_id must be from 1 through 9")
    if not track.observations:
        raise ValueError("a track must contain at least one observation")
    previous_key: tuple[str, int, str] | None = None
    for observation in track.observations:
        if not observation.frame_id:
            raise ValueError("observation frame_id must be non-empty")
        if observation.detection_index < 0:
            raise ValueError("detection_index cannot be negative")
        _validate_detection(observation.detection)
        if observation.detection.class_id != track.class_id:
            raise ValueError("all track detections must share the track class_id")
        current_key = _frame_sort_key(observation.frame_id)
        if previous_key is not None and current_key <= previous_key:
            raise ValueError("track observations must be in strict temporal order")
        previous_key = current_key


def project_centroid(
    centroid: Centroid,
    homography: Homography | None,
) -> Centroid:
    """Project a centroid from the previous frame into the current frame.

    Invalid or missing homographies use identity and emit ``RuntimeWarning``.

    Args:
        centroid: Previous-frame center in pixel coordinates.
        homography: Matrix mapping the previous frame to the current frame.

    Returns:
        Projected finite centroid in the current frame.

    Raises:
        ValueError: If the input centroid is non-finite.
    """
    return _project_with_matrix(centroid, _normalize_homography(homography))


def estimate_homography(
    previous_gray: Image,
    current_gray: Image,
    previous_polygons: Sequence[Polygon] | None = None,
    nfeatures: int = 2_500,
    min_keypoints: int = 15,
    min_matches: int = 10,
) -> tuple[Homography, bool]:
    """Estimate previous-to-current camera motion using ORB and RANSAC.

    Delegates homography calculation to the central ``src.utils.homography`` module.

    Args:
        previous_gray: Previous grayscale frame.
        current_gray: Current grayscale frame.
        previous_polygons: Foreground OBB polygons to exclude.
        nfeatures: Maximum ORB feature count.
        min_keypoints: Minimum keypoints required in each frame.
        min_matches: Minimum matches and RANSAC inliers required.

    Returns:
        Estimated homography and a success flag. Failure returns identity.

    Raises:
        ValueError: If images or numeric parameters are invalid.
    """
    matrix, success = estimate_interframe_homography(
        prev_gray=previous_gray,
        curr_gray=current_gray,
        previous_polygons=previous_polygons,
        nfeatures=nfeatures,
        min_keypoints=min_keypoints,
        min_matches=min_matches,
        return_status=True,
    )
    return matrix, success


def associate_detections(
    active_tracks: Sequence[Track],
    detections: Sequence[Detection],
    homography: Homography | None,
    max_distance_px: float = MAX_ASSOCIATION_DISTANCE_PX,
) -> list[Association]:
    """Associate detections greedily with projected tracks of the same class.

    Candidate pairs are sorted by distance. Each track and detection may occur
    in at most one accepted association, and the distance comparison is strict.

    Args:
        active_tracks: Tracks observed in the immediately previous frame.
        detections: Current-frame raw predictions.
        homography: Matrix mapping the previous frame to the current frame.
        max_distance_px: Strict upper bound for an accepted association.

    Returns:
        Deterministically ordered one-to-one associations.
    """
    _validate_tracking_parameters(max_distance_px, 1, 0.0)
    matrix = _normalize_homography(homography)
    candidates: list[tuple[float, int, int]] = []
    seen_track_ids: set[int] = set()

    for detection in detections:
        _validate_detection(detection)

    for track in active_tracks:
        _validate_track(track)
        if track.track_id in seen_track_ids:
            raise ValueError("active track IDs must be unique")
        seen_track_ids.add(track.track_id)
        last_detection = track.observations[-1].detection
        projected = _project_with_matrix((last_detection.cx, last_detection.cy), matrix)
        for detection_index, detection in enumerate(detections):
            if detection.class_id != track.class_id:
                continue
            distance = math.hypot(
                detection.cx - projected[0],
                detection.cy - projected[1],
            )
            candidates.append((distance, track.track_id, detection_index))

    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    associations: list[Association] = []
    for distance, track_id, detection_index in sorted(candidates):
        if distance >= max_distance_px:
            continue
        if track_id in used_tracks or detection_index in used_detections:
            continue
        associations.append(Association(track_id, detection_index, distance))
        used_tracks.add(track_id)
        used_detections.add(detection_index)
    return associations


def build_tracks(
    predictions_by_frame: PredictionsByFrame,
    homographies_by_frame: HomographiesByFrame,
    max_distance_px: float = MAX_ASSOCIATION_DISTANCE_PX,
) -> list[Track]:
    """Build temporal tracks while processing frames sequentially.

    Frame IDs are naturally sorted by their final numeric component. A track is
    active only when it was observed in the immediately previous frame.

    Args:
        predictions_by_frame: Raw predictions from one clip grouped by frame.
        homographies_by_frame: Previous-to-current matrices keyed by current frame.
        max_distance_px: Strict association distance threshold.

    Returns:
        All tracks with references to their original frame detections.
    """
    _validate_tracking_parameters(max_distance_px, 1, 0.0)
    for frame_id, detections in predictions_by_frame.items():
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("frame IDs must be non-empty strings")
        for detection in detections:
            _validate_detection(detection)

    observations_by_track: dict[int, list[TrackObservation]] = {}
    class_by_track: dict[int, int] = {}
    active_track_ids: set[int] = set()
    next_track_id = 0

    for frame_position, frame_id in enumerate(
        sorted(predictions_by_frame, key=_frame_sort_key)
    ):
        detections = predictions_by_frame[frame_id]
        active_tracks = [
            Track(
                track_id,
                class_by_track[track_id],
                tuple(observations_by_track[track_id]),
            )
            for track_id in sorted(active_track_ids)
        ]
        associations = (
            []
            if frame_position == 0 or not active_tracks or not detections
            else associate_detections(
                active_tracks,
                detections,
                homographies_by_frame.get(frame_id),
                max_distance_px,
            )
        )
        matched_detection_indices: set[int] = set()
        current_track_ids: set[int] = set()

        for association in associations:
            observation = TrackObservation(
                frame_id,
                association.detection_index,
                detections[association.detection_index],
            )
            observations_by_track[association.track_id].append(observation)
            matched_detection_indices.add(association.detection_index)
            current_track_ids.add(association.track_id)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indices:
                continue
            observations_by_track[next_track_id] = [
                TrackObservation(frame_id, detection_index, detection)
            ]
            class_by_track[next_track_id] = detection.class_id
            current_track_ids.add(next_track_id)
            next_track_id += 1

        active_track_ids = current_track_ids

    return [
        Track(track_id, class_by_track[track_id], tuple(observations))
        for track_id, observations in sorted(observations_by_track.items())
    ]


def compute_compensated_displacement(
    track: Track,
    homographies_by_frame: HomographiesByFrame,
) -> float:
    """Compute maximum trajectory dispersion after camera-motion compensation.

    Residual motion is accumulated after projecting every previous centroid to
    the current frame. The returned diagonal range matches the Phase 0 tracker
    while preventing a moving-out-and-back trajectory from cancelling to zero.

    Args:
        track: Temporally ordered observations of one vehicle.
        homographies_by_frame: Previous-to-current matrices keyed by current frame.

    Returns:
        Maximum compensated trajectory dispersion in pixels.
    """
    _validate_track(track)
    compensated_positions: list[Centroid] = [(0.0, 0.0)]
    accumulated_x = 0.0
    accumulated_y = 0.0

    for previous, current in zip(track.observations, track.observations[1:]):
        matrix = _normalize_homography(homographies_by_frame.get(current.frame_id))
        previous_detection = previous.detection
        current_detection = current.detection
        projected = _project_with_matrix(
            (previous_detection.cx, previous_detection.cy), matrix
        )
        accumulated_x += current_detection.cx - projected[0]
        accumulated_y += current_detection.cy - projected[1]
        compensated_positions.append((accumulated_x, accumulated_y))

    x_positions = [position[0] for position in compensated_positions]
    y_positions = [position[1] for position in compensated_positions]
    width = max(x_positions) - min(x_positions)
    height = max(y_positions) - min(y_positions)
    return math.hypot(width, height)


def classify_static_tracks(
    tracks: Sequence[Track],
    homographies_by_frame: HomographiesByFrame,
    min_frames: int = MIN_STATIC_TRACK_FRAMES,
    max_displacement_px: float = MAX_STATIC_DISPLACEMENT_PX,
) -> set[int]:
    """Return IDs lasting at least 10 frames and moving less than 8 pixels.

    Args:
        tracks: Completed temporal tracks.
        homographies_by_frame: Previous-to-current matrices keyed by current frame.
        min_frames: Inclusive minimum duration for static classification.
        max_displacement_px: Strict upper displacement bound.

    Returns:
        IDs of tracks classified as static.
    """
    _validate_tracking_parameters(1.0, min_frames, max_displacement_px)
    static_track_ids: set[int] = set()
    seen_track_ids: set[int] = set()
    for track in tracks:
        _validate_track(track)
        if track.track_id in seen_track_ids:
            raise ValueError("track IDs must be unique")
        seen_track_ids.add(track.track_id)
        if len(track.observations) < min_frames:
            continue
        displacement = compute_compensated_displacement(track, homographies_by_frame)
        if displacement < max_displacement_px:
            static_track_ids.add(track.track_id)
    return static_track_ids


def filter_static_predictions(
    predictions_by_frame: PredictionsByFrame,
    homographies_by_frame: HomographiesByFrame,
    max_distance_px: float = MAX_ASSOCIATION_DISTANCE_PX,
    min_frames: int = MIN_STATIC_TRACK_FRAMES,
    max_displacement_px: float = MAX_STATIC_DISPLACEMENT_PX,
) -> tuple[PredictionsByFrame, MotionFilterDiagnostics]:
    """Remove static-track detections and return auditable diagnostics.

    Args:
        predictions_by_frame: Raw predictions from one clip grouped by frame.
        homographies_by_frame: Previous-to-current matrices keyed by current frame.
        max_distance_px: Strict association distance threshold.
        min_frames: Inclusive minimum duration for static classification.
        max_displacement_px: Strict upper displacement bound.

    Returns:
        A new prediction mapping and diagnostics identifying removed tracks.
    """
    _validate_tracking_parameters(max_distance_px, min_frames, max_displacement_px)
    tracks = build_tracks(
        predictions_by_frame,
        homographies_by_frame,
        max_distance_px,
    )
    static_track_ids = classify_static_tracks(
        tracks,
        homographies_by_frame,
        min_frames,
        max_displacement_px,
    )
    removed_references = {
        (observation.frame_id, observation.detection_index)
        for track in tracks
        if track.track_id in static_track_ids
        for observation in track.observations
    }
    filtered_predictions = {
        frame_id: [
            detection
            for detection_index, detection in enumerate(detections)
            if (frame_id, detection_index) not in removed_references
        ]
        for frame_id, detections in predictions_by_frame.items()
    }
    all_track_ids = {track.track_id for track in tracks}
    diagnostics = MotionFilterDiagnostics(
        total_tracks=len(tracks),
        static_track_ids=tuple(sorted(static_track_ids)),
        retained_track_ids=tuple(sorted(all_track_ids - static_track_ids)),
        removed_predictions=len(removed_references),
    )
    return filtered_predictions, diagnostics
