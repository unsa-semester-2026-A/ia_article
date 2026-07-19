"""Public contract for temporal filtering of static vehicle predictions."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

MAX_ASSOCIATION_DISTANCE_PX = 30.0
MIN_STATIC_TRACK_FRAMES = 10
MAX_STATIC_DISPLACEMENT_PX = 8.0

Centroid: TypeAlias = tuple[float, float]
Homography: TypeAlias = NDArray[np.float64]


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


def project_centroid(
    centroid: Centroid,
    homography: Homography | None,
) -> Centroid:
    """Project a centroid from the previous frame into the current frame.

    Invalid or missing homographies will use the identity transformation and
    emit a warning in the definitive implementation.

    Raises:
        NotImplementedError: Until the homography experiment is migrated.
    """
    raise NotImplementedError


def associate_detections(
    active_tracks: Sequence[Track],
    detections: Sequence[Detection],
    homography: Homography | None,
    max_distance_px: float = MAX_ASSOCIATION_DISTANCE_PX,
) -> list[Association]:
    """Associate detections greedily with projected tracks of the same class.

    Raises:
        NotImplementedError: Until the matching experiment is migrated.
    """
    raise NotImplementedError


def build_tracks(
    predictions_by_frame: PredictionsByFrame,
    homographies_by_frame: HomographiesByFrame,
    max_distance_px: float = MAX_ASSOCIATION_DISTANCE_PX,
) -> list[Track]:
    """Build temporal tracks while processing frames sequentially.

    Raises:
        NotImplementedError: Until the tracking experiment is migrated.
    """
    raise NotImplementedError


def compute_compensated_displacement(
    track: Track,
    homographies_by_frame: HomographiesByFrame,
) -> float:
    """Compute a track's net displacement after camera-motion compensation.

    Raises:
        NotImplementedError: Until the displacement experiment is migrated.
    """
    raise NotImplementedError


def classify_static_tracks(
    tracks: Sequence[Track],
    homographies_by_frame: HomographiesByFrame,
    min_frames: int = MIN_STATIC_TRACK_FRAMES,
    max_displacement_px: float = MAX_STATIC_DISPLACEMENT_PX,
) -> set[int]:
    """Return IDs lasting at least 10 frames and moving less than 8 pixels.

    Raises:
        NotImplementedError: Until the classification experiment is migrated.
    """
    raise NotImplementedError


def filter_static_predictions(
    predictions_by_frame: PredictionsByFrame,
    homographies_by_frame: HomographiesByFrame,
    max_distance_px: float = MAX_ASSOCIATION_DISTANCE_PX,
    min_frames: int = MIN_STATIC_TRACK_FRAMES,
    max_displacement_px: float = MAX_STATIC_DISPLACEMENT_PX,
) -> tuple[PredictionsByFrame, MotionFilterDiagnostics]:
    """Remove static-track detections and return auditable diagnostics.

    Raises:
        NotImplementedError: Until all prototype stages are migrated.
    """
    raise NotImplementedError
