"""Tests for the temporal motion filter."""

import math

import numpy as np
import pytest
from src.evaluation.motion_filter import (
    MAX_ASSOCIATION_DISTANCE_PX,
    MAX_STATIC_DISPLACEMENT_PX,
    MIN_STATIC_TRACK_FRAMES,
    Detection,
    MotionFilterDiagnostics,
    Track,
    TrackObservation,
    _normalize_homography,
    _project_with_matrix,
    associate_detections,
    build_tracks,
    classify_static_tracks,
    compute_compensated_displacement,
    estimate_homography,
    filter_static_predictions,
    project_centroid,
)


def _detection(
    class_id: int,
    cx: float,
    cy: float = 100.0,
    score: float = 0.9,
) -> Detection:
    """Build a small valid detection for temporal tests."""
    return Detection(class_id, score, cx, cy, 40.0, 20.0, 0.0)


def _track(track_id: int, centers: list[float], class_id: int = 1) -> Track:
    """Build an identity-camera track from horizontal centers."""
    observations = tuple(
        TrackObservation(
            f"clip_test_{frame_index:04d}",
            0,
            _detection(class_id, center),
        )
        for frame_index, center in enumerate(centers)
    )
    return Track(track_id, class_id, observations)


def test_motion_filter_public_contract() -> None:
    """Expose the agreed thresholds, data structures, and six functions."""
    detection = Detection(1, 0.9, 200.0, 150.0, 120.0, 60.0, 15.0)
    observation = TrackObservation("clip_0001", 0, detection)
    track = Track(0, detection.class_id, (observation,))
    diagnostics = MotionFilterDiagnostics(1, (), (track.track_id,), 0)

    assert MAX_ASSOCIATION_DISTANCE_PX == 30.0
    assert MIN_STATIC_TRACK_FRAMES == 10
    assert MAX_STATIC_DISPLACEMENT_PX == 8.0
    assert track.observations[0].detection == detection
    assert diagnostics.removed_predictions == 0
    assert all(
        callable(function)
        for function in (
            project_centroid,
            associate_detections,
            build_tracks,
            compute_compensated_displacement,
            classify_static_tracks,
            filter_static_predictions,
        )
    )


def test_project_centroid_and_invalid_homography_fallback() -> None:
    """Project homogeneous coordinates and safely replace a missing matrix."""
    translation = np.asarray(((1.0, 0.0, 2.0), (0.0, 1.0, -3.0), (0.0, 0.0, 1.0)))

    assert project_centroid((100.0, 50.0), translation) == pytest.approx((102.0, 47.0))
    with pytest.warns(RuntimeWarning, match="missing homography"):
        assert project_centroid((5.0, 7.0), None) == (5.0, 7.0)
    with pytest.warns(RuntimeWarning, match="invalid homography"):
        assert project_centroid((5.0, 7.0), np.zeros((3, 3))) == (5.0, 7.0)
    with pytest.warns(RuntimeWarning, match="invalid homography"):
        assert project_centroid((5.0, 7.0), np.full((3, 3), math.nan)) == (5.0, 7.0)


def test_normalize_homography_warnings_and_fallback() -> None:
    """Verify fallback and warnings for non-convertible, wrong-shape, or singular homographies."""
    with pytest.warns(RuntimeWarning, match="invalid homography"):
        res_str = _normalize_homography("invalid")  # type: ignore[arg-type]
        assert np.allclose(res_str, np.eye(3))

    with pytest.warns(RuntimeWarning, match="invalid homography"):
        res_shape = _normalize_homography(np.eye(2))
        assert np.allclose(res_shape, np.eye(3))

    with pytest.warns(RuntimeWarning, match="invalid homography"):
        res_singular = _normalize_homography(np.zeros((3, 3)))
        assert np.allclose(res_singular, np.eye(3))


def test_project_with_matrix_non_finite_and_zero_denom() -> None:
    """Verify centroid projection handling of non-finite inputs and near-zero denominator matrix."""
    with pytest.raises(ValueError, match="centroid coordinates must be finite"):
        _project_with_matrix((math.nan, 10.0), np.eye(3))

    zero_denom_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.warns(RuntimeWarning, match="invalid homogeneous projection"):
        res = _project_with_matrix((10.0, 20.0), zero_denom_matrix)
        assert res == (10.0, 20.0)


def test_homography_estimator_falls_back_on_featureless_frames() -> None:
    """Return identity when ORB cannot find enough background features."""
    blank = np.zeros((64, 64), dtype=np.uint8)
    homography, success = estimate_homography(blank, blank)

    assert success is False
    assert np.allclose(homography, np.eye(3))


def test_estimate_homography_with_polygons_and_params() -> None:
    """Verify estimate_homography wrapper with polygons and custom feature bounds."""
    blank = np.zeros((64, 64), dtype=np.uint8)
    polys = [np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32)]
    H, ok = estimate_homography(
        blank,
        blank,
        previous_polygons=polys,
        nfeatures=1000,
        min_keypoints=10,
        min_matches=5,
    )
    assert ok is False
    assert np.allclose(H, np.eye(3))


def test_association_is_same_class_one_to_one_and_strict() -> None:
    """Use global greedy distance without reusing tracks or detections."""
    tracks = [_track(0, [0.0], 1), _track(1, [10.0], 1)]
    detections = [_detection(1, 9.0), _detection(1, 1.0), _detection(2, 0.0)]
    associations = associate_detections(tracks, detections, np.eye(3))

    assert {(item.track_id, item.detection_index) for item in associations} == {
        (0, 1),
        (1, 0),
    }
    boundary_track = [_track(2, [0.0], 1)]
    assert (
        len(associate_detections(boundary_track, [_detection(1, 29.9)], np.eye(3))) == 1
    )
    assert associate_detections(boundary_track, [_detection(1, 30.0)], np.eye(3)) == []


def test_build_tracks_sorts_frames_and_preserves_references() -> None:
    """Process shuffled frames sequentially and retain source detection indices."""
    predictions = {
        "clip_0001": [_detection(1, 2.0)],
        "clip_0000": [_detection(1, 0.0)],
    }
    tracks = build_tracks(predictions, {"clip_0001": np.eye(3)})

    assert len(tracks) == 1
    assert [item.frame_id for item in tracks[0].observations] == [
        "clip_0000",
        "clip_0001",
    ]
    assert [item.detection_index for item in tracks[0].observations] == [0, 0]


def test_compensated_dispersion_handles_camera_and_return_motion() -> None:
    """Remove camera translation and detect movement that returns to its origin."""
    camera_step = np.asarray(((1.0, 0.0, 2.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    static_track = _track(0, [100.0, 102.0, 104.0])
    moving_track = _track(1, [100.0, 107.0, 104.0])
    homographies = {
        "clip_test_0001": camera_step,
        "clip_test_0002": camera_step,
    }

    assert compute_compensated_displacement(
        static_track, homographies
    ) == pytest.approx(0.0)
    assert compute_compensated_displacement(
        moving_track, homographies
    ) == pytest.approx(5.0)


def test_static_classification_respects_duration_and_strict_displacement() -> None:
    """Eliminate only tracks with at least 10 frames and dispersion below 8 px."""
    static_ten = _track(0, [100.0] * 10)
    static_nine = _track(1, [100.0] * 9)
    exactly_eight = _track(2, [100.0] * 9 + [108.0])
    greater_than_eight = _track(3, [100.0] * 9 + [109.0])
    homographies = {
        f"clip_test_{frame_index:04d}": np.eye(3) for frame_index in range(1, 10)
    }

    static_ids = classify_static_tracks(
        [static_ten, static_nine, exactly_eight, greater_than_eight],
        homographies,
    )

    assert static_ids == {0}


def test_filter_reproduces_validated_synthetic_prototype() -> None:
    """Remove the long static track while retaining moving and short tracks."""
    frame_ids = [f"clip_demo_{index:04d}" for index in range(12)]
    camera_step = np.asarray(((1.0, 0.0, 2.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    homographies = {frame_id: camera_step for frame_id in frame_ids[1:]}
    predictions = {
        frame_id: [
            _detection(1, 100.0 + 2.0 * index),
            _detection(2, 250.0 + 5.0 * index),
            *([_detection(3, 380.0 + 2.0 * index)] if index < 6 else []),
        ]
        for index, frame_id in enumerate(frame_ids)
    }
    original_count = sum(map(len, predictions.values()))

    filtered, diagnostics = filter_static_predictions(predictions, homographies)

    assert original_count == 30
    assert sum(map(len, predictions.values())) == 30
    assert sum(map(len, filtered.values())) == 18
    assert diagnostics == MotionFilterDiagnostics(3, (0,), (1, 2), 12)
    assert all(
        detection.class_id != 1 for items in filtered.values() for detection in items
    )


def test_empty_frames_do_not_fail() -> None:
    """Return empty frames and diagnostics without index errors."""
    predictions = {"clip_0000": [], "clip_0001": []}
    filtered, diagnostics = filter_static_predictions(predictions, {})

    assert filtered == predictions
    assert diagnostics == MotionFilterDiagnostics(0, (), (), 0)


@pytest.mark.parametrize(
    "invalid_detection",
    [
        Detection(0, 0.9, 0.0, 0.0, 1.0, 1.0, 0.0),
        Detection(1, math.nan, 0.0, 0.0, 1.0, 1.0, 0.0),
        Detection(1, 0.9, 0.0, 0.0, 0.0, 1.0, 0.0),
        Detection(1, -0.1, 0.0, 0.0, 1.0, 1.0, 0.0),
        Detection(1, 1.5, 0.0, 0.0, 1.0, 1.0, 0.0),
        Detection(1, 0.9, 0.0, 0.0, -1.0, 1.0, 0.0),
    ],
)
def test_invalid_detections_are_rejected(invalid_detection: Detection) -> None:
    """Reject unsupported classes, non-finite scores, and invalid OBB sizes."""
    with pytest.raises(ValueError):
        build_tracks({"clip_0000": [invalid_detection]}, {})
    with pytest.raises(ValueError):
        associate_detections([], [invalid_detection], np.eye(3))


def test_invalid_tracks_are_rejected() -> None:
    """Reject invalid track IDs, class IDs, empty observations, negative indices, or out-of-order observations."""
    det = _detection(1, 10.0)
    obs1 = TrackObservation("clip_0001", 0, det)
    obs0 = TrackObservation("clip_0000", 0, det)

    # Negative track ID
    with pytest.raises(ValueError, match="track_id cannot be negative"):
        classify_static_tracks([Track(-1, 1, (obs1,))], {})

    # Invalid class ID (0 or 10)
    with pytest.raises(ValueError, match="track class_id"):
        classify_static_tracks([Track(0, 10, (obs1,))], {})

    # Empty observations
    with pytest.raises(ValueError, match="at least one observation"):
        classify_static_tracks([Track(0, 1, ())], {})

    # Empty frame ID
    with pytest.raises(ValueError, match="frame_id must be non-empty"):
        classify_static_tracks([Track(0, 1, (TrackObservation("", 0, det),))], {})

    # Negative detection index
    with pytest.raises(ValueError, match="detection_index cannot be negative"):
        classify_static_tracks(
            [Track(0, 1, (TrackObservation("clip_0001", -1, det),))], {}
        )

    # Class ID mismatch between detection and track
    det_class2 = _detection(2, 10.0)
    with pytest.raises(ValueError, match="share the track class_id"):
        classify_static_tracks(
            [Track(0, 1, (TrackObservation("clip_0001", 0, det_class2),))], {}
        )

    # Strict temporal order violation
    with pytest.raises(ValueError, match="strict temporal order"):
        classify_static_tracks([Track(0, 1, (obs1, obs0))], {})


def test_duplicate_track_ids_are_rejected() -> None:
    """Reject duplicate active track IDs in association and classification."""
    t0 = _track(0, [10.0])
    with pytest.raises(ValueError, match="unique"):
        associate_detections([t0, t0], [_detection(1, 10.0)], np.eye(3))

    with pytest.raises(ValueError, match="unique"):
        classify_static_tracks([t0, t0], {})


@pytest.mark.parametrize("invalid_frame_id", ["", 123])
def test_build_tracks_invalid_frame_id_rejected(invalid_frame_id: object) -> None:
    """Reject non-string or empty frame ID keys in prediction dictionary."""
    with pytest.raises(ValueError, match="non-empty strings"):
        build_tracks({invalid_frame_id: [_detection(1, 10.0)]}, {})  # type: ignore[dict-item]


def test_build_tracks_non_numeric_frame_ids() -> None:
    """Cover natural sorting fallback when frame IDs lack trailing numeric digits."""
    predictions = {
        "frame_b": [_detection(1, 2.0)],
        "frame_a": [_detection(1, 0.0)],
    }
    tracks = build_tracks(predictions, {})
    assert len(tracks) == 1
    assert [obs.frame_id for obs in tracks[0].observations] == ["frame_a", "frame_b"]


@pytest.mark.parametrize(
    ("max_distance", "min_frames", "max_displacement"),
    [(0.0, 10, 8.0), (30.0, 0, 8.0), (30.0, 10, -1.0)],
)
def test_invalid_filter_thresholds_are_rejected(
    max_distance: float,
    min_frames: int,
    max_displacement: float,
) -> None:
    """Reject thresholds that contradict the public contract."""
    with pytest.raises(ValueError):
        filter_static_predictions(
            {},
            {},
            max_distance_px=max_distance,
            min_frames=min_frames,
            max_displacement_px=max_displacement,
        )
