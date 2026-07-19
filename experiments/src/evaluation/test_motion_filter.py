"""Contract tests for the temporal motion filter."""

from src.evaluation.motion_filter import (
    MAX_ASSOCIATION_DISTANCE_PX,
    MAX_STATIC_DISPLACEMENT_PX,
    MIN_STATIC_TRACK_FRAMES,
    Detection,
    MotionFilterDiagnostics,
    Track,
    TrackObservation,
    associate_detections,
    build_tracks,
    classify_static_tracks,
    compute_compensated_displacement,
    filter_static_predictions,
    project_centroid,
)


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
