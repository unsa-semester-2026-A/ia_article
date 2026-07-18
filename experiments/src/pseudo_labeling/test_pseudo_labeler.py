import os

import numpy as np
import pandas as pd
import pytest
from src.pseudo_labeling.pseudo_labeler import (
    CentroidTracker,
    PseudoLabeler,
    check_gt_matching,
    filter_static_vehicles,
    get_vehicle_class_ids,
    project_centroid,
)


class DummyYOLO:
    """Mock YOLO class to verify DOTA dynamic naming resolution."""

    def __init__(self, names: dict):
        self.names = names


def test_get_vehicle_class_ids() -> None:
    """Verifies that DOTA dynamic matching isolates vehicles under spaces/hyphens."""
    mock_names = {
        0: "plane",
        1: "ship",
        2: "storage-tank",
        3: "small-vehicle",
        4: "large vehicle",
        5: "tennis-court",
        6: "harbor",
        7: "truck",
    }
    model = DummyYOLO(mock_names)
    vehicle_ids = get_vehicle_class_ids(model)

    # "small-vehicle", "large vehicle", and "truck" should be classified as vehicles
    assert 3 in vehicle_ids
    assert 4 in vehicle_ids
    assert 7 in vehicle_ids
    # "plane", "ship", "storage-tank", "tennis-court", and "harbor" should not
    assert 0 not in vehicle_ids
    assert 1 not in vehicle_ids
    assert 2 not in vehicle_ids
    assert 5 not in vehicle_ids


def test_project_centroid_identity() -> None:
    """Verifies coordinates are preserved when using an identity matrix."""
    H = np.eye(3)
    cx, cy = 500.0, 300.0
    pcx, pcy = project_centroid(cx, cy, H)
    assert pcx == cx
    assert pcy == cy


def test_project_centroid_translation() -> None:
    """Verifies coordinates shift accurately when applying a translation homography."""
    # Translate X by +50, Y by -20
    H = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, -20.0], [0.0, 0.0, 1.0]])
    cx, cy = 100.0, 100.0
    pcx, pcy = project_centroid(cx, cy, H)
    assert pcx == 150.0
    assert pcy == 80.0


def test_tracker_association() -> None:
    """Checks tracker associates close centroids across sequential updates."""
    tracker = CentroidTracker(max_distance=30.0)
    H = np.eye(3)

    # Frame 0 detections
    dets_f0 = [(100.0, 100.0, 10.0, 5.0, 0.0, 0.90)]
    tracker.update(0, dets_f0, H)
    assert len(tracker.objects) == 1
    assert 0 in tracker.objects

    # Frame 1 detections: close centroid (displacement of 10 pixels, < max_distance)
    dets_f1 = [(108.0, 95.0, 10.0, 5.0, 0.0, 0.88)]
    tracker.update(1, dets_f1, H)
    assert len(tracker.objects) == 1
    assert 0 in tracker.objects  # Track ID 0 preserved

    # Frame 2 detections: distant centroid (displacement of 50 pixels, > max_distance)
    dets_f2 = [(200.0, 200.0, 10.0, 5.0, 0.0, 0.85)]
    tracker.update(2, dets_f2, H)
    assert len(tracker.objects) == 1
    assert (
        1 in tracker.objects
    )  # Track ID 0 dropped (not in new frame), new ID 1 created


def test_check_gt_matching() -> None:
    """Verifies distance threshold matching against ground truth annotations."""
    gt_data = pd.DataFrame(
        [
            {"frame_idx": 0, "cx": 100.0, "cy": 100.0},
            {"frame_idx": 0, "cx": 300.0, "cy": 300.0},
            {"frame_idx": 1, "cx": 150.0, "cy": 150.0},
        ]
    )

    # Centroid close to GT (distance < 30.0)
    assert check_gt_matching(105.0, 98.0, gt_data, 0, dist_threshold=30.0) is True

    # Centroid far from GT (distance > 30.0)
    assert check_gt_matching(200.0, 200.0, gt_data, 0, dist_threshold=30.0) is False

    # Centroid close but frame index does not contain annotations
    assert check_gt_matching(100.0, 100.0, gt_data, 5, dist_threshold=30.0) is False


def test_filter_static_vehicles() -> None:
    """Validates temporal consensus rules (length and spatial drift constraints)."""
    gt_df = pd.DataFrame(columns=["frame_idx", "cx", "cy"])

    # Trajectory 0: Static (11 frames, drift < 8 pixels, no GT match)
    traj_static = []
    for f in range(11):
        # Coordinates fluctuate slightly around 100.0, 100.0
        traj_static.append((f, 100.0 + (f % 2), 100.0 - (f % 2), 10.0, 5.0, 0.0, 0.90))

    # Trajectory 1: Dynamic (11 frames, but drift > 8 pixels)
    traj_moving = []
    for f in range(11):
        # Coordinates drift continuously from 200 to 220
        traj_moving.append((f, 200.0 + f * 2, 200.0, 10.0, 5.0, 0.0, 0.90))

    # Trajectory 2: Short (5 frames, drift < 8 pixels)
    traj_short = []
    for f in range(5):
        traj_short.append((f, 300.0, 300.0, 10.0, 5.0, 0.0, 0.90))

    history = {0: traj_static, 1: traj_moving, 2: traj_short}

    static_map = filter_static_vehicles(
        history, gt_df, min_frames=10, motion_threshold=8.0
    )

    # Frame 0 should contain a detection for trajectory 0, but NOT for 1 or 2
    assert "frame_0000" in static_map
    assert len(static_map["frame_0000"]) == 1
    assert static_map["frame_0000"][0]["cx"] == 100.0


def test_pseudo_labeler_initialization() -> None:
    """Verifies that the orchestrator initialization correctly maps custom variables."""
    labeler = PseudoLabeler(
        csv_path="test_train.csv",
        zip_path="test_train.zip",
        metadata_path="test_metadata.csv",
        output_dir="test_output",
        model_name="yolo_test.pt",
        conf_threshold=0.33,
        min_frames=15,
        motion_threshold=5.5,
        batch_size=8,
    )

    assert labeler.csv_path == "test_train.csv"
    assert labeler.zip_path == "test_train.zip"
    assert labeler.metadata_path == "test_metadata.csv"
    assert labeler.output_dir == "test_output"
    assert labeler.model_name == "yolo_test.pt"
    assert labeler.conf_threshold == 0.33
    assert labeler.min_frames == 15
    assert labeler.motion_threshold == 5.5
    assert labeler.batch_size == 8


def test_google_drive_api_connection() -> None:
    """Pre-flight check: Validates Google Drive connection, token auth, and folder accessibility in cloud environments."""
    token_path = "/content/drive/MyDrive/ia_article/token/token.json"
    folder_id = "1J5ogC3q6jyYlk3wuYyxpYZHslUg6eGtN"
    checkpoints_folder_id = "1anPtHNwHYgcq4BImhbJ_xzouiJ-Sh035"

    # Skip if we are not in Google Colab or if Drive is not mounted
    if not os.path.exists("/content/drive") or not os.path.exists(token_path):
        pytest.skip(
            "Not running in Google Colab or Google Drive is not mounted. Skipping cloud integration test."
        )

    # Attempt to initialize credentials and authenticate
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    try:
        creds = Credentials.from_authorized_user_file(
            token_path, ["https://www.googleapis.com/auth/drive.file"]
        )
        service = build("drive", "v3", credentials=creds)

        # Test 1: Query the parent folder info
        folder_info = service.files().get(fileId=folder_id, fields="id, name").execute()
        assert folder_info.get("id") == folder_id

        # Test 2: Query the checkpoints subfolder info
        checkpoints_info = (
            service.files()
            .get(fileId=checkpoints_folder_id, fields="id, name")
            .execute()
        )
        assert checkpoints_info.get("id") == checkpoints_folder_id

        print(
            f"✓ Cloud connection pre-flight check passed. Connected to {folder_info.get('name')} and {checkpoints_info.get('name')}."
        )
    except Exception as e:
        pytest.fail(f"Cloud connection pre-flight check failed! Error details: {e}")
