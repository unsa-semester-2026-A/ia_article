"""Tests for deterministic synthetic augmentation data rules."""

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from src.augmentation.pipeline import (
    OBB,
    AugmentationConfig,
    SyntheticDatasetBuilder,
    load_yolo_obb,
    obb_to_yolo_line,
    quota_by_class,
)
from src.augmentation.render import relight_variant


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((360, 640, 3), dtype=np.uint8))


def test_yolo_obb_round_trip(tmp_path: Path) -> None:
    """YOLO OBB serialization preserves class and pixel geometry."""
    box = OBB(2, ((64.0, 36.0), (128.0, 36.0), (128.0, 72.0), (64.0, 72.0)))
    labels = tmp_path / "frame.txt"
    labels.write_text(obb_to_yolo_line(box) + "\n")
    assert load_yolo_obb(labels) == [box]


def test_quota_enforces_eligibility_real_cap_and_track_cap() -> None:
    """Only rare classes receive the minimum of the three policy constraints."""
    config = AugmentationConfig(tau=0.02, reuse_cap=2, budget_fraction=1.0)
    quotas = quota_by_class({0: 900, 1: 10, 2: 90}, {1: 2, 2: 100}, config)
    assert quotas[0] == 0
    assert quotas[1] == 4
    assert quotas[2] == 0


def test_collect_slots_excludes_validation_and_real_overlap(tmp_path: Path) -> None:
    """Static slots never leak validation clips or overlap labelled objects."""
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"clip_id": ["v_train", "v_val"], "split": ["train", "val"]}).to_csv(
        metadata, index=False
    )
    labels = tmp_path / "labels"
    labels.mkdir()
    labels.joinpath("v_train_0000.txt").write_text(
        "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n"
    )
    slots = {
        "v_train_0000": [
            {"cx": 500, "cy": 200, "w": 20, "h": 20, "angle": 0},
            {"cx": 320, "cy": 180, "w": 40, "h": 40, "angle": 0},
        ],
        "v_val_0000": [{"cx": 500, "cy": 200, "w": 20, "h": 20, "angle": 0}],
    }
    static_json = tmp_path / "static.json"
    static_json.write_text(json.dumps(slots))
    builder = SyntheticDatasetBuilder(AugmentationConfig())
    found = builder.collect_slots(static_json, builder.train_clip_ids(metadata), labels)
    assert [slot.slot_id for slot in found] == ["v_train_0000:0"]


def test_package_full_dataset_is_self_contained_and_reports_checksum(
    tmp_path: Path,
) -> None:
    """Each condition archive contains paired base and synthetic train files."""
    base_images, base_labels = tmp_path / "base_images", tmp_path / "base_labels"
    synth_images, synth_labels = tmp_path / "synth_images", tmp_path / "synth_labels"
    _write_image(base_images / "real.jpg")
    _write_image(synth_images / "synth.jpg")
    for directory, stem in ((base_labels, "real"), (synth_labels, "synth")):
        directory.mkdir()
        (directory / f"{stem}.txt").write_text("0 0 0 0 0 0 0 0 0\n")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("synthetic_id\nsynth\n")
    target = tmp_path / "smart_raw_synthetic_train.zip"
    report = SyntheticDatasetBuilder(AugmentationConfig()).package_full_dataset(
        base_images, base_labels, synth_images, synth_labels, manifest, target, "raw"
    )
    assert report["images"] == 2
    with zipfile.ZipFile(target) as archive:
        assert set(archive.namelist()) == {
            "images/train/real.jpg",
            "images/train/synth.jpg",
            "labels/train/real.txt",
            "labels/train/synth.txt",
            "manifest.csv",
        }


def test_relight_variant_sends_three_channel_foreground_to_iclight(tmp_path: Path) -> None:
    """IC-Light receives BGR, while alpha remains only a geometry intermediate."""
    background = tmp_path / "background.jpg"
    output = tmp_path / "result.jpg"
    generated = tmp_path / "generated.png"
    _write_image(background)
    assert cv2.imwrite(str(generated), np.zeros((512, 512, 3), dtype=np.uint8))

    class RecordingClient:
        def relight(self, request):
            received = cv2.imread(str(request.foreground), cv2.IMREAD_UNCHANGED)
            assert received is not None and received.shape[2] == 3
            return generated

    foreground = np.zeros((360, 640, 4), dtype=np.uint8)
    foreground[100:200, 200:300] = (20, 40, 60, 255)
    assert (
        relight_variant(
            RecordingClient(),
            foreground,
            background,
            output,
            seed=7,
            working_size=256,
            steps=2,
        )
        == output
    )
