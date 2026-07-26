"""Tests for staging an augmentation release into a Kaggle dataset snapshot."""

import json
from pathlib import Path

from scripts.publish_sam_copy_paste_release import stage_release


def test_stage_release_preserves_train_only_contract(tmp_path: Path) -> None:
    """The release becomes a new top-level directory without validation files."""
    release = tmp_path / "sam_copy_paste_test"
    (release / "images" / "train").mkdir(parents=True)
    (release / "labels" / "train").mkdir(parents=True)
    (release / "images" / "train" / "synth.jpg").write_bytes(b"image")
    (release / "labels" / "train" / "synth.txt").write_text("0 0 0 0 0 0 0 0 0\n")
    (release / "release_manifest.json").write_text(
        json.dumps({"validation": {"validation_unchanged": True}})
    )
    dataset = tmp_path / "mtc"
    for path in (
        dataset / "train_resized" / "train",
        dataset / "smart_lama_corrected" / "train",
        dataset / "yolo_obb_labels" / "train",
    ):
        path.mkdir(parents=True)
    (dataset / "split_metadata.csv").write_text("clip_id,split\n")

    destination = stage_release(release, dataset)

    assert (destination / "images" / "train" / "synth.jpg").is_file()
    assert not (destination / "images" / "val").exists()
