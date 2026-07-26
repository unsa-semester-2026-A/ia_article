"""Small end-to-end tests for the semantic copy-paste command path."""

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from src.augmentation.pipeline import AugmentationConfig
from src.augmentation.run import _production_source_limits, package_delta, render


def test_production_source_limits_skip_supported_classes() -> None:
    """Production allocates SAM only to classes that policy can augment."""
    limits = _production_source_limits(
        {0: 900, 1: 10, 2: 90},
        AugmentationConfig(tau=0.02, reuse_cap=2),
    )
    assert limits[0] == 0
    assert limits[2] == 0
    assert limits[1] == 10  # five needed source tracks plus a 2x reserve


def test_render_and_package_delta_without_cloud_sync(tmp_path: Path) -> None:
    """One semantic crop reaches a compact, self-contained training delta."""
    background = tmp_path / "lama.jpg"
    assert cv2.imwrite(str(background), np.full((360, 640, 3), 40, np.uint8))
    crop = np.zeros((12, 18, 4), dtype=np.uint8)
    crop[2:10, 3:15, :3] = (80, 140, 200)
    crop[2:10, 3:15, 3] = 255
    crop_path = tmp_path / "crop.png"
    assert cv2.imwrite(str(crop_path), crop)
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(
        json.dumps(
            {
                "synthetic_id": "synth_000000",
                "lama_background": str(background),
                "base_label_lines": [],
                "seed": 42,
                "objects": [
                    {
                        "class_id": 1,
                        "crop_path": str(crop_path),
                        "target_points": [
                            [100, 100],
                            [130, 100],
                            [130, 115],
                            [100, 115],
                        ],
                        "source_track_id": "track_1",
                        "slot_id": "slot_1",
                    }
                ],
            }
        )
        + "\n"
    )
    output = tmp_path / "delta"

    assert (
        render(
            argparse.Namespace(
                jobs_jsonl=str(jobs), output_dir=str(output), no_drive_sync=True
            )
        )
        == 0
    )
    assert (output / "images" / "synth_000000.jpg").is_file()
    assert (output / "labels" / "synth_000000.txt").is_file()

    exports = tmp_path / "exports"
    assert (
        package_delta(
            argparse.Namespace(
                synthetic_images=str(output / "images"),
                synthetic_labels=str(output / "labels"),
                manifest=str(output / "manifest.csv"),
                output_dir=str(exports),
                run_id="smoke",
                no_drive_sync=True,
            )
        )
        == 0
    )
    archive = exports / "sam_copy_paste_delta_smoke.zip"
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == {
            "images/train/synth_000000.jpg",
            "labels/train/synth_000000.txt",
            "manifest.csv",
        }
