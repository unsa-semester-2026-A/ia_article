"""Small end-to-end tests for the semantic copy-paste command path."""

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import src.augmentation.run as augmentation_run
from src.augmentation.metrics import ProductionMonitor
from src.augmentation.pipeline import AugmentationConfig
from src.augmentation.run import (
    _production_source_limits,
    _stage_dataset_release,
    _validate_release,
    package_delta,
    render,
)


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
    assert (
        json.loads((output / "render_progress.json").read_text())["status"]
        == "complete"
    )

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


def test_staged_release_is_train_only_and_validated(tmp_path: Path) -> None:
    """Production staging has standard train paths and no validation mutation."""
    rendered = tmp_path / "rendered"
    for directory in (rendered / "images", rendered / "labels"):
        directory.mkdir(parents=True)
    assert cv2.imwrite(
        str(rendered / "images" / "synth_000000.jpg"),
        np.zeros((360, 640, 3), dtype=np.uint8),
    )
    (rendered / "labels" / "synth_000000.txt").write_text(
        "1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
    )
    (rendered / "manifest.csv").write_text(
        'synthetic_id,objects\nsynth_000000,"[{""class_id"": 1}]"\n'
    )
    output_root = tmp_path / "output"
    jobs = output_root / "work" / "jobs.jsonl"
    jobs.parent.mkdir(parents=True)
    jobs.write_text('{"background_clip_id": "v_train"}\n')
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"clip_id": ["v_train", "v_val"], "split": ["train", "val"]}).to_csv(
        metadata, index=False
    )
    release = output_root / "release" / "sam_copy_paste_test"
    _stage_dataset_release(rendered, release)
    result = _validate_release(release, {"jobs": 1}, metadata, jobs)
    assert result["synthetic_images"] == 1
    assert result["validation_unchanged"] is True


def test_production_monitor_writes_resumable_hardware_report(tmp_path: Path) -> None:
    """Metrics always include process, disk, GPU and completed-stage evidence."""
    output = tmp_path / "metrics.json"
    monitor = ProductionMonitor(output, tmp_path, interval_seconds=0.01)
    monitor.start()
    monitor.stage("prepare_complete", jobs=1)
    monitor.stop()
    report = json.loads(output.read_text())
    assert report["status"] == "success"
    assert report["process"]["peak_rss_bytes"] >= 0
    assert "gpu_usage" in report
    assert report["stages"][0]["name"] == "prepare_complete"


def test_production_creates_train_delta_and_audit_zip(
    tmp_path: Path, monkeypatch
) -> None:
    """The orchestrator produces a train-only release without real GPU work."""
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"clip_id": ["v_train"], "split": ["train"]}).to_csv(
        metadata, index=False
    )

    def fake_prepare(args) -> int:
        work = Path(args.workdir)
        work.mkdir(parents=True)
        (work / "jobs.jsonl").write_text('{"background_clip_id": "v_train"}\n')
        (work / "augmentation_state.json").write_text(
            json.dumps(
                {
                    "status": "prepared",
                    "jobs": 1,
                    "crop_tracks": 1,
                    "policy": {},
                    "real_counts": {"1": 1},
                    "track_counts": {"1": 1},
                    "quotas": {"1": 1},
                }
            )
        )
        return 0

    def fake_render(args) -> int:
        destination = Path(args.output_dir)
        (destination / "images").mkdir(parents=True)
        (destination / "labels").mkdir(parents=True)
        assert cv2.imwrite(
            str(destination / "images" / "synth_000000.jpg"),
            np.zeros((360, 640, 3), dtype=np.uint8),
        )
        (destination / "labels" / "synth_000000.txt").write_text(
            "1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
        )
        (destination / "manifest.csv").write_text(
            'synthetic_id,objects\nsynth_000000,"[{""class_id"": 1}]"\n'
        )
        (destination / "render_progress.json").write_text(
            '{"completed_jobs": 1, "total_jobs": 1, "status": "complete"}'
        )
        return 0

    monkeypatch.setattr(augmentation_run, "prepare", fake_prepare)
    monkeypatch.setattr(augmentation_run, "render", fake_render)
    args = argparse.Namespace(
        output_dir=str(tmp_path / "output"),
        run_id="test",
        resume=False,
        split_metadata=str(metadata),
        no_drive_sync=True,
        token_path="",
        drive_results_folder_id="",
        drive_checkpoints_folder_id="",
    )
    assert augmentation_run.production(args) == 0
    output = tmp_path / "output" / "sam_copy_paste_test"
    release = output / "release" / "sam_copy_paste_test"
    assert (release / "images" / "train" / "synth_000000.jpg").is_file()
    assert (release / "labels" / "train" / "synth_000000.txt").is_file()
    assert not (release / "images" / "val").exists()
    assert (output / "sam_copy_paste_delta_test.zip").is_file()
    assert (output / "sam_copy_paste_audit_test.zip").is_file()
