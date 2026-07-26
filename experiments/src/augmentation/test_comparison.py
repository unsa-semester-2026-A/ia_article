"""Tests for the bounded ten-example Raw/LaMa comparison preparation."""

from pathlib import Path

import cv2
import numpy as np
from src.augmentation.comparison import prepare_three_frame_comparison


def _write_image(path: Path) -> None:
    """Write a small valid frame used as a test fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((360, 640, 3), dtype=np.uint8))


def _label(class_id: int, x: float) -> str:
    """Return one normalized 20x20 OBB line at a safe horizontal position."""
    return f"{class_id} {x} 0.4 {x + 0.03} 0.4 {x + 0.03} 0.46 {x} 0.46\n"


def test_prepare_three_frame_comparison_covers_five_classes(tmp_path: Path) -> None:
    """Ten paired frames produce ten LaMa jobs and persist review inputs."""
    labels, raw, lama = tmp_path / "labels", tmp_path / "raw", tmp_path / "lama"
    labels.mkdir()
    for index, class_id in enumerate((1, 2, 4, 5, 7)):
        stem = f"source_{index}"
        _write_image(raw / f"{stem}.jpg")
        labels.joinpath(f"{stem}.txt").write_text(_label(class_id, 0.1))
    for index in range(10):
        stem = f"target_{index}"
        _write_image(raw / f"{stem}.jpg")
        _write_image(lama / f"{stem}.jpg")
        labels.joinpath(f"{stem}.txt").write_text(_label(0, 0.1))

    jobs, manifest = prepare_three_frame_comparison(
        labels_dir=labels,
        raw_images_dir=raw,
        lama_images_dir=lama,
        output_dir=tmp_path / "output",
    )

    assert len(jobs) == 10
    assert manifest["class_coverage"] == {
        "1": "combi",
        "2": "microbus",
        "4": "omnibus",
        "5": "articulado",
        "7": "mototaxi",
    }
    assert (tmp_path / "output" / "comparison_manifest.json").is_file()
    assert (
        len(list((tmp_path / "output" / "comparisons").rglob("raw_original.jpg"))) == 10
    )
    assert (
        len(list((tmp_path / "output" / "comparisons").rglob("lama_background.jpg")))
        == 10
    )
    assert all("output_path" in job for job in jobs)
