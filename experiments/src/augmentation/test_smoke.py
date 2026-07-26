"""Tests for bounded IC-Light smoke reporting."""

from pathlib import Path

import cv2
import numpy as np
from src.augmentation.smoke import run_smoke_batch


def test_smoke_batch_writes_metrics_and_validates_outputs(tmp_path: Path) -> None:
    """A successful small batch records every output and resource summary."""
    background = tmp_path / "background.jpg"
    assert cv2.imwrite(str(background), np.zeros((360, 640, 3), dtype=np.uint8))

    def fake_render(_client, _foreground, _background, output, _seed, **_kwargs):
        output.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(output), np.zeros((360, 640, 3), dtype=np.uint8))
        return output

    report = run_smoke_batch(
        object(),
        [
            {"id": "real_000", "foreground_bgra": np.zeros((360, 640, 4), np.uint8), "background_path": background, "seed": 42},
            {"id": "real_001", "foreground_bgra": np.zeros((360, 640, 4), np.uint8), "background_path": background, "seed": 43},
        ],
        tmp_path / "output",
        render=fake_render,
    )

    assert report["status"] == "passed"
    assert report["batch_size"] == 2
    assert all(row["shape"] == [360, 640, 3] for row in report["images"])
    assert (tmp_path / "output" / "iclight_smoke_metrics.json").is_file()
