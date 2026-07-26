"""Integration tests for packaging and configurable final evaluation."""

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from src.evaluation.final_runner import (
    ConditionSpec,
    EvaluationConfig,
    default_conditions,
    run_final_evaluation,
    validation_frame_ids,
)


class _Tensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class _Obb:
    def __init__(self) -> None:
        self.xywhr = _Tensor(np.asarray(((20.0, 20.0, 10.0, 8.0, 0.0),)))
        self.conf = _Tensor(np.asarray((0.9,)))
        self.cls = _Tensor(np.asarray((0,), dtype=np.int64))


class _Result:
    obb = _Obb()
    speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5}


class _Model:
    def predict(self, source: list[np.ndarray], **_: object) -> list[_Result]:
        return [_Result() for _ in source]


def test_default_conditions_include_known_f1_weights() -> None:
    conditions = {item.name: item for item in default_conditions()}
    assert conditions["Base 1"].checkpoint_name == "f1_c1_best.pt"
    assert conditions["Mejora A"].checkpoint_name == "f1_c3_best.pt"
    assert conditions["Base 0"].enabled is False


def test_final_runner_packages_complete_local_condition(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels" / "val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index in range(2):
        frame_id = f"clip_demo_{index:04d}"
        assert cv2.imwrite(
            str(images / f"{frame_id}.jpg"), np.zeros((64, 64, 3), dtype=np.uint8)
        )
        (labels / f"{frame_id}.txt").write_text("0 0 0 0\n", encoding="utf-8")
    ground_truth = tmp_path / "train.csv"
    ground_truth.write_text(
        "Id,Target\nclip_demo_0000,1 20 20 10 8 0\nclip_demo_0001,1 20 20 10 8 0\n",
        encoding="utf-8",
    )
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"deterministic-test-weight")
    config = EvaluationConfig(
        images,
        labels,
        ground_truth,
        tmp_path / "output",
        (ConditionSpec("Base 1", "unused", local_weights=str(weights)),),
        batch_size=2,
    )

    result = run_final_evaluation(config, model_loader=lambda _: _Model())

    assert result["skipped"] == {}
    archive = Path(result["completed"]["Base 1"]["archive"])
    assert archive.is_file()
    with zipfile.ZipFile(archive) as packaged:
        assert set(packaged.namelist()) == {
            "hardware.json",
            "homographies.json.gz",
            "manifest.json",
            "metrics_by_class.csv",
            "metrics_by_threshold.csv",
            "predictions_filtered.json.gz",
            "predictions_raw.json.gz",
            "summary.json",
        }
        summary = json.loads(packaged.read("summary.json"))
    assert summary["metric_name"] == "Macro AP-rIoU"
    assert summary["macro_ap_riou"] > 0.0


def test_validation_manifest_rejects_empty_directory(tmp_path: Path) -> None:
    labels = tmp_path / "val"
    labels.mkdir()
    try:
        validation_frame_ids(labels)
    except ValueError as exc:
        assert "no validation labels" in str(exc)
    else:
        raise AssertionError("an empty validation manifest must fail")
