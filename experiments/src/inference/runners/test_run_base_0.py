import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from experiments.src.inference.runners.run_base_0 import Base0Runner


@pytest.fixture
@patch("experiments.src.inference.runners.run_base_0.YOLO")
@patch("experiments.src.inference.runners.run_base_0.IOManager")
def runner(mock_io, mock_yolo):
    """Instancia del orquestador con Mocks para YOLO e IOManager."""
    mock_model = MagicMock()
    mock_model.names = {
        0: "plane",
        9: "small-vehicle",
        10: "large-vehicle",
        14: "swimming-pool",
    }
    mock_yolo.return_value = mock_model

    config = {
        "metadata_path": "dummy_meta.csv",
        "images_dir": "dummy_img",
        "output_dir": "dummy_out",
        "device": "cpu",
        "batch_size": 2,
    }
    return Base0Runner(config=config, model_path="dummy.pt")


# ==========================================
# Pruebas de Mapeo Directo (Caja Negra)
# ==========================================
def test_dota_to_mtc_mapping(runner):
    """Caja Negra: El dict de mapeo directo contiene solo vehículos con IDs MTC correctos."""
    assert runner.dota_to_mtc == {9: 0, 10: 6}


def test_non_vehicle_classes_excluded(runner):
    """Caja Negra: Clases no vehiculares (plane, swimming-pool) no aparecen en el mapeo."""
    assert 0 not in runner.dota_to_mtc    # plane
    assert 14 not in runner.dota_to_mtc   # swimming-pool


# ==========================================
# Pruebas _process_batch_results (Caja Blanca)
# ==========================================
def test_process_batch_accumulates_speed(runner):
    """Caja Blanca: Verifica que los acumuladores se actualicen por cada resultado del batch."""
    mock_r1 = MagicMock()
    mock_r1.speed = {"preprocess": 1.0, "inference": 10.0, "postprocess": 2.0}
    mock_r1.orig_shape = (1080, 1920)
    mock_r1.obb = None

    mock_r2 = MagicMock()
    mock_r2.speed = {"preprocess": 1.5, "inference": 12.0, "postprocess": 2.5}
    mock_r2.orig_shape = (1080, 1920)
    mock_r2.obb = None

    frames = runner._process_batch_results([mock_r1, mock_r2], frame_offset=0)

    assert len(frames) == 2
    assert runner.total_frames_processed == 2
    assert runner._time_sums["inference"] == 22.0


def test_process_batch_extracts_obb_corners(runner):
    """Caja Blanca: Verifica extracción de vértices OBB y filtrado por clase vehicular."""
    mock_r = MagicMock()
    mock_r.speed = {"preprocess": 1.0, "inference": 10.0, "postprocess": 2.0}
    mock_r.orig_shape = (1080, 1920)

    # Simular 2 detecciones: vehicular (class 9) y no vehicular (class 0)
    mock_obb = MagicMock()
    mock_obb.__len__ = lambda self: 2
    corners = np.array([
        [[10, 20], [50, 20], [50, 80], [10, 80]],
        [[100, 200], [150, 200], [150, 280], [100, 280]],
    ], dtype=np.float32)
    mock_obb.xyxyxyxy.cpu().numpy.return_value = corners
    mock_obb.cls.cpu().numpy.return_value = np.array([9, 0])
    mock_obb.conf.cpu().numpy.return_value = np.array([0.95, 0.80])
    mock_r.obb = mock_obb

    frames = runner._process_batch_results([mock_r], frame_offset=5)

    assert frames[0]["frame_idx"] == 5
    # Solo 1 detección vehicular (class 9 -> MTC 0)
    assert len(frames[0]["detections"]) == 1
    det = frames[0]["detections"][0]
    assert det["class_id"] == 0
    assert det["score"] == pytest.approx(0.95, abs=0.01)
    assert len(det["obb_corners"]) == 8


# ==========================================
# Pruebas execute (Caja Blanca)
# ==========================================
@patch("experiments.src.inference.runners.run_base_0.gc")
def test_execute_pipeline_batching(mock_gc, runner):
    """Caja Blanca: Verifica que el pipeline ejecute por lotes y limpie memoria."""
    runner.io_manager.load_csv.return_value = [
        {"clip_id": "clip1", "split": "val"},
        {"clip_id": "clip2", "split": "val"},
        {"clip_id": "clip3", "split": "train"},  # No val -> ignorar
    ]

    # 3 frames por clip, batch_size=2 -> 2 llamadas a predict por clip
    runner.io_manager.list_files_in_dir.return_value = [
        Path("f0.jpg"), Path("f1.jpg"), Path("f2.jpg"),
    ]

    mock_result = MagicMock()
    mock_result.speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}
    mock_result.orig_shape = (1080, 1920)
    mock_result.obb = None
    runner.model.predict.return_value = [mock_result]

    runner.start_hardware_monitoring = MagicMock()
    runner.record_hardware_metrics = MagicMock(return_value={"mock": True})

    res = runner.execute()

    # 2 clips val × 2 batches cada uno = 4 llamadas a predict
    assert runner.model.predict.call_count == 4
    # gc.collect por cada clip
    assert mock_gc.collect.call_count == 2
    assert res["status"] == "success"


@patch("experiments.src.inference.runners.run_base_0.gc")
def test_execute_skips_empty_clips(mock_gc, runner):
    """Caja Negra: Clips sin frames se omiten sin error."""
    runner.io_manager.load_csv.return_value = [
        {"clip_id": "empty_clip", "split": "val"},
    ]
    runner.io_manager.list_files_in_dir.return_value = []

    runner.start_hardware_monitoring = MagicMock()
    runner.record_hardware_metrics = MagicMock(return_value={})

    res = runner.execute()

    runner.model.predict.assert_not_called()
    assert res["status"] == "success"


@patch("experiments.src.inference.runners.run_base_0.gc")
def test_execute_resumes_existing_clips(mock_gc, runner, tmp_path):
    """Caja Blanca: Si un clip ya tiene su JSON, se omite sin reprocesar."""
    runner.config["output_dir"] = str(tmp_path)

    runner.io_manager.load_csv.return_value = [
        {"clip_id": "done_clip", "split": "val"},
        {"clip_id": "new_clip", "split": "val"},
    ]

    # Crear el JSON del primer clip para simular reanudación
    (tmp_path / "done_clip_predictions.json").write_text("{}")

    runner.io_manager.list_files_in_dir.return_value = [Path("f0.jpg")]

    mock_result = MagicMock()
    mock_result.speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}
    mock_result.orig_shape = (1080, 1920)
    mock_result.obb = None
    runner.model.predict.return_value = [mock_result]

    runner.start_hardware_monitoring = MagicMock()
    runner.record_hardware_metrics = MagicMock(return_value={})

    res = runner.execute()

    # Solo debe procesar new_clip (1 llamada a predict)
    assert runner.model.predict.call_count == 1
    assert res["status"] == "success"


@patch("experiments.src.inference.runners.run_base_0.gc")
def test_execute_saves_metrics_on_error(mock_gc, runner):
    """Caja Blanca: Ante una excepción, el bloque finally SIEMPRE guarda métricas parciales."""
    runner.io_manager.load_csv.return_value = [
        {"clip_id": "clip1", "split": "val"},
    ]
    runner.io_manager.list_files_in_dir.return_value = [Path("f0.jpg")]

    # Forzar un error durante la inferencia
    runner.model.predict.side_effect = RuntimeError("GPU OOM simulado")

    runner.start_hardware_monitoring = MagicMock()
    runner.record_hardware_metrics = MagicMock(return_value={"partial": True})

    res = runner.execute()

    # record_hardware_metrics DEBE haberse llamado a pesar del error
    runner.record_hardware_metrics.assert_called_once()
    # save_json debe haberse llamado al menos 1 vez (para las métricas parciales)
    assert runner.io_manager.save_json.call_count >= 1
