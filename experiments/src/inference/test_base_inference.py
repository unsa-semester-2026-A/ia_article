from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from src.inference.base_inference import BaseInferencePipeline


class DummyInferencePipeline(BaseInferencePipeline):
    """Clase dummy concreta para instanciar la interfaz abstracta."""

    def execute(self) -> dict[str, Any]:
        return {}


@pytest.fixture
def pipeline():
    config = {
        "device": 0,
        "experiment_condition": "TestCondition",
        "hardware_name": "TestHW",
    }
    return DummyInferencePipeline(config=config, model_path="dummy.pt")


# ==========================================
# Pruebas accumulate_speed (Caja Blanca)
# ==========================================
def test_accumulate_speed_sums_correctly(pipeline):
    """Caja Blanca: Verifica que los acumuladores sumen sin almacenar listas."""
    pipeline.accumulate_speed(
        {"preprocess": 1.0, "inference": 10.0, "postprocess": 2.0}
    )
    pipeline.accumulate_speed(
        {"preprocess": 3.0, "inference": 20.0, "postprocess": 4.0}
    )

    assert pipeline._time_sums["preprocess"] == 4.0
    assert pipeline._time_sums["inference"] == 30.0
    assert pipeline._time_sums["postprocess"] == 6.0
    assert pipeline.total_frames_processed == 2


def test_accumulate_speed_handles_missing_keys(pipeline):
    """Caja Negra: Claves faltantes en speed no deben romper la acumulación."""
    pipeline.accumulate_speed({"preprocess": 5.0})  # faltan inference y postprocess

    assert pipeline._time_sums["preprocess"] == 5.0
    assert pipeline._time_sums["inference"] == 0.0
    assert pipeline.total_frames_processed == 1


# ==========================================
# Pruebas start_hardware_monitoring (Caja Blanca)
# ==========================================
@patch("src.inference.base_inference.time.time")
@patch("src.inference.base_inference.torch.cuda")
def test_start_monitoring_with_cuda(mock_cuda, mock_time, pipeline):
    """Caja Blanca: Verifica llamadas si existe CUDA."""
    mock_cuda.is_available.return_value = True
    mock_time.return_value = 1000.0

    pipeline.start_hardware_monitoring()

    mock_cuda.reset_peak_memory_stats.assert_called_once()
    assert pipeline.start_time == 1000.0


@patch("src.inference.base_inference.torch.cuda")
def test_start_monitoring_without_cuda(mock_cuda, pipeline):
    """Caja Blanca: Flujo alternativo si NO existe CUDA (ej: CI sin GPU)."""
    mock_cuda.is_available.return_value = False
    pipeline.start_hardware_monitoring()
    mock_cuda.reset_peak_memory_stats.assert_not_called()


# ==========================================
# Pruebas record_hardware_metrics (Caja Negra / Blanca)
# ==========================================
@patch("src.inference.base_inference.resource")
@patch("src.inference.base_inference.torch.cuda")
@patch("src.inference.base_inference.time.time")
def test_record_hardware_metrics_calculus(
    mock_time, mock_cuda, mock_resource, pipeline
):
    """Caja Blanca: Inyecta valores estáticos y verifica cálculos matemáticos."""
    mock_time.return_value = 1010.0
    pipeline.start_time = 1000.0

    # Simular acumulación de 2 frames
    pipeline.accumulate_speed(
        {"preprocess": 1.0, "inference": 10.0, "postprocess": 2.0}
    )
    pipeline.accumulate_speed(
        {"preprocess": 1.0, "inference": 20.0, "postprocess": 2.0}
    )

    # 1 GB RAM
    mock_ru = MagicMock()
    mock_ru.ru_maxrss = 1048576
    mock_resource.getrusage.return_value = mock_ru

    # 2 GB VRAM
    mock_cuda.is_available.return_value = True
    mock_cuda.max_memory_allocated.return_value = 2147483648

    metrics = pipeline.record_hardware_metrics()

    assert metrics["experiment_condition"] == "TestCondition"
    assert metrics["hardware"] == "TestHW"
    assert metrics["total_time_seconds"] == 10.0
    assert metrics["peak_cpu_ram_gb"] == 1.0
    assert metrics["peak_vram_mb"] == 2048.0
    assert metrics["average_speed_ms"]["preprocess"] == 1.0
    assert metrics["average_speed_ms"]["inference"] == 15.0
    assert metrics["average_speed_ms"]["postprocess"] == 2.0
    # total avg = 18.0 ms -> FPS = 1000 / 18 ≈ 55.56
    assert metrics["theoretical_fps"] == pytest.approx(55.56, abs=0.01)


@patch("src.inference.base_inference.resource")
@patch("src.inference.base_inference.torch.cuda")
@patch("src.inference.base_inference.time.time")
def test_record_metrics_zero_frames(mock_time, mock_cuda, mock_resource, pipeline):
    """Caja Negra: Sin frames procesados no debe haber división por cero."""
    mock_time.return_value = 1001.0
    pipeline.start_time = 1000.0
    mock_ru = MagicMock()
    mock_ru.ru_maxrss = 0
    mock_resource.getrusage.return_value = mock_ru
    mock_cuda.is_available.return_value = False

    metrics = pipeline.record_hardware_metrics()

    assert metrics["total_frames_processed"] == 0
    assert metrics["theoretical_fps"] == 0.0
