import csv
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from src.training.base_training import BaseTrainingPipeline


class ConcreteTrainer(BaseTrainingPipeline):
    """Concrete subclass of BaseTrainingPipeline for testing."""

    def get_hyperparameters(self) -> dict[str, Any]:
        """Return dummy hyperparameters."""
        return {"epochs": 10, "batch": 16}

    def get_dataset_config(self) -> dict[str, Any]:
        """Return dummy dataset config."""
        return {"data_yaml_path": "dummy.yaml"}

    def prepare_dataset(self) -> Path:
        """Return dummy path."""
        return Path("dummy.yaml")

    def execute(self) -> dict[str, Any]:
        """Return dummy execution results."""
        return {"status": "success"}


@pytest.fixture
def config() -> dict[str, Any]:
    """Provide a default configuration dictionary."""
    return {
        "output_dir": "/tmp/test_output",
        "experiment_condition": "Test_Condition",
        "hardware_name": "Test_Hardware",
    }


@pytest.fixture
def trainer(config: dict[str, Any]) -> ConcreteTrainer:
    """Provide a ConcreteTrainer instance."""
    return ConcreteTrainer(config)


# ===================================================================
# Black-Box Tests
# ===================================================================


@patch("src.training.base_training.torch.cuda")
def test_detect_device_no_cuda(mock_cuda: MagicMock, trainer: ConcreteTrainer) -> None:
    """Test detect_device when CUDA is not available."""
    mock_cuda.is_available.return_value = False
    assert trainer.detect_device() == "cpu"


@patch("src.training.base_training.torch.cuda")
def test_detect_device_1_gpu(mock_cuda: MagicMock, trainer: ConcreteTrainer) -> None:
    """Test detect_device when 1 GPU is available."""
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 1
    mock_cuda.get_device_name.return_value = "Tesla T4"
    assert trainer.detect_device() == "0"


@patch("src.training.base_training.torch.cuda")
def test_detect_device_2_gpus(mock_cuda: MagicMock, trainer: ConcreteTrainer) -> None:
    """Test detect_device when 2 GPUs are available."""
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 2
    mock_cuda.get_device_name.return_value = "Tesla T4"
    assert trainer.detect_device() == "0,1"


@patch("src.training.base_training.torch.cuda")
def test_detect_device_4_gpus(mock_cuda: MagicMock, trainer: ConcreteTrainer) -> None:
    """Test detect_device when 4 GPUs are available."""
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 4
    mock_cuda.get_device_name.return_value = "Tesla T4"
    assert trainer.detect_device() == "0,1,2,3"


def test_parse_results_csv_valid(trainer: ConcreteTrainer, tmp_path: Path) -> None:
    """Test parse_results_csv with a valid CSV with padded headers."""
    csv_path = tmp_path / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["  epoch ", " train/loss  ", "metrics/mAP50(B) "])
        writer.writerow(["1", "0.5", "0.9"])
        writer.writerow([" 2 ", " 0.4 ", " 0.95 "])

    data = trainer.parse_results_csv(csv_path)
    assert len(data) == 2
    assert data[0] == {"epoch": 1.0, "train/loss": 0.5, "metrics/mAP50(B)": 0.9}
    assert data[1] == {"epoch": 2.0, "train/loss": 0.4, "metrics/mAP50(B)": 0.95}


def test_parse_results_csv_empty(trainer: ConcreteTrainer, tmp_path: Path) -> None:
    """Test parse_results_csv with an empty CSV file."""
    csv_path = tmp_path / "empty.csv"
    csv_path.touch()
    data = trainer.parse_results_csv(csv_path)
    assert data == []


def test_parse_results_csv_missing(trainer: ConcreteTrainer) -> None:
    """Test parse_results_csv with a missing file."""
    with pytest.raises(FileNotFoundError):
        trainer.parse_results_csv(Path("/nonexistent/path/results.csv"))


def test_parse_results_csv_non_numeric(
    trainer: ConcreteTrainer, tmp_path: Path
) -> None:
    """Test parse_results_csv with non-numeric values."""
    csv_path = tmp_path / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "status"])
        writer.writerow(["1", "completed"])
        writer.writerow(["2", " "])

    data = trainer.parse_results_csv(csv_path)
    assert len(data) == 2
    assert data[0] == {"epoch": 1.0, "status": "completed"}
    assert data[1] == {"epoch": 2.0, "status": ""}


def test_build_training_metrics_without_batch_size(trainer: ConcreteTrainer) -> None:
    """Test build_training_metrics without effective_batch_size."""
    metrics = trainer.build_training_metrics(
        epochs_data=[{"epoch": 1.0}],
        hardware_metrics={"gpu_vram": {}},
        hyperparameters={"epochs": 10},
    )
    assert metrics["experiment_condition"] == "Test_Condition"
    assert metrics["total_epochs_completed"] == 1
    assert metrics["oom_fallback_triggered"] is False
    assert "effective_batch_size" not in metrics


def test_build_training_metrics_with_batch_size(trainer: ConcreteTrainer) -> None:
    """Test build_training_metrics with effective_batch_size."""
    metrics = trainer.build_training_metrics(
        epochs_data=[{"epoch": 1.0}],
        hardware_metrics={"gpu_vram": {}},
        hyperparameters={"epochs": 10},
        effective_batch_size=16,
    )
    assert metrics["oom_fallback_triggered"] is True
    assert metrics["effective_batch_size"] == 16


def test_build_training_metrics_empty_epochs(trainer: ConcreteTrainer) -> None:
    """Test build_training_metrics with empty epochs_data."""
    metrics = trainer.build_training_metrics([], {}, {})
    assert metrics["total_epochs_completed"] == 0
    assert metrics["per_epoch_metrics"] == []


@patch("src.training.base_training.torch.cuda")
def test_health_check_gpu_available(
    mock_cuda: MagicMock, trainer: ConcreteTrainer, tmp_path: Path
) -> None:
    """Test health_check when GPU is available."""
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 1
    mock_cuda.get_device_name.return_value = "Test GPU"
    mock_cuda.get_device_properties.return_value.total_memory = 16 * 1024**3

    trainer.config["output_dir"] = str(tmp_path / "out")

    health = trainer.health_check()
    assert health["passed"] is True
    assert health["details"]["gpu"]["available"] is True
    assert health["details"]["gpu"]["count"] == 1
    assert health["details"]["gpu"]["devices"][0]["vram_gb"] == 16.0


@patch("src.training.base_training.torch.cuda")
def test_health_check_no_gpu(
    mock_cuda: MagicMock, trainer: ConcreteTrainer, tmp_path: Path
) -> None:
    """Test health_check when GPU is not available."""
    mock_cuda.is_available.return_value = False
    trainer.config["output_dir"] = str(tmp_path / "out")

    health = trainer.health_check()
    assert health["passed"] is True
    assert health["details"]["gpu"]["available"] is False


@patch("src.training.base_training.Path.mkdir")
@patch("src.training.base_training.torch.cuda")
def test_health_check_non_writable_dir(
    mock_cuda: MagicMock, mock_mkdir: MagicMock, trainer: ConcreteTrainer
) -> None:
    """Test health_check with a non-writable output directory."""
    mock_cuda.is_available.return_value = False
    mock_mkdir.side_effect = OSError("Permission denied")

    health = trainer.health_check()
    assert health["passed"] is False
    assert health["details"]["output_dir"]["writable"] is False
    assert "Permission denied" in health["details"]["output_dir"]["error"]


# ===================================================================
# White-Box Tests
# ===================================================================


@patch("src.training.base_training.torch.cuda")
def test_start_hardware_monitoring(
    mock_cuda: MagicMock, trainer: ConcreteTrainer
) -> None:
    """Test start_hardware_monitoring calls reset per GPU."""
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 2

    trainer.start_hardware_monitoring()
    assert mock_cuda.reset_peak_memory_stats.call_count == 2
    mock_cuda.reset_peak_memory_stats.assert_any_call(0)
    mock_cuda.reset_peak_memory_stats.assert_any_call(1)
    assert trainer.start_time > 0


@patch("src.training.base_training.resource")
@patch("src.training.base_training.torch.cuda")
def test_record_hardware_metrics(
    mock_cuda: MagicMock, mock_resource: MagicMock, trainer: ConcreteTrainer
) -> None:
    """Test record_hardware_metrics calculates VRAM correctly per GPU."""
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 2
    mock_cuda.max_memory_allocated.side_effect = [
        1024**2 * 500,
        1024**2 * 1500,
    ]  # 500MB and 1500MB

    mock_resource.getrusage.return_value.ru_maxrss = 1024 * 1024 * 2  # 2GB
    mock_resource.RUSAGE_SELF = 0

    trainer.start_time = 0.0
    with patch("src.training.base_training.time.time", return_value=3600.0):
        metrics = trainer.record_hardware_metrics()

    assert metrics["total_time_seconds"] == 3600.0
    assert metrics["total_time_hours"] == 1.0
    assert metrics["peak_cpu_ram_gb"] == 2.0
    assert metrics["gpu_vram"]["gpu_0_peak_vram_mb"] == 500.0
    assert metrics["gpu_vram"]["gpu_1_peak_vram_mb"] == 1500.0


def test_parse_results_csv_stripping_and_fallback(
    trainer: ConcreteTrainer, tmp_path: Path
) -> None:
    """Test parse_results_csv properly strips keys and falls back for unparseable floats."""
    csv_path = tmp_path / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["  key1  ", "key2 "])
        writer.writerow(["  1.23  ", " not_a_float "])

    data = trainer.parse_results_csv(csv_path)
    assert len(data) == 1
    assert "key1" in data[0]
    assert data[0]["key1"] == 1.23
    assert data[0]["key2"] == "not_a_float"


@patch("src.training.base_training.Path.mkdir")
@patch("src.training.base_training.torch.cuda")
def test_health_check_oserror_branch(
    mock_cuda: MagicMock, mock_mkdir: MagicMock, trainer: ConcreteTrainer
) -> None:
    """Test health_check correctly handles OSError when creating output dir."""
    mock_cuda.is_available.return_value = False
    mock_mkdir.side_effect = OSError("Read-only file system")

    health = trainer.health_check()
    assert health["passed"] is False
    assert health["details"]["output_dir"]["writable"] is False
    assert "Read-only" in health["details"]["output_dir"]["error"]
