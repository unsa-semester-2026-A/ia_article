"""Abstract base class for orchestrating YOLO OBB training pipelines with hardware monitoring."""

import csv
import resource
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch


class BaseTrainingPipeline(ABC):
    """Abstract base class for YOLO OBB model training across ablation study conditions.

    Encapsulates the shared training workflow logic including:
    - Multi-GPU auto-detection and device assignment.
    - Hardware monitoring (peak VRAM per GPU, CPU RAM, elapsed time).
    - Per-epoch metric extraction from Ultralytics results.csv.
    - Periodic checkpoint backup tracking.
    - Google Drive sync coordination via IOManager.

    Subclasses (Base1Trainer, Base2Trainer, MejoraATrainer, etc.) override
    ``get_hyperparameters()`` and ``get_dataset_config()`` to provide
    condition-specific settings.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize base training pipeline with shared configuration.

        Args:
            config: General configuration dictionary containing:
                - output_dir: Local directory for training outputs.
                - drive_folder_id: Google Drive folder ID for sync (optional).
                - token_path: Path to Google Drive token.json (optional).
                - experiment_condition: Name of the ablation condition.
                - hardware_name: Description of hardware platform.
        """
        self.config: dict[str, Any] = config
        self.start_time: float = 0.0
        self.total_time_seconds: float = 0.0

    # ===================================================================
    # Abstract Methods (must be overridden by subclasses)
    # ===================================================================

    @abstractmethod
    def get_hyperparameters(self) -> dict[str, Any]:
        """Return the complete hyperparameter dictionary for model.train().

        Returns:
            Dictionary of YOLO training arguments (epochs, batch, lr0, etc.).
        """

    @abstractmethod
    def get_dataset_config(self) -> dict[str, Any]:
        """Return dataset-specific paths and configuration.

        Returns:
            Dictionary containing:
                - data_yaml_path: Path to the YOLO dataset YAML.
                - model_weights: Path to pretrained model weights (.pt).
                - labels_zip_path: Path to YOLO-OBB labels zip (optional).
                - images_dir: Path to images root directory (optional).
        """

    @abstractmethod
    def prepare_dataset(self) -> Path:
        """Prepare dataset workspace: unzip labels, symlink images, validate YAML.

        Returns:
            Path to the validated dataset YAML file.
        """

    # ===================================================================
    # GPU Detection
    # ===================================================================

    def detect_device(self) -> str:
        """Auto-detect available CUDA GPUs and return optimal device specification.

        Returns:
            Device specification string for YOLO model.train():
            - Comma-separated GPU indices ``"0,1"`` for multi-GPU DDP.
            - Single GPU index ``"0"`` for single GPU.
            - ``"cpu"`` if no CUDA GPUs available.
        """
        if not torch.cuda.is_available():
            print("[BaseTrainingPipeline] No CUDA GPUs found. Using CPU.", flush=True)
            return "cpu"
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 2:
            device_str = ",".join(str(i) for i in range(gpu_count))
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            print(
                f"[BaseTrainingPipeline] Multi-GPU DDP: {gpu_count} GPUs detected "
                f"({', '.join(gpu_names)}) -> device='{device_str}'",
                flush=True,
            )
            return device_str
        print(
            f"[BaseTrainingPipeline] Single GPU: {torch.cuda.get_device_name(0)} -> device='0'",
            flush=True,
        )
        return "0"

    # ===================================================================
    # Hardware Monitoring
    # ===================================================================

    def start_hardware_monitoring(self) -> None:
        """Reset GPU peak memory statistics and record training start timestamp."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.reset_peak_memory_stats(i)
                except Exception:
                    pass
        self.start_time = time.time()

    def record_hardware_metrics(self) -> dict[str, Any]:
        """Capture peak CPU RAM, per-GPU peak VRAM, and total elapsed time.

        Returns:
            Dictionary with hardware utilization metrics.
        """
        self.total_time_seconds = time.time() - self.start_time

        # Peak CPU RAM (ru_maxrss is in kilobytes on Linux)
        peak_ram_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_cpu_ram_gb = peak_ram_kb / (1024 * 1024)

        # Per-GPU peak VRAM
        gpu_vram: dict[str, float] = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    peak_bytes = torch.cuda.max_memory_allocated(i)
                    gpu_vram[f"gpu_{i}_peak_vram_mb"] = round(peak_bytes / (1024**2), 2)
                except Exception:
                    gpu_vram[f"gpu_{i}_peak_vram_mb"] = 0.0

        return {
            "experiment_condition": self.config.get("experiment_condition", "Unknown"),
            "hardware": self.config.get("hardware_name", "Unknown"),
            "total_time_seconds": round(self.total_time_seconds, 2),
            "total_time_hours": round(self.total_time_seconds / 3600, 3),
            "peak_cpu_ram_gb": round(peak_cpu_ram_gb, 2),
            "gpu_vram": gpu_vram,
        }

    # ===================================================================
    # Results Parsing
    # ===================================================================

    def parse_results_csv(self, results_csv_path: Path) -> list[dict[str, Any]]:
        """Parse Ultralytics results.csv into structured per-epoch metric records.

        Args:
            results_csv_path: Path to the results.csv generated by YOLO training.

        Returns:
            List of dictionaries, one per completed epoch, with cleaned column
            names and float-converted values.

        Raises:
            FileNotFoundError: If results_csv_path does not exist.
        """
        if not results_csv_path.exists():
            raise FileNotFoundError(f"Results CSV not found: {results_csv_path}")

        epochs_data: list[dict[str, Any]] = []
        with results_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned: dict[str, Any] = {}
                for key, value in row.items():
                    clean_key = key.strip()
                    try:
                        cleaned[clean_key] = float(value.strip())
                    except (ValueError, AttributeError):
                        cleaned[clean_key] = value.strip() if value else ""
                epochs_data.append(cleaned)

        return epochs_data

    # ===================================================================
    # Training Metrics Assembly
    # ===================================================================

    def build_training_metrics(
        self,
        epochs_data: list[dict[str, Any]],
        hardware_metrics: dict[str, Any],
        hyperparameters: dict[str, Any],
        effective_batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Assemble the complete training_metrics.json structure.

        Args:
            epochs_data: Parsed per-epoch metrics from results.csv.
            hardware_metrics: Hardware utilization metrics from record_hardware_metrics.
            hyperparameters: The hyperparameter dictionary used for training.
            effective_batch_size: Actual batch size used (may differ from requested
                if OOM fallback triggered). None means no override occurred.

        Returns:
            Complete structured dictionary ready for JSON serialization.
        """
        metrics: dict[str, Any] = {
            "experiment_condition": self.config.get("experiment_condition", "Unknown"),
            "total_epochs_completed": len(epochs_data),
            "hyperparameters": hyperparameters,
            "hardware": hardware_metrics,
            "per_epoch_metrics": epochs_data,
        }

        if effective_batch_size is not None:
            metrics["effective_batch_size"] = effective_batch_size
            metrics["oom_fallback_triggered"] = True
        else:
            metrics["oom_fallback_triggered"] = False

        return metrics

    # ===================================================================
    # Health Check
    # ===================================================================

    def health_check(self) -> dict[str, Any]:
        """Run pre-flight validation of resources required for training.

        Checks:
            1. CUDA GPU availability and count.
            2. Output directory writability.

        Returns:
            Dictionary with check results and overall pass/fail status.
        """
        checks: dict[str, Any] = {"passed": True, "details": {}}

        # GPU check
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_info: list[dict[str, Any]] = []
            for i in range(gpu_count):
                gpu_info.append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "vram_gb": round(
                            torch.cuda.get_device_properties(i).total_memory
                            / (1024**3),
                            1,
                        ),
                    }
                )
            checks["details"]["gpu"] = {
                "available": True,
                "count": gpu_count,
                "devices": gpu_info,
            }
        else:
            checks["details"]["gpu"] = {
                "available": False,
                "count": 0,
                "devices": [],
            }

        # Output directory check
        output_dir = Path(self.config.get("output_dir", "/tmp/training_output"))
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            checks["details"]["output_dir"] = {
                "path": str(output_dir),
                "writable": True,
            }
        except OSError as e:
            checks["details"]["output_dir"] = {
                "path": str(output_dir),
                "writable": False,
                "error": str(e),
            }
            checks["passed"] = False

        return checks

    # ===================================================================
    # Main Execution
    # ===================================================================

    @abstractmethod
    def execute(self) -> dict[str, Any]:
        """Execute the complete training pipeline.

        Returns:
            Dictionary containing training results, metrics, and file paths.
        """
