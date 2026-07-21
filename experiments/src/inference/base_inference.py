import time
import resource
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseInferencePipeline(ABC):
    """Clase base abstracta para orquestar la inferencia de modelos YOLO OBB."""

    def __init__(self, config: dict[str, Any], model_path: str) -> None:
        """Inicializa parámetros base de hardware, modelo y contadores.

        Args:
            config: Configuración general (device, conf, iou, imgsz, etc.).
            model_path: Ruta a los pesos del modelo YOLO (.pt).
        """
        self.config: dict[str, Any] = config
        self.model_path: Path = Path(model_path)
        self.device: str | int = config.get("device", 0)

        # Acumuladores de tiempos (suma + conteo) en vez de listas enormes
        # Esto evita que miles de frames consuman RAM innecesaria
        self._time_sums: dict[str, float] = {
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
        }
        self.total_frames_processed: int = 0

        self.start_time: float = 0.0
        self.total_time_seconds: float = 0.0
        self.peak_cpu_ram_gb: float = 0.0
        self.peak_gpu_vram_gb: float = 0.0

    def accumulate_speed(self, speed: dict[str, float]) -> None:
        """Acumula tiempos de un frame sin almacenar cada valor individual.

        Args:
            speed: Diccionario con claves 'preprocess', 'inference', 'postprocess' en ms.
        """
        for key in self._time_sums:
            self._time_sums[key] += speed.get(key, 0.0)
        self.total_frames_processed += 1

    def start_hardware_monitoring(self) -> None:
        """Reinicia las estadísticas de la GPU (VRAM) usando torch.cuda.reset_peak_memory_stats
        y registra el tiempo de inicio total.
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()

    def record_hardware_metrics(self) -> dict[str, Any]:
        """Consulta el consumo máximo de memoria RAM (resource.ru_maxrss) y VRAM (torch.cuda.max_memory_allocated).
        Calcula también la velocidad promedio del modelo.

        Returns:
            Estructura compatible con 'inference_metrics.json'.
        """
        self.total_time_seconds = time.time() - self.start_time

        # Peak RAM (ru_maxrss is in kilobytes on Linux)
        peak_ram_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.peak_cpu_ram_gb = peak_ram_kb / (1024 * 1024)

        # Peak VRAM
        if torch.cuda.is_available():
            peak_vram_bytes = torch.cuda.max_memory_allocated()
            self.peak_gpu_vram_gb = peak_vram_bytes / (1024 ** 3)
            peak_vram_mb = peak_vram_bytes / (1024 ** 2)
        else:
            self.peak_gpu_vram_gb = 0.0
            peak_vram_mb = 0.0

        n = self.total_frames_processed or 1  # Evitar división por cero
        avg_preprocess = self._time_sums["preprocess"] / n
        avg_inference = self._time_sums["inference"] / n
        avg_postprocess = self._time_sums["postprocess"] / n

        avg_total_ms = avg_preprocess + avg_inference + avg_postprocess
        theoretical_fps = (1000.0 / avg_total_ms) if avg_total_ms > 0 else 0.0

        metrics = {
            "experiment_condition": self.config.get("experiment_condition", "Unknown"),
            "hardware": self.config.get("hardware_name", "Unknown"),
            "total_frames_processed": self.total_frames_processed,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "peak_cpu_ram_gb": round(self.peak_cpu_ram_gb, 2),
            "average_speed_ms": {
                "preprocess": round(avg_preprocess, 2),
                "inference": round(avg_inference, 2),
                "postprocess": round(avg_postprocess, 2),
            },
            "theoretical_fps": round(theoretical_fps, 2),
        }

        return metrics

    @abstractmethod
    def execute(self) -> dict[str, Any]:
        """Ejecuta el pipeline completo de inferencia y guardado por lotes."""
        pass
