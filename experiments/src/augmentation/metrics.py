"""Resumable resource metrics for long-running augmentation jobs."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.gpu_monitor import GpuSampler, sample_gpus


@dataclass
class ProductionMonitor:
    """Sample CPU, memory, disk and GPU usage into a JSON-safe report.

    The monitor intentionally writes its report at every stage boundary. A
    cancelled Kaggle session therefore leaves enough evidence to decide whether
    it can resume and whether resources were actually used.
    """

    output_path: Path
    working_directory: Path
    interval_seconds: float = 5.0
    _started_at: float = field(default=0.0, init=False)
    _gpu_sampler: GpuSampler = field(init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _peak_rss_bytes: int = field(default=0, init=False)
    _peak_system_memory_percent: float = field(default=0.0, init=False)
    _peak_cpu_percent: float = field(default=0.0, init=False)
    _samples: int = field(default=0, init=False)
    _stages: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Create the shared GPU sampler."""
        self._gpu_sampler = GpuSampler(self.interval_seconds)

    def start(self) -> None:
        """Start background sampling and persist the initial snapshot."""
        self._started_at = time.monotonic()
        self._gpu_sampler.start()
        self._record()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.write(status="running")

    def _run(self) -> None:
        """Sample process/system usage until stopped."""
        while not self._stop_event.wait(self.interval_seconds):
            self._record()

    def _record(self) -> None:
        """Record one CPU/RAM observation without requiring psutil at import time."""
        try:
            import psutil

            process = psutil.Process(os.getpid())
            rss = int(process.memory_info().rss)
            memory_percent = float(psutil.virtual_memory().percent)
            cpu_percent = float(process.cpu_percent(interval=None))
        except (ImportError, OSError):
            rss, memory_percent, cpu_percent = 0, 0.0, 0.0
        self._peak_rss_bytes = max(self._peak_rss_bytes, rss)
        self._peak_system_memory_percent = max(
            self._peak_system_memory_percent, memory_percent
        )
        self._peak_cpu_percent = max(self._peak_cpu_percent, cpu_percent)
        self._samples += 1

    def stage(self, name: str, **details: Any) -> None:
        """Checkpoint a named stage and persist the in-progress metrics report."""
        self._record()
        self._stages.append(
            {
                "name": name,
                "elapsed_seconds": round(time.monotonic() - self._started_at, 2),
                **details,
            }
        )
        self.write(status="running")

    def _report(self, status: str, error: str | None = None) -> dict[str, Any]:
        """Build the current JSON report."""
        disk = shutil.disk_usage(self.working_directory)
        return {
            "status": status,
            "error": error,
            "elapsed_seconds": round(time.monotonic() - self._started_at, 2),
            "sample_count": self._samples,
            "process": {
                "peak_rss_bytes": self._peak_rss_bytes,
                "peak_rss_gib": round(self._peak_rss_bytes / 1024**3, 3),
                "peak_cpu_percent": round(self._peak_cpu_percent, 2),
                "peak_system_memory_percent": round(
                    self._peak_system_memory_percent, 2
                ),
            },
            "disk": {
                "path": str(self.working_directory),
                "free_gib": round(disk.free / 1024**3, 3),
                "used_gib": round(disk.used / 1024**3, 3),
            },
            "gpu_before_or_latest": sample_gpus(),
            "gpu_usage": self._gpu_sampler.stop() if status != "running" else {},
            "stages": self._stages,
        }

    def write(self, *, status: str, error: str | None = None) -> Path:
        """Atomically write the current report and return its path."""
        report = self._report(status, error)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(self.output_path)
        return self.output_path

    def stop(self, *, error: str | None = None) -> Path:
        """Stop sampling and write the final success or failure report."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 5)
            self._thread = None
        self._record()
        return self.write(status="failed" if error else "success", error=error)
