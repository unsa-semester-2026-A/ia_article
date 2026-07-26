"""Background sampling of per-GPU utilization through ``nvidia-smi``.

``torch.cuda.max_memory_allocated`` only reports allocations made by the calling
process. Ultralytics runs multi-GPU training by spawning one DDP child process
per device, so the parent process sees activity on a single GPU and cannot prove
that every device did work. Polling ``nvidia-smi`` observes the whole machine and
therefore captures the children as well.
"""

import shutil
import subprocess
import threading
from typing import Any

#: A device holding less than this much memory is considered idle. Loading CUDA
#: context alone costs a few hundred MiB, so the threshold sits above that to
#: avoid reporting a GPU as engaged when no model was ever placed on it.
ENGAGED_MEMORY_THRESHOLD_MIB = 512

_QUERY_FIELDS = "index,name,utilization.gpu,memory.used,memory.total"


def nvidia_smi_available() -> bool:
    """Report whether the ``nvidia-smi`` binary can be found on PATH."""
    return shutil.which("nvidia-smi") is not None


def sample_gpus() -> list[dict[str, Any]]:
    """Take a single snapshot of every visible GPU.

    Returns:
        One dictionary per GPU with ``index``, ``name``, ``utilization_pct``,
        ``memory_used_mib`` and ``memory_total_mib``. Empty if ``nvidia-smi`` is
        unavailable or returns unparseable output.
    """
    if not nvidia_smi_available():
        return []

    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={_QUERY_FIELDS}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []

    samples: list[dict[str, Any]] = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            samples.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_pct": float(parts[2]),
                    "memory_used_mib": float(parts[3]),
                    "memory_total_mib": float(parts[4]),
                }
            )
        except ValueError:
            continue
    return samples


class GpuSampler:
    """Poll ``nvidia-smi`` on a daemon thread and aggregate per-device peaks.

    Example:
        >>> sampler = GpuSampler(interval_seconds=5.0)
        >>> sampler.start()
        >>> ...  # run training
        >>> stats = sampler.stop()
    """

    def __init__(self, interval_seconds: float = 5.0) -> None:
        """Initialize the sampler.

        Args:
            interval_seconds: Delay between consecutive snapshots.
        """
        self.interval_seconds = interval_seconds
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._per_gpu: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        """Begin sampling in the background. A no-op if already running."""
        if self._thread is not None or not nvidia_smi_available():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _record(self, samples: list[dict[str, Any]]) -> None:
        with self._lock:
            for sample in samples:
                index = sample["index"]
                entry = self._per_gpu.setdefault(
                    index,
                    {
                        "index": index,
                        "name": sample["name"],
                        "memory_total_mib": sample["memory_total_mib"],
                        "peak_memory_used_mib": 0.0,
                        "peak_utilization_pct": 0.0,
                        "utilization_sum": 0.0,
                        "samples": 0,
                    },
                )
                entry["peak_memory_used_mib"] = max(
                    entry["peak_memory_used_mib"], sample["memory_used_mib"]
                )
                entry["peak_utilization_pct"] = max(
                    entry["peak_utilization_pct"], sample["utilization_pct"]
                )
                entry["utilization_sum"] += sample["utilization_pct"]
                entry["samples"] += 1

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._record(sample_gpus())
            self._stop_event.wait(self.interval_seconds)

    def stop(self) -> dict[str, Any]:
        """Stop sampling and return the aggregated report.

        Returns:
            Dictionary with ``available``, ``devices`` (one entry per GPU with its
            peaks and mean utilization), ``gpu_count`` and ``gpus_engaged``. The
            last field is the assertion target for multi-GPU runs: it counts the
            devices that held a real workload rather than just a CUDA context.
        """
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=self.interval_seconds + 5)
            self._thread = None

        with self._lock:
            devices: list[dict[str, Any]] = []
            for index in sorted(self._per_gpu):
                entry = dict(self._per_gpu[index])
                samples = entry.pop("samples")
                utilization_sum = entry.pop("utilization_sum")
                entry["mean_utilization_pct"] = round(
                    utilization_sum / samples if samples else 0.0, 2
                )
                entry["sample_count"] = samples
                entry["peak_memory_used_mib"] = round(entry["peak_memory_used_mib"], 1)
                devices.append(entry)

        engaged = [
            d
            for d in devices
            if d["peak_memory_used_mib"] >= ENGAGED_MEMORY_THRESHOLD_MIB
        ]
        return {
            "available": nvidia_smi_available(),
            "gpu_count": len(devices),
            "gpus_engaged": len(engaged),
            "engaged_threshold_mib": ENGAGED_MEMORY_THRESHOLD_MIB,
            "devices": devices,
        }
