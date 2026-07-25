"""Unit tests for the nvidia-smi based GPU sampler."""

import subprocess
import time
from unittest.mock import MagicMock, patch

from src.utils.gpu_monitor import (
    ENGAGED_MEMORY_THRESHOLD_MIB,
    GpuSampler,
    nvidia_smi_available,
    sample_gpus,
)

TWO_T4 = "0, Tesla T4, 97, 9000, 15360\n1, Tesla T4, 95, 8800, 15360\n"


def _mock_smi(stdout: str) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    return result


# ==========================================
# sample_gpus Tests
# ==========================================
def test_sample_gpus_without_nvidia_smi():
    """Black Box: Return an empty sample list when the binary is absent."""
    with patch("src.utils.gpu_monitor.shutil.which", return_value=None):
        assert sample_gpus() == []
        assert nvidia_smi_available() is False


def test_sample_gpus_parses_two_devices():
    """White Box: Parse index, name, utilization and memory for every device."""
    with patch("src.utils.gpu_monitor.shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch("src.utils.gpu_monitor.subprocess.run", return_value=_mock_smi(TWO_T4)):
            samples = sample_gpus()

    assert [s["index"] for s in samples] == [0, 1]
    assert samples[0]["name"] == "Tesla T4"
    assert samples[0]["utilization_pct"] == 97.0
    assert samples[1]["memory_used_mib"] == 8800.0
    assert samples[1]["memory_total_mib"] == 15360.0


def test_sample_gpus_ignores_malformed_lines():
    """White Box: A truncated or non-numeric row must not break the sampler."""
    noisy = "0, Tesla T4, 97, 9000, 15360\ngarbage line\n1, Tesla T4, N/A, 100, 15360\n"
    with patch("src.utils.gpu_monitor.shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch("src.utils.gpu_monitor.subprocess.run", return_value=_mock_smi(noisy)):
            samples = sample_gpus()

    assert [s["index"] for s in samples] == [0]


def test_sample_gpus_survives_subprocess_failure():
    """Black Box: A failing nvidia-smi degrades to no samples, not an exception."""
    with patch("src.utils.gpu_monitor.shutil.which", return_value="/usr/bin/nvidia-smi"):
        with patch(
            "src.utils.gpu_monitor.subprocess.run",
            side_effect=subprocess.TimeoutExpired("nvidia-smi", 10),
        ):
            assert sample_gpus() == []


# ==========================================
# GpuSampler Tests
# ==========================================
def test_sampler_reports_both_gpus_engaged():
    """Black Box: Two busy devices are reported as two engaged GPUs."""
    sampler = GpuSampler(interval_seconds=0.01)
    with patch("src.utils.gpu_monitor.nvidia_smi_available", return_value=True):
        sampler._record(
            [
                {
                    "index": 0,
                    "name": "Tesla T4",
                    "utilization_pct": 90.0,
                    "memory_used_mib": 9000.0,
                    "memory_total_mib": 15360.0,
                },
                {
                    "index": 1,
                    "name": "Tesla T4",
                    "utilization_pct": 70.0,
                    "memory_used_mib": 8800.0,
                    "memory_total_mib": 15360.0,
                },
            ]
        )
        report = sampler.stop()

    assert report["gpu_count"] == 2
    assert report["gpus_engaged"] == 2
    assert report["devices"][0]["peak_memory_used_mib"] == 9000.0


def test_sampler_detects_an_idle_second_gpu():
    """White Box: A device holding only a CUDA context is not counted as engaged."""
    sampler = GpuSampler(interval_seconds=0.01)
    with patch("src.utils.gpu_monitor.nvidia_smi_available", return_value=True):
        sampler._record(
            [
                {
                    "index": 0,
                    "name": "Tesla T4",
                    "utilization_pct": 90.0,
                    "memory_used_mib": 9000.0,
                    "memory_total_mib": 15360.0,
                },
                {
                    "index": 1,
                    "name": "Tesla T4",
                    "utilization_pct": 0.0,
                    "memory_used_mib": ENGAGED_MEMORY_THRESHOLD_MIB - 1,
                    "memory_total_mib": 15360.0,
                },
            ]
        )
        report = sampler.stop()

    assert report["gpu_count"] == 2
    assert report["gpus_engaged"] == 1


def test_sampler_aggregates_peaks_and_means_across_samples():
    """White Box: Peaks take the maximum while utilization is averaged."""
    sampler = GpuSampler(interval_seconds=0.01)
    for util, mem in [(10.0, 1000.0), (90.0, 5000.0), (50.0, 3000.0)]:
        sampler._record(
            [
                {
                    "index": 0,
                    "name": "Tesla T4",
                    "utilization_pct": util,
                    "memory_used_mib": mem,
                    "memory_total_mib": 15360.0,
                }
            ]
        )

    with patch("src.utils.gpu_monitor.nvidia_smi_available", return_value=True):
        device = sampler.stop()["devices"][0]

    assert device["peak_memory_used_mib"] == 5000.0
    assert device["peak_utilization_pct"] == 90.0
    assert device["mean_utilization_pct"] == 50.0
    assert device["sample_count"] == 3


def test_sampler_start_is_noop_without_nvidia_smi():
    """Black Box: Without nvidia-smi the sampler stays inert and reports nothing."""
    sampler = GpuSampler(interval_seconds=0.01)
    with patch("src.utils.gpu_monitor.nvidia_smi_available", return_value=False):
        sampler.start()
        report = sampler.stop()

    assert report["available"] is False
    assert report["devices"] == []


def test_sampler_thread_collects_samples():
    """White Box: The background thread actually polls while training runs."""
    sampler = GpuSampler(interval_seconds=0.01)
    with patch("src.utils.gpu_monitor.nvidia_smi_available", return_value=True):
        with patch("src.utils.gpu_monitor.shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch(
                "src.utils.gpu_monitor.subprocess.run", return_value=_mock_smi(TWO_T4)
            ):
                sampler.start()
                time.sleep(0.08)
                report = sampler.stop()

    assert report["gpus_engaged"] == 2
    assert report["devices"][0]["sample_count"] >= 1
