"""Small, measurable IC-Light integration smoke runs.

This module deliberately evaluates execution integrity, not visual realism.  A
human must inspect the saved images before synthetic data is admitted into a
training set.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from src.augmentation.render import relight_variant
from src.utils.gpu_monitor import GpuSampler


def _ram_gib() -> float:
    """Return the process high-water RAM mark in GiB on Linux."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)


def run_smoke_batch(
    client: Any,
    jobs: list[dict[str, Any]],
    output_dir: Path,
    *,
    working_size: int | tuple[int, int] = 256,
    steps: int = 2,
    sample_interval_seconds: float = 0.25,
    render: Callable[..., Path] = relight_variant,
) -> dict[str, Any]:
    """Render a bounded batch and persist an auditable metrics report.

    Args:
        client: Ready IC-Light client.
        jobs: Records with ``id``, ``foreground_bgra``, ``background_path`` and
            ``seed``.  Foregrounds must already be full-frame BGRA layers.
        output_dir: Directory where JPEG outputs and the JSON report are saved.
        working_size: IC-Light input edge or ``(width, height)`` pair; every
            dimension must be divisible by 64.
        steps: Diffusion steps per image.
        sample_interval_seconds: ``nvidia-smi`` polling interval.
        render: Injectable render function for CPU tests.

    Returns:
        The report written to ``iclight_smoke_metrics.json``.
    """
    if not jobs:
        raise ValueError("Smoke batch must contain at least one job")
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler = GpuSampler(interval_seconds=sample_interval_seconds)
    started_at = time.time()
    started = time.perf_counter()
    sampler.start()
    rows: list[dict[str, Any]] = []
    try:
        for job in jobs:
            job_id = str(job["id"])
            destination = output_dir / "images" / f"{job_id}.jpg"
            image_started = time.perf_counter()
            row: dict[str, Any] = {"id": job_id, "seed": int(job["seed"])}
            try:
                result = render(
                    client,
                    np.asarray(job["foreground_bgra"]),
                    Path(job["background_path"]),
                    destination,
                    int(job["seed"]),
                    working_size=working_size,
                    steps=steps,
                )
                image = cv2.imread(str(result), cv2.IMREAD_COLOR)
                if image is None or image.shape != (360, 640, 3):
                    raise RuntimeError(
                        f"Invalid output {result}: "
                        f"{None if image is None else image.shape}"
                    )
                row.update(
                    {
                        "status": "passed",
                        "output": str(result),
                        "shape": list(image.shape),
                        "bytes": result.stat().st_size,
                    }
                )
            except Exception as exc:  # report every failed item before stopping
                row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                rows.append(row)
                raise
            row["elapsed_seconds"] = round(time.perf_counter() - image_started, 3)
            rows.append(row)
    finally:
        gpu = sampler.stop()

    elapsed = time.perf_counter() - started
    report = {
        "status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
        "started_at_unix": started_at,
        "batch_size": len(jobs),
        "working_size": working_size,
        "steps": steps,
        "elapsed_seconds": round(elapsed, 3),
        "images_per_second": round(len(rows) / elapsed, 4) if elapsed else 0.0,
        "peak_process_ram_gib": _ram_gib(),
        "gpu": gpu,
        "images": rows,
        "quality_scope": "execution-only; manual visual review required",
    }
    (output_dir / "iclight_smoke_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
