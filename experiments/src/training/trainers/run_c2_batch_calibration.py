"""Select a reproducible C2 batch before the production DDP training run.

The C2 augmentation profile can create far more OBB instances per batch than
C1/C3.  This utility runs disposable one-epoch trials on the same two GPUs,
starting from the planned global batch and descending only when Ultralytics
reports that ``TaskAlignedAssigner`` had to move to CPU.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.training.trainers.train_base_1 import Base1Trainer

TASK_ALIGNED_ASSIGNER_FALLBACK = (
    "CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU"
)


def evaluate_trial_output(return_code: int, output: str) -> dict[str, Any]:
    """Classify a completed calibration trial from its complete console output.

    The TaskAlignedAssigner catches its own CUDA OOM and continues on CPU, so a
    successful process exit alone is insufficient.  The warning is part of the
    public Ultralytics behaviour and is therefore the explicit rejection signal.
    """
    used_cpu_fallback = TASK_ALIGNED_ASSIGNER_FALLBACK in output
    return {
        "return_code": return_code,
        "task_aligned_assigner_cpu_fallback": used_cpu_fallback,
        "ddp_requested": "Multi-GPU DDP:" in output,
        "accepted": return_code == 0 and not used_cpu_fallback,
    }


def calibration_command(batch: int, calibration_images: int) -> list[str]:
    """Return the isolated child command for one DDP calibration candidate."""
    return [
        sys.executable,
        "-m",
        "src.training.trainers.train_base_1",
        "--condition",
        "c2",
        "--c2-calibration-batch",
        str(batch),
        "--calibration-images",
        str(calibration_images),
    ]


def run_trial(command: list[str], log_path: Path) -> tuple[int, str]:
    """Run one candidate, preserving a live console stream and a full log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_parts: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output_parts.append(line)
            log_file.write(line)
            print(line, end="", flush=True)
        return_code = process.wait()
    return return_code, "".join(output_parts)


def _default_output_dir() -> Path:
    """Keep disposable calibration artifacts outside the experiment's Drive tree."""
    if os.path.exists("/kaggle/working"):
        return Path("/kaggle/working/c2_batch_calibration")
    return Path("/tmp/c2_batch_calibration")


def main() -> int:
    """Run the candidate ladder and write the auditable selected configuration."""
    parser = argparse.ArgumentParser(description="C2 DDP batch calibration")
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=list(Base1Trainer.C2_CALIBRATION_CANDIDATES),
        help="Descending global DDP batch candidates",
    )
    parser.add_argument(
        "--calibration-images",
        type=int,
        default=384,
        help="Deterministic dense train images used in every trial",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Local destination for logs and c2_batch_selection.json",
    )
    args = parser.parse_args()

    candidates = tuple(args.candidates)
    if not candidates:
        parser.error("At least one calibration candidate is required.")
    if tuple(sorted(candidates, reverse=True)) != candidates:
        parser.error("Candidates must be passed in descending order.")
    unsupported = set(candidates) - set(Base1Trainer.C2_CALIBRATION_CANDIDATES)
    if unsupported:
        parser.error(
            f"Unsupported candidates {sorted(unsupported)}; expected a subset of "
            f"{Base1Trainer.C2_CALIBRATION_CANDIDATES}."
        )
    if args.calibration_images < max(candidates):
        parser.error("--calibration-images must be at least the largest candidate.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "purpose": "Operational C2 DDP batch calibration; not an article metric run.",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "candidates": [],
        "selected_batch": None,
        "selection_rule": (
            "First descending candidate with exit code 0 and without the "
            "TaskAlignedAssigner CPU fallback warning."
        ),
        "task_aligned_assigner_cpu_fallback_warning": TASK_ALIGNED_ASSIGNER_FALLBACK,
        "calibration_images": args.calibration_images,
    }

    for batch in candidates:
        print(
            f"\n{'=' * 72}\nC2 batch calibration candidate: global batch={batch}\n{'=' * 72}"
        )
        command = calibration_command(batch, args.calibration_images)
        log_path = output_dir / f"c2_batchcal_b{batch}.log"
        return_code, output = run_trial(command, log_path)
        verdict = evaluate_trial_output(return_code, output)
        plan = Base1Trainer.c2_batch_plan(batch)
        trial = {
            "batch": batch,
            "command": command,
            "log_path": str(log_path),
            "plan": plan,
            **verdict,
        }
        report["candidates"].append(trial)
        if verdict["accepted"]:
            report["selected_batch"] = batch
            report["selected_plan"] = plan
            break

    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    report_path = output_dir / "c2_batch_selection.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    selected_batch = report["selected_batch"]
    if selected_batch is None:
        print(
            "\n[FAILED] No candidate avoided the TaskAlignedAssigner CPU fallback. "
            f"Review {report_path} before altering the experimental recipe."
        )
        return 2

    selected_plan = report["selected_plan"]
    print(
        "\n[SELECTED] C2 production configuration: "
        f"--c2-batch {selected_batch} "
        f"(nbs={selected_plan['nbs']}, "
        f"expected_accumulate={selected_plan['expected_accumulate']}, "
        f"effective_global_batch={selected_plan['effective_global_batch']})."
    )
    print(f"Selection report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
