"""Tests for the C2 DDP batch-calibration runner."""

from src.training.trainers.run_c2_batch_calibration import (
    TASK_ALIGNED_ASSIGNER_FALLBACK,
    calibration_command,
    evaluate_trial_output,
)


def test_trial_is_rejected_when_task_aligned_assigner_falls_back_to_cpu() -> None:
    """The warning must outweigh a successful process exit."""
    verdict = evaluate_trial_output(0, f"warning: {TASK_ALIGNED_ASSIGNER_FALLBACK}")
    assert verdict["accepted"] is False
    assert verdict["task_aligned_assigner_cpu_fallback"] is True


def test_trial_is_accepted_only_after_a_clean_exit() -> None:
    """A clean DDP trial is eligible to become the fixed production batch."""
    verdict = evaluate_trial_output(0, "Multi-GPU DDP: 2 GPUs detected")
    assert verdict["accepted"] is True
    assert verdict["ddp_requested"] is True


def test_calibration_command_uses_the_disposable_c2_mode() -> None:
    """A trial must not accidentally execute a production C2 run."""
    command = calibration_command(48, 384)
    assert command[-4:] == [
        "--c2-calibration-batch",
        "48",
        "--calibration-images",
        "384",
    ]
    assert "--condition" in command
    assert command[command.index("--condition") + 1] == "c2"
