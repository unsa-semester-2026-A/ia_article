# Base 2 (Classic Augmentation): TaskAlignedAssigner memory incident

Date: 2026-07-25

## Experiment terminology

The scientific condition is **Base 2 (Classic Augmentation)**, as defined in
`07_evaluation.md`. The implementation identifier `c2` is only an internal
runner key. The relevant study names are:

- Base 1: Raw Data.
- Base 2: Classic Augmentation.
- Improvement A: LaMa Data.

## Incident

The first Base 2 run with global `batch=96` (48 images per Tesla T4) repeatedly
logged `CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU`. Training did
not stop, but target assignment moved to CPU and epoch time was no longer a
useful hardware measurement. The run was stopped before spending more GPU
quota.

## Implemented mitigation

An operational calibration was added before Base 2 production training. It is
not a scientific result and it does not upload artifacts to Google Drive.

1. Use the same raw images, labels, seed, model, and Base 2 augmentation
   profile as production.
2. Select a deterministic 384-image subset with the largest label counts from
   the shared Raw/LaMa manifest. This intentionally stresses Mosaic and the
   target assigner.
3. Test global DDP batches in descending order: `96 -> 48 -> 32 -> 24`.
4. Reject any candidate whose complete log contains the TaskAlignedAssigner CPU
   fallback warning, even if the process exits successfully.
5. Use the first clean candidate for production and save the decision locally
   as `/kaggle/working/c2_batch_calibration/c2_batch_selection.json`.

For a selected batch below 96, `nbs=96` is used so gradient accumulation keeps
the effective global update at 96 images. Weight decay is adjusted to preserve
the scaling of the original `batch=96, nbs=64` recipe. This is a resource
configuration change only; it does not change the training split, validation
split, annotations, seed, epoch cap, or patience.

If 96, 48, and 32 fail, the predefined fallback is 24. If 24 also triggers the
CPU fallback, Base 2 must not be launched until a new experimental decision is
documented.

## Technical basis

Ultralytics catches this specific CUDA OOM inside `TaskAlignedAssigner` and
falls back to CPU. Multi-GPU DDP does not automatically retry a lower batch.
Ultralytics also derives gradient accumulation from `nbs` and the global batch.

References:

- https://docs.ultralytics.com/reference/utils/tal/
- https://docs.ultralytics.com/reference/engine/trainer/
- https://docs.ultralytics.com/modes/train/
