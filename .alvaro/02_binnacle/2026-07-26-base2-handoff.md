# Base 2 (Classic Augmentation) — session handoff

Recorded: 2026-07-26, America/Lima.

## Remote run

- Kaggle kernel: `fabianapachecopalo/f1-base2-production`
- Kernel version: 1
- Account: Fabiana
- Code cloned by the running notebook: branch `13-f1-kaggle-runner`, commit
  `01ee0cf`
- Scientific condition: **Base 2 (Classic Augmentation)**.
- Internal implementation key and artifact prefix: `c2` / `f1_c2`.
- Initial observed remote state: `KernelWorkerStatus.RUNNING`.

## Completed remote gates

1. Two Tesla T4 GPUs were detected by Kaggle.
2. Editable package installation completed. Kaggle emitted pre-installed
   dependency-resolver warnings, but they did not stop the run.
3. Unit-test gate passed: 45 tests.
4. The Drive OAuth token refreshed successfully. The earlier `invalid_grant`
   / `invalid_client` failures did not recur.
5. Dataset preflight reported `OVERALL: PASS`.
6. The shared Raw/LaMa training manifest contains 43,310 frames; 79 frames are
   excluded consistently from the F1-family training sets. Validation contains
   10,873 untouched Raw frames.

## Batch calibration outcome

The disposable calibration uses dense images and the real Base 2 augmentation
profile. It must not be reported as an article metric result.

| Global batch | Result | Notes |
|---:|---|---|
| 96 | Rejected | Repeated `CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU`; roughly 5,500--5,800 instances per Mosaic batch. |
| 48 | Selected | Completed without that fallback; `nbs=96`, expected accumulation `2`, effective global update batch `96`. |

Both calibration trials verified engagement of 2/2 expected GPUs.

## Production configuration currently launched

```text
run name:       f1_c2
scientific name: Base 2 (Classic Augmentation)
device:         0,1 (two Tesla T4 GPUs)
batch:          48 global / 24 per GPU
nbs:            96
epochs:         40
patience:       5
seed:           42
optimizer:      AdamW
lr0:            0.001
weight_decay:   0.00075
mosaic:         1.0
mixup:          0.15
close_mosaic:   10
```

At the last inspection, the production log had reached `Starting training for
40 epochs`. It had CUDA initialisation for both GPUs and no production
TaskAlignedAssigner CPU fallback, general CUDA OOM, traceback, or fatal error.
No production batch/epoch line had appeared yet, so actual per-batch GPU
utilisation and the first Drive checkpoint sync remain to be confirmed.

## Safe monitoring commands

Keep the Kaggle credential outside the repository and substitute it through a
local secret or environment variable.

```bash
KAGGLE_API_TOKEN="$FABIANAS_KAGGLE_TOKEN" \
KAGGLE_CONFIG_DIR=/tmp/kaggle-fabiana-publish \
kaggle kernels status fabianapachecopalo/f1-base2-production

KAGGLE_API_TOKEN="$FABIANAS_KAGGLE_TOKEN" \
KAGGLE_CONFIG_DIR=/tmp/kaggle-fabiana-publish \
kaggle kernels logs -f fabianapachecopalo/f1-base2-production
```

Useful regex for a saved log:

```bash
rg -n -i 'F1 TRAINING \[f1_c2\]|Starting training for 40|^[[:space:]]*[0-9]+/40|TaskAlignedAssigner|using CPU|CUDA out of memory|Traceback|\[FATAL\]|GPU USAGE \[f1_c2\]|Engaged [0-9]+ of|Drive sync' /tmp/fabiana_base2_live.log
```

## Required next checks

1. Confirm the first Base 2 epoch has batch progress and validation output.
2. Confirm no CPU-fallback warning occurs after the `f1_c2` production marker.
3. Confirm the first `Drive sync completed` line and check the dedicated Base 2
   checkpoints folder for `f1_c2_last.pt`, `f1_c2_best.pt`, and
   `f1_c2_checkpoint_state.json`.
4. On completion, record final epoch count, elapsed time, GPU utilisation,
   metrics, and Drive artifact names in the execution record.
