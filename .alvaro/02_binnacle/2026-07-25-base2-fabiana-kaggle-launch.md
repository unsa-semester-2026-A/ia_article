# Base 2 (Classic Augmentation) — Fabiana Kaggle launch record

Date: 2026-07-25
Status at recording time: running.

## Scope

This record covers the Base 2 (Classic Augmentation) launch on Fabiana's Kaggle
account. The internal implementation key is `c2`; it is not the scientific
condition name used in the evaluation plan.

## Published notebook

- Kernel: `fabianapachecopalo/f1-base2-production`
- Kaggle version: 1
- Git branch and commit: `13-f1-kaggle-runner` at `01ee0cf`
- Accelerator request: two NVIDIA Tesla T4 GPUs
- Internet: enabled
- Attached private inputs: `alvaroquispeunsa/mtc-challenge` and
  `alvaroquispeunsa/ia-article-drive-token`

## Reproducible publication and inspection commands

The access token must be supplied through a local secret or environment
variable; it must never be committed or copied into a notebook.

```bash
# Publish the private Base 2 notebook with Fabiana's Kaggle credential.
KAGGLE_API_TOKEN="$FABIANAS_KAGGLE_TOKEN" \
KAGGLE_CONFIG_DIR=/tmp/kaggle-fabiana-publish \
kaggle kernels push -p /tmp/kaggle-production-c2-v2

# Query execution state.
KAGGLE_API_TOKEN="$FABIANAS_KAGGLE_TOKEN" \
KAGGLE_CONFIG_DIR=/tmp/kaggle-fabiana-publish \
kaggle kernels status fabianapachecopalo/f1-base2-production

# Follow the remote output when available.
KAGGLE_API_TOKEN="$FABIANAS_KAGGLE_TOKEN" \
KAGGLE_CONFIG_DIR=/tmp/kaggle-fabiana-publish \
kaggle kernels logs -f fabianapachecopalo/f1-base2-production
```

## Notebook execution sequence

1. Confirm that two Tesla T4 GPUs are visible.
2. Sparse-clone `experiments/` from the pinned branch.
3. Install the editable cloud package and run the Base 2 trainer and calibration
   unit tests.
4. Run the training preflight with the shared Raw/LaMa train manifest allowed.
5. Run the disposable DDP batch calibration.
6. Read `c2_batch_selection.json` and start Base 2 with the selected
   `--c2-batch` value.
7. Use the attached Drive token for production checkpoint and result uploads.

## Observed remote evidence

- Two Tesla T4 GPUs were detected.
- The unit-test gate passed: 45 passed.
- The Google Drive OAuth token was refreshed successfully; the previous
  `invalid_grant` failure did not recur.
- Preflight reported `OVERALL: PASS`.
- Calibration rejected global batch 96 after repeated TaskAlignedAssigner CPU
  fallbacks on dense Mosaic batches (about 5,500--5,800 instances per batch).
- Calibration accepted global batch 48 with `nbs=96` and expected gradient
  accumulation of 2, preserving an effective global update batch of 96.
- Production Base 2 then started with global `batch=48` on DDP.

## Pending checks

- Confirm the first periodic Drive checkpoint upload in the production log.
- Confirm that production Base 2 emits no TaskAlignedAssigner CPU fallback.
- Record final status, elapsed time, metrics, and uploaded artifact names after
  the run completes.
