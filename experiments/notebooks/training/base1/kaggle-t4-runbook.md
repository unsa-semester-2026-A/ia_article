# Kaggle two-T4 runbook for F1 YOLO26s-OBB

This runbook is the reusable procedure for the F1 YOLO26s-OBB notebooks. It
uses the scientific names from `07_evaluation.md`:

- Base 1: Raw Data (`c1` internal key)
- Base 2: Classic Augmentation (`c2` internal key)
- Improvement A: LaMa Data (`c3` internal key)

The internal keys are implementation details used in run names and CLI flags;
they are not the names to use in the article's result tables.

## 1. Prepare a private Kaggle notebook

Create a folder containing the notebook and `kernel-metadata.json`:

```json
{
  "id": "<owner>/<notebook-slug>",
  "title": "<notebook title>",
  "code_file": "<notebook>.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": [
    "alvaroquispeunsa/mtc-challenge",
    "alvaroquispeunsa/ia-article-drive-token"
  ],
  "machine_shape": "NvidiaTeslaT4"
}
```

The notebook should start with `nvidia-smi` and reject a session that does not
expose the expected two GPUs. Kaggle provisions the hardware when the notebook
is launched; YOLO uses both cards only when the trainer receives `device="0,1"`.

## 2. Publish and monitor without exposing credentials

Keep the Kaggle token outside the repository. Replace the placeholder below
with a local environment variable or secret manager value.

```bash
KAGGLE_API_TOKEN="$KAGGLE_ACCESS_TOKEN" \
KAGGLE_CONFIG_DIR=/tmp/kaggle-session \
kaggle kernels push -p /path/to/notebook-directory

KAGGLE_API_TOKEN="$KAGGLE_ACCESS_TOKEN" \
KAGGLE_CONFIG_DIR=/tmp/kaggle-session \
kaggle kernels status <owner>/<notebook-slug>

KAGGLE_API_TOKEN="$KAGGLE_ACCESS_TOKEN" \
KAGGLE_CONFIG_DIR=/tmp/kaggle-session \
kaggle kernels logs -f <owner>/<notebook-slug>
```

## 3. Required notebook sequence

1. Verify both GPUs with `nvidia-smi`.
2. Sparse-clone the `experiments/` directory from the intended Git branch.
3. Install the package with `pip install -e .[cloud]`; do not install PyTorch
   separately because Kaggle provides its CUDA build.
4. Run colocated trainer tests and preflight.
5. Start the selected scientific condition only after Drive authentication,
   dataset checks, and dual-GPU checks pass.

The training command has the following reusable form:

```bash
python -m src.training.trainers.train_base_1 --condition <c1|c2|c3>
```

## 4. Base 2 memory calibration

Run this before a new Base 2 production run:

```bash
python -m src.training.trainers.run_c2_batch_calibration
```

It tests global DDP batches `96 -> 48 -> 32 -> 24` on a deterministic dense
subset. It writes the decision to:

```text
/kaggle/working/c2_batch_calibration/c2_batch_selection.json
```

Run Base 2 only with the selected batch:

```bash
python -m src.training.trainers.train_base_1 \
  --condition c2 \
  --c2-batch <selected_batch>
```

The calibration is operational only. It does not upload weights or contribute
metrics to the article. If the log contains
`TaskAlignedAssigner, using CPU`, reject that candidate.

## 5. Evidence to retain

For every production run, save the following in the corresponding execution
record:

- Git branch and commit cloned by Kaggle.
- Kernel URL, version, account, attached datasets, and requested hardware.
- Preflight status and Drive OAuth result.
- GPU evidence: `CUDA:0`, `CUDA:1`, DDP device `0,1`, and the final GPU usage
  report showing two engaged devices.
- Effective training arguments, selected Base 2 batch where applicable, final
  metrics, elapsed time, and Drive artifact names.

Never place Kaggle or Google Drive tokens in notebooks, logs, Git commits, or
this runbook.
