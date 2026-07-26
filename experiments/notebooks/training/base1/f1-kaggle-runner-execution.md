# F1 Kaggle runner — execution record

This file records the remote executions of `f1-kaggle-runner.ipynb` so that a
future iteration starts from observed evidence rather than an ambiguous Kaggle
status.

## 2026-07-25 — smoke test C1, version 5

- Notebook: [f1-kaggle-runner](https://www.kaggle.com/code/alvaroquispeunsa/f1-kaggle-runner)
- Kaggle version: 5
- Branch used by the notebook: `13-f1-kaggle-runner` at `23e6eab`
- Mode: `--condition c1 --smoke-test --smoke-images 10 --smoke-epochs 3`
- Requested accelerator: `NvidiaTeslaT4`
- Attached datasets: `alvaroquispeunsa/mtc-challenge` and
  `alvaroquispeunsa/ia-article-drive-token`
- Final status reported by `kaggle kernels status`: `KernelWorkerStatus.COMPLETE`
- Last-run timestamp reported by the CLI: `2026-07-25 17:30:30 UTC`

### Context and interpretation

Version 4 stopped before cloning the repository because Kaggle assigned one
Tesla P100 while the runner required two GPUs. Version 5 changes the runner so
that a one-GPU allocation can still exercise the dataset, YOLO training and
Drive paths. It deliberately leaves the final verdict as **not cleared for
production** when fewer than two GPUs are available, because DDP/NCCL and
dual-GPU utilization remain unverified.

The notebook output was inspected in the Kaggle web UI. The smoke training
completed successfully: three epochs in 53.65 seconds, DDP launched with
`--nproc_per_node 2`, and both Tesla T4 GPUs were engaged (peaks: 1113 MiB and
1103 MiB). The unit-test gate also passed: 226 passed, 1 skipped.

The run is nevertheless **not cleared for production** for two independent
reasons:

1. The LaMa training directory is missing 79 of the 43,389 label-derived image
   stems (`smart_lama_corrected/train` has 43,310). Raw and LaMa resolutions
   match at 640x360 for sampled shared images, but C1 and C3 would not use the
   same frames.
2. The Drive token was found in the attached private dataset but Google rejected
   it with `invalid_grant: Token has been expired or revoked.` As a result every
   smoke artifact remained local to the Kaggle session and cannot be used for
   checkpoint resume.

The smoke validation subset also selected ten val images with no OBB labels, so
its mAP values are necessarily zero. This does not invalidate the DDP smoke
result, but it is not a model-quality measurement.

## 2026-07-25 — smoke test C1, version 6

Version 6 used commit `7f12258` and the shared 43,310-frame Raw/LaMa training
manifest. The image-set, package, GPU and NCCL preflight checks passed; the
smoke completed three DDP epochs and engaged both T4 GPUs. The reported peaks
were 1113 MiB (GPU 0) and 1103 MiB (GPU 1).

Drive still failed. Kaggle copied the token from the private dataset, and its
SHA-256 matches the local `experiments/token.json`, but Google returned
`invalid_client: The provided client secret is invalid.` The copied file records
an expired access token (`2026-07-23T15:54:44`) and cannot be refreshed. Do not
start production until the exact token file uploaded to the private dataset has
been regenerated with valid OAuth client credentials and has passed preflight.

### Next iteration

1. The trainer now builds `common_train_stems.txt` from the intersection of
   labels, Raw and LaMa. All F1 conditions train on those 43,310 shared frames;
   validation remains Raw and unchanged. This is the chosen reproducible
   alternative to regenerating the 79 missing LaMa frames.
2. Dataset `alvaroquispeunsa/ia-article-drive-token` was updated to version 2
   with the renewed credential. The next smoke run must verify both Drive
   folders are reachable and that artifacts upload successfully.
3. Only run production after a session with two GPUs reports successful NCCL,
   two engaged devices and `CLEARED FOR PRODUCTION`.

## 2026-07-25 — Base 2 (Classic Augmentation), Fabiana account, version 1

- Notebook: [f1-base2-production](https://www.kaggle.com/code/fabianapachecopalo/f1-base2-production)
- Kaggle version: 1
- Branch and commit: `13-f1-kaggle-runner` at `01ee0cf`
- Scientific condition: **Base 2 (Classic Augmentation)**. The `c2` label is an
  internal runner identifier only.
- Initial remote status: `KernelWorkerStatus.RUNNING`

The notebook first validates the trainer with 45 unit tests, then executes a
disposable DDP memory calibration before production. Global batch 96 was
rejected because `TaskAlignedAssigner` fell back to CPU; global batch 48 was
selected cleanly. The production command is therefore equivalent to:

```bash
python -m src.training.trainers.train_base_1 --condition c2 --c2-batch 48
```

The calibration uses `nbs=96` and expected accumulation 2, retaining an
effective global update batch of 96. It does not upload to Drive or contribute
metrics to the article. Remote logs also confirmed two Tesla T4 devices,
`OVERALL: PASS` in preflight, and a successful OAuth token refresh. The next
required evidence is the first periodic Drive checkpoint upload.
