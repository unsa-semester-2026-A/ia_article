# F1 training matrix

The three F1 conditions use the same `YOLO26s-OBB` architecture, seed, image
size, optimizer, epoch budget and validation split. The only experimental
variables are the training pixels and augmentation profile.

| Order | Scientific condition | Internal key | Training pixels | Augmentation profile | Run name |
|---:|---|---|---|---|---|
| 1 | Base 1 (Raw Data) | `c1` | Raw 640x360, shared 43,310-frame manifest | Minimal geometric | `f1_c1` |
| 2 | Base 2 (Classic Augmentation) | `c2` | Raw 640x360, same manifest as Base 1 | Classic YOLO: mosaic, mixup, copy-paste | `f1_c2` |
| 3 | Improvement A (LaMa Data) | `c3` | LaMa 640x360, same manifest as Base 1/Base 2 | Minimal geometric, identical to Base 1 | `f1_c3` |

## Execution protocol

1. Set the internal `RUN_CONDITION` key in `f1-kaggle-runner.ipynb` for the
   intended scientific condition shown above.
2. Run the smoke test and require every preflight check, Drive upload and
   dual-GPU report to pass.
3. Run production only after that condition reports `CLEARED FOR PRODUCTION`.
4. Before Base 2 production, run `python -m src.training.trainers.run_c2_batch_calibration`.
   It probes the DDP batch ladder `96 → 48 → 32 → 24` with the real Base 2
   augmentation and stores its local decision in
   `/kaggle/working/c2_batch_calibration/c2_batch_selection.json`. Launch Base 2
   only with its selected `--c2-batch` value; the probe is operational and is
   not included in the article comparison.
5. Run one condition at a time; do not overlap training runs on the shared
   Drive folders.

## Storage layout

No additional Drive folders are required. The configured results and checkpoint
folders are shared safely because every remote artifact is prefixed by run name
(for example, `f1_c2_last.pt` and `f1_c3_training_metrics.json`). Kaggle also
preserves each run under `/kaggle/working/runs/f1_c*` as notebook output.
