# F1 training matrix

The three F1 conditions use the same `YOLO26s-OBB` architecture, seed, image
size, optimizer, epoch budget and validation split. The only experimental
variables are the training pixels and augmentation profile.

| Order | Condition | Training pixels | Augmentation profile | Run name |
|---:|---|---|---|---|
| 1 | C1 | Raw 640x360, shared 43,310-frame manifest | Minimal geometric | `f1_c1` |
| 2 | C2 | Raw 640x360, same manifest as C1 | Classic YOLO: mosaic, mixup, copy-paste | `f1_c2` |
| 3 | C3 | LaMa 640x360, same manifest as C1/C2 | Minimal geometric, identical to C1 | `f1_c3` |

## Execution protocol

1. Set `RUN_CONDITION` in `f1-kaggle-runner.ipynb` to the next condition.
2. Run the smoke test and require every preflight check, Drive upload and
   dual-GPU report to pass.
3. Run production only after that condition reports `CLEARED FOR PRODUCTION`.
4. Run one condition at a time; do not overlap training runs on the shared
   Drive folders.

## Storage layout

No additional Drive folders are required. The configured results and checkpoint
folders are shared safely because every remote artifact is prefixed by run name
(for example, `f1_c2_last.pt` and `f1_c3_training_metrics.json`). Kaggle also
preserves each run under `/kaggle/working/runs/f1_c*` as notebook output.
