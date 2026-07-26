# Mejora B and Mejora C Kaggle runbook

The two final F1 runs use the frozen SAM copy-paste release, never synthetic
validation data.

| Condition | Real training pixels | Synthetic delta | Kaggle account | Run name |
| --- | --- | --- | --- | --- |
| Mejora B | raw `train_resized/train` | `augmentation_images/images/train` + matching labels | Dolly | `f1_mejora_b` |
| Mejora C | `smart_lama_corrected/train` | the exact same delta | Alvaro | `f1_mejora_c` |

Both use YOLO26s-OBB, seed 42, 640 px, global DDP batch 96 (48 per T4), 40
epochs, patience 5, AMP, no RAM cache, and the minimal online augmentation
profile. This is intentionally the same profile as Base 1 and Mejora A.

Launch from the Kaggle notebook after installing the editable `experiments`
package:

```bash
python -m src.training.trainers.train_base_1 --condition mb
python -m src.training.trainers.train_base_1 --condition mc
```

The runner fails before training if a synthetic image/label pair is missing,
empty, malformed, out of the nine classes, outside normalized coordinates, or
collides with a real frame. It writes `augmentation_delta_manifest.json` into
the temporary dataset workspace. Validation is linked exclusively from the raw
real validation split.

Each condition sends results and checkpoints to distinct Drive folders. The
callback uploads `last.pt`, `best.pt`, and `checkpoint_state.json` every five
epochs and at completion; files are prefixed with their run name.
