# SAM copy-paste production handoff

## Kaggle dataset release

Dataset: `alvaroquispeunsa/mtc-challenge`.

Train-only augmentation release: `sam_copy_paste_sam_cp_production_v1/`.

- Images: `sam_copy_paste_sam_cp_production_v1/images/train/*.jpg`
- YOLO OBB labels: `sam_copy_paste_sam_cp_production_v1/labels/train/*.txt`
- Pairing/audit: `sam_copy_paste_sam_cp_production_v1/manifest.csv`

The release contains 1,501 paired synthetic images/labels and 3,391 inserted
instances. It contains no validation images or labels.

## Training contract

Keep the original validation split exactly unchanged. Link the release only to
the training split:

- Improvement B: raw training images + this release.
- Improvement C: LaMa training images + this exact release.

Use `src.training.trainers.train_base_1.Base1Trainer` configuration keys
`augmentation_delta_images_dir` and `augmentation_delta_labels_dir`; the
trainer adds them only under `images/train` and `labels/train`.

## Production evidence

Run ID: `sam_cp_production_v1`. The kernel completed in 652.35 seconds. GPU 0
(Tesla T4) peaked at 3,297 MiB VRAM and 99% utilization; validation remained
unchanged according to the release manifest.
