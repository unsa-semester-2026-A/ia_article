# Final evaluation launch record

Date: 2026-07-26

## Code and validation gate

- Branch: `feature/final-evaluation`
- Pinned commit for the current independent evaluations:
  `f084b070009055f1419d583cb27d7c9b124a20df`
- Local validation before publication: `21 passed` for
  `src/evaluation/test_pipeline.py` and `src/evaluation/test_final_runner.py`.
- The runner creates one self-contained ZIP per condition and uploads it to the
  configured Drive result folder. It includes metrics, raw/filtered
  predictions, homographies, manifest, checkpoint checksum, and hardware data.

## Active independent Kaggle evaluations

These were published using Saúl's Kaggle credential. The credential itself is
not stored in this record or in Git.

| Condition | Kernel | Checkpoint | Drive result folder |
| --- | --- | --- | --- |
| Base 1 | `saulsiv/final-evaluation-base1-production` | `f1_c1_best.pt` from `1pn8OzJX_kctgluEZkSC6WEfbSCaPyKMa` | `1Roazcv3c72jGmGy2M0nrlx5kL75-BY3m` |
| Mejora A | `saulsiv/final-evaluation-mejoraa-production` | `f1_c3_best.pt` from `1Hi8OmTIMNzLfadjFbk79OL8yiIZewhpz` | `1Sf2SEMHhl3jQoqnGbXiuVXd_BpT6yBgf` |

Both kernels request the Kaggle `NvidiaTeslaT4` machine shape, use the shared
private inputs `alvaroquispeunsa/mtc-challenge` and
`alvaroquispeunsa/ia-article-drive-token`, and process the full validation
manifest (`yolo_obb_labels/val`). They are deliberately separate environments:
a failure in one must not prevent the other from producing a result.

## Base 0

Base 0 already completed successfully in the owner's Kaggle account and is not
to be relaunched. Its zero-shot DOTA mapping is explicit in the runner:

- DOTA class 10, `small vehicle` -> official SMART class 1, Auto.
- DOTA class 9, `large vehicle` -> official SMART class 7, Camion.

## Resolved incident

The first combined Base 1 + Mejora A evaluation stopped during Base 1 because
Ultralytics emitted an OBB with zero width or height. The evaluation adapter
previously raised an exception. Commit `f084b07` now discards only zero-area
model outputs (which cannot contribute to rIoU), while retaining strict errors
for negative dimensions, non-finite values, invalid shape, and unmapped
classes. A unit test covers this behavior.

The combined kernel `saulsiv/final-evaluation-base1-mejoraa-production` is a
failed obsolete run; do not use it for results.

## Useful inspection commands

Keep the Saúl token outside Git and supply it through a shell variable or local
credential file. Do not print it.

```bash
kaggle kernels status saulsiv/final-evaluation-base1-production
kaggle kernels logs -f saulsiv/final-evaluation-base1-production
kaggle kernels status saulsiv/final-evaluation-mejoraa-production
kaggle kernels logs -f saulsiv/final-evaluation-mejoraa-production
```

On completion, download the result ZIPs from their Drive folders and inspect
`summary.json`, `manifest.json`, and `hardware.json` inside each archive.
