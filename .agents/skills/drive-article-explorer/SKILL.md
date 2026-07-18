---
name: drive-article-explorer
description: Provides the directory tree structure of the Google Drive folder /content/drive/MyDrive/ia_article (used for the article) up to commit 85eadca. Use to view and check available files and paths on Drive.
license: MIT
---

# Drive Article Explorer

This skill provides a cached representation of the Google Drive directory structure used for the article's data and results. It helps agents reference files in Colab environments without requiring active filesystem queries.

---

## 1. Environment & Setup

When executing notebooks or scripts in Google Colab, mount Google Drive using the following standard snippet:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The target directory for the article is:
`/content/drive/MyDrive/ia_article`

---

## 2. Directory Tree Structure (Up to Commit 85eadca)

Below is the directory layout of `/content/drive/MyDrive/ia_article` frozen up to commit `85eadca`:

```
.
├── 00_raw
│   ├── sample_submission.csv
│   ├── test.zip
│   ├── train.csv
│   └── train.zip
├── 01_processed
│   ├── smart_dataset.yaml
│   ├── split_metadata.csv
│   └── yolo_obb_labels.zip
└── temp
    ├── prototipo_prep_files
    │   └── prototipo_prep_10_0.png
    └── prototipo_prep.md
```

---

## 3. Directory & File Details

### `00_raw/`
Contains raw inputs and datasets directly downloaded or provided for the task:
- `sample_submission.csv`: Template format for final submissions.
- `train.csv`: Training metadata or raw labels.
- `train.zip`: Raw training image dataset zip.
- `test.zip`: Raw testing image dataset zip.

### `01_processed/`
Contains processed, split, and structured datasets ready for model training:
- `smart_dataset.yaml`: Dataset configuration file (usually for YOLO training).
- `split_metadata.csv`: Details of training, validation, and test splits.
- `yolo_obb_labels.zip`: Processed Oriented Bounding Box labels zip in YOLO format.

### `temp/`
Contains temporary working files, visual prototype logs, and pre-processing summaries:
- `prototipo_prep.md`: Markdown documenting the prototype preprocessing phase.
- `prototipo_prep_files/prototipo_prep_10_0.png`: Visual output showing pre-processing samples/plots.

---

## 4. Updates & Maintenance
This representation is current as of commit `85eadca`. The tree and details will be updated progressively as new files are processed or directories are added.
