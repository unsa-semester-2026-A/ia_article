---
name: drive-article-explorer
description: Provides the directory tree structure of the Google Drive folder /content/drive/MyDrive/ia_article, including Shared Drive connection details and latest paths.
license: MIT
---

# Drive Article Explorer

This skill provides a cached representation of the Google Drive directory structure used for the article's data and results. It helps agents reference files in Colab environments without requiring active filesystem queries.

---

## 1. Environment & Setup

### 1.1 Direct Folder Mount
When executing notebooks or scripts in Google Colab, mount Google Drive using the following standard snippet:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The target directory for the article is:
`/content/drive/MyDrive/ia_article`

### 1.2 Google Drive API (Bypassing Local Storage Limits)
To upload large datasets or continuous checkpoints without consuming the user's personal Google Drive quota, the pipeline uses direct REST API queries with authorized team credentials.

* **OAuth2 Token Location**: `/content/drive/MyDrive/ia_article/token/token.json`
* **Authentication Scope**: `['https://www.googleapis.com/auth/drive.file']` (restricted scope, allows writing/reading files created by this app).
* **Shared Drives Support**: Since the destination folders are in a Shared Team Drive, all API calls **must** explicitly pass:
  - `supportsAllDrives=True` in all `.get()`, `.create()`, `.delete()`, and `.get_media()` calls.
  - `supportsAllDrives=True` and `includeItemsFromAllDrives=True` in `.list()` queries.

---

## 2. Directory Tree Structure (Updated)

Below is the layout of the project directories on Google Drive:

```
/content/drive/MyDrive/ia_article/
.
├── 00_raw/
│   ├── sample_submission.csv
│   ├── test.zip
│   ├── train.csv
│   └── train.zip
├── 01_processed/
│   ├── smart_dataset.yaml
│   ├── split_metadata.csv
│   └── yolo_obb_labels.zip
├── 02_pseudo_labeling/                  <── (Shared Folder ID: 1J5ogC3q6jyYlk3wuYyxpYZHslUg6eGtN)
│   ├── static_vehicles.json             <── Final consolidated pseudo-labels mapping
│   ├── smart_lama_640.zip               <── (Future) Cleaned dataset after LaMa
│   ├── smart_synthetic.zip              <── (Future) Synthetic augmented dataset
│   └── for_each_clip/                   <── (Checkpoints Folder ID: 1anPtHNwHYgcq4BImhbJ_xzouiJ-Sh035)
│       ├── v_009evckk5b.json            <── Single-clip checkpoint files
│       ├── v_016is7moli.json
│       └── ...
├── 05_evaluations/                      <── (Shared Folder ID: 1VdM16679CS9t7dE60ShEK2tAqv9kPd9p)
│   ├── validation_homographies.json     <── Serialized inter-frame homographies
│   └── base0/
│       └── ...
└── token/
    └── token.json                       <── Authorized OAuth2 team credentials
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

### `02_pseudo_labeling/`
Central shared repository for labels cleaning and augmentation phases:
- `static_vehicles.json`: Consolidated JSON mapping frame IDs to detected static vehicles coordinates.
- `for_each_clip/`: Subfolder holding atomic clip-level JSON checkpoints, allowing the pipeline to resume seamlessly in case of disconnects.

---

## 4. Updates & Maintenance
This representation is dynamically updated. Always ensure that `supportsAllDrives=True` is verified in any script performing direct file uploads or downloads.
