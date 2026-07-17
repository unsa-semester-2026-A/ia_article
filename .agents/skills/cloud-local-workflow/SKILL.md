---
name: cloud-local-workflow
description: Guides the agent in executing research experiments using a unified Python package workflow across Local (using uv), Google Colab, and Kaggle platforms, enforcing editable packages, colocated tests, and memory-safe interactive execution.
---

# Cloud & Local Research Workflow Skill

This skill documents the unified package layout, dev-tool configurations, and execution workflows used to develop, test, and run machine learning models both on local developer machines and cloud GPU instances (Google Colab and Kaggle).

---

## 1. Directory Layout & Package Structure

The project `experiments` is organized as an installable Python package under the `src/` namespace. This avoids nested script import errors and allows absolute imports.

### Directory Tree
```
experiments/
├── pyproject.toml                     # Dependency definitions & dev tools config (PEP 621)
├── main_colab_kaggle.ipynb            # Cloud orchestrator notebook
├── notebooks/                         # Interactive prototype notebooks (dirty/exploration code)
└── src/                               # Production-ready package source code
    ├── data_preparation/
    │   ├── __init__.py
    │   ├── parser.py                  # CLI modules with argparse
    │   └── test_parser.py             # Colocated unit tests
    ├── pseudo_labeling/
    │   ├── __init__.py
    │   ├── tracker.py
    │   └── test_tracker.py
    └── utils/
        ├── __init__.py
        ├── geometry.py
        └── test_geometry.py
```

### Key Architectural Guidelines
- **Co-located Tests**: Test modules MUST be located next to the code they verify (e.g. `test_parser.py` sits directly next to `parser.py` inside `src/data_preparation/`).
- **No Numeric Prefixes**: Do not name Python directories or scripts with numbers (use `data_preparation`, not `01_data_preparation`).
- **Google Docstring Style**: All functions and classes must be documented strictly following the Google Python Style Guide. Pyright and Ruff are configured to enforce this.

---

## 2. Dependency Management & Platform Considerations

We follow PEP 621 for metadata and PEP 735 (via `uv`) for development dependency groups.

### PyTorch Cloud Exclusion Rule
`torch` and `torchvision` must **NEVER** be specified in the base `dependencies` or the `cloud` extra group. Colab and Kaggle provide heavily optimized, pre-installed PyTorch installations for their specific hardware. Redownloading PyTorch inside the VM is extremely slow and can break hardware acceleration.

- **Base Dependencies**: Minimal CPU-only packages (`numpy`, `opencv-python`, `pandas`, `pyyaml`).
- **Optional Dependencies (`cloud`)**: Used on Colab/Kaggle. Installs secondary packages (`ultralytics`, `pytest`, `pytest-cov`, `albumentations`, `wandb`) without touching PyTorch.
- **Optional Dependencies (`local`)**: Installs base, secondary, and local PyTorch/Torchvision binaries.
- **Dependency Groups (`dev`)**: Locked toolchain for developer environment (`pytest`, `pyright`, `ruff`, `pytest-cov`).

### Technical Edge Cases & Safe Mitigation

#### A. Ultralytics Implicit Dependency
`ultralytics` requires `torch` and `torchvision`. 
- **Colab/Kaggle**: When installing the `cloud` extra, `pip` detects the pre-installed PyTorch in the system and skips its download, functioning as expected.
- **Clean Cloud Instances (AWS, GCP, RunPod, Clean Docker)**: If the `cloud` extra is installed on a raw machine without PyTorch, `pip` will automatically force-download the default PyTorch from PyPI. **The `cloud` extra assumes a pre-existing PyTorch environment is already active.**

#### B. Local Binary Sources & GPU Acceleration
Installing the `local` extra directly from PyPI installs the default PyTorch binaries.
- **CUDA Support (Windows/Linux)**: PyPI installs PyTorch compiled with the latest supported CUDA version. If your local hardware uses a different CUDA version, use index URLs during installation:
  ```bash
  uv pip install -e .[local] --extra-index-url https://download.pytorch.org/whl/cu121
  ```
- **Apple Silicon (macOS)**: No extra index is needed; standard PyPI PyTorch works out of the box with MPS acceleration.

#### C. Version Parity
Colab and Kaggle update their pre-installed PyTorch/CUDA environments slowly (often months behind). Running a newer PyTorch version locally (e.g. 2.5) than in Colab (e.g. 2.2) can cause silent inference discrepancies.
- **Action**: Check Colab's PyTorch version (`import torch; print(torch.__version__)`) and anchor the local version in `pyproject.toml` or install arguments accordingly (e.g., `torch==2.x.x`).

---

## 3. Cloud Execution Workflow (Colab / Kaggle)

To run scripts efficiently in Colab or Kaggle:

### Step 1: Clone and Install Editable
Mount Google Drive, pull the latest code, and install the package using `%pip install -q -e .[cloud]`.
```python
# Synchronize code repository
import os
if not os.path.exists('article'):
    !git clone -q https://github.com/unsa-semester-2026-A/ia_article.git
    %cd article/experiments
else:
    %cd article
    !git pull -q
    %cd experiments

# Install the experiments package in editable mode
%pip install -q -e .[cloud]
```

### Step 2: Run Colocated Tests
Verify everything is working before running:
```bash
!pytest src/
```

### Step 3: Run Modules via `%run`
Always run Python scripts using the `%run` magic command instead of `!python`.
```python
%run src/data_preparation/parser.py --csv_path dummy.csv --images_dir . --output_dir ./dataset
```
*Why `%run`?*
Using `%run` executes the script inside the active IPython kernel space. If the script throws an error or finishes, the model, tensores, and variables remain alive in the Colab notebook memory, allowing immediate debugging.

---

## 4. Local Execution Workflow (using `uv`)

For local code editing and testing, use the `uv` toolchain to run commands in isolated environments:

### Lockfile Synchronization
To guarantee exact reproducibility across all developers and platforms, **all team members MUST commit the `uv.lock` file** to the Git repository.

### Running Tests and Coverage
```bash
uv run pytest src/
```

### Running Static Type Checking
Pyright is configured in strict mode. All parameter and return types must be fully annotated.
```bash
uv run pyright src/
```

### Running Linting and Code Formatting
Ruff checks for standard syntax formatting and enforces Google docstring conventions:
```bash
uv run ruff check src/
```

---

## 5. Google Drive Sharing & Data Storage Guidelines

To handle heavy datasets (~60GB zipped) across different cloud runtimes or shared Google accounts:

- **Google Drive Sharing**: Share the folder as "Editor". On the secondary account, add a shortcut to "My Drive" (Organize > Add shortcut > My Drive).
- **Fast SSD I/O**: Never unzip dataset files directly inside Google Drive. Copy the `.zip` archive from Drive to the local VM disk `/content/` and unzip it there for extremely fast SSD reads.
- **Output Sync**: Only write small serialized metadata (like `.json` or `.txt` label zips) back to Drive.
