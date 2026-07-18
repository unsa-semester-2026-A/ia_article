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

- **Base Dependencies**: Minimal CPU-only packages (`numpy`, `opencv-python`, `pandas`, `pyyaml`, `tqdm`).
- **Optional Dependencies (`cloud`)**: Used on Colab/Kaggle. Installs secondary packages (`ultralytics`, `pytest`, `pytest-cov`, `albumentations`, `wandb`, `google-api-python-client`, `google-auth-httplib2`, `google-auth-oauthlib`) without touching PyTorch.
- **Optional Dependencies (`local`)**: Installs base, secondary, and local PyTorch/Torchvision binaries.
- **Dependency Groups (`dev`)**: Locked toolchain for developer environment (`pytest`, `pyright`, `ruff`, `pytest-cov`).

---

## 3. Recommended 5-Step Development Workflow

All developers and AI agents working on this repository **must** strictly adhere to the following workflow before making any commits:

### Step 1: Rapid Prototyping (Interactive Notebooks)
Explore datasets and run initial algorithms on a small subset of data (few images/clips) using scratch notebooks inside `experiments/notebooks/`.

### Step 2: Modularization
Move the tested code into structured, clean classes or functions inside `experiments/src/`. Avoid hardcoded configurations.

### Step 3: Colocated Unit Tests
Write comprehensive unit tests colocated next to your production scripts. Verify that 100% of tests pass using:
```bash
uv run pytest src/
```

### Step 4: Ruff Code Formatting & Lint Compliance
Ruff formatting and style checks are **compulsory**. Run these commands to auto-format and fix lints:
```bash
uv run ruff format src/
uv run ruff check src/ --fix
```
The checks must output `All checks passed!` without any warnings before committing code.

### Step 5: Deployment in a Lightweight Orchestrator Notebook
Execute the production script on the full cloud dataset (Kaggle/Colab) using a lightweight orchestrator notebook (like `final-notebook-optimized.ipynb`). The notebook must perform:
1. Sparse clone to fetch only the lightweight `experiments/` directory.
2. Editable installation: `%pip install -e .[cloud]`.
3. Pre-flight check via `!pytest src/`.
4. Script execution via `%run src/path_to_module/script.py`.

---

## 4. Google Drive Sharing & Data Storage Guidelines

To handle heavy datasets (~60GB zipped) across different cloud runtimes or shared Google accounts:

- **Google Drive Sharing**: Share the folder as "Editor". On the secondary account, add a shortcut to "My Drive" (Organize > Add shortcut > My Drive).
- **Fast SSD I/O**: Never unzip dataset files directly inside Google Drive. Copy the `.zip` archive from Drive to the local VM disk `/content/` and unzip it there for extremely fast SSD reads.
- **Output Sync**: Only write small serialized metadata (like `.json` or `.txt` label zips) back to Drive. Use direct Drive API uploads with `supportsAllDrives=True` to bypass local storage quotas.
