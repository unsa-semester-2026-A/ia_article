## Dependencies

- `make`
- `texlive-full`

```sh
# Installation in Debian based systems
sudo apt update
sudo apt install -y \
    texlive-full \
    make
```

## Usage

> [!NOTE]
> There is two types of compilation, the `complete` and the `fast`
> use the second one for fast check and the first one for the final result


- Complete compilation

```sh
make
```

- Fast compilation (to view your changes)

```sh
make fast
```

See the pdf generated in `./latex_report/build/article.pdf`

## Google Drive Directory Structure

To collaborate across local environments, Google Colab, and Kaggle, the project stores datasets and outputs in shared Google Drive folders:

* **02_pseudo_labeling** (Folder ID: `1J5ogC3q6jyYlk3wuYyxpYZHslUg6eGtN`): Contains the static vehicles detection outputs, cleaning checkpoints, and processed datasets.
* **05_evaluations** (Folder ID: `1VdM16679CS9t7dE60ShEK2tAqv9kPd9p`): Contains the validation set homographies and the evaluation metrics/summaries for all models.

