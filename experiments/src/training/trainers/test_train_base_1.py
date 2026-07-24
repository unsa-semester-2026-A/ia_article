import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from src.training.trainers.train_base_1 import Base1Trainer


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    """Provide a default configuration dictionary for Base1Trainer."""
    return {
        "output_dir": str(tmp_path / "runs"),
        "model_weights": "yolo26s-obb.pt",
        "labels_path": str(tmp_path / "yolo_obb_labels.zip"),
        "labels_zip_path": str(tmp_path / "yolo_obb_labels.zip"),
        "data_yaml_path": str(tmp_path / "dataset" / "smart_dataset.yaml"),
        "images_dir": str(tmp_path / "images_source"),
        "dataset_workspace": str(tmp_path / "dataset"),
        "drive_folder_id": "dummy_folder_id",
        "experiment_condition": "Base_1_Raw_Data",
    }


@pytest.fixture
@patch("src.training.trainers.train_base_1.IOManager")
def trainer(mock_io_manager: MagicMock, config: dict[str, Any]) -> Base1Trainer:
    """Provide a Base1Trainer instance with a mocked IOManager."""
    trainer = Base1Trainer(config)
    trainer.io_manager = mock_io_manager.return_value
    trainer.io_manager.drive_service = MagicMock()
    return trainer


# ===================================================================
# Black-Box Tests
# ===================================================================


def test_get_hyperparameters_defaults(trainer: Base1Trainer) -> None:
    """Test get_hyperparameters returns default values when no overrides exist."""
    params = trainer.get_hyperparameters()
    for key, value in Base1Trainer.DEFAULT_HYPERPARAMS.items():
        assert params[key] == value


def test_get_hyperparameters_overrides(trainer: Base1Trainer) -> None:
    """Test get_hyperparameters applies allowed config overrides."""
    trainer.config["epochs"] = 50
    trainer.config["batch"] = 16
    params = trainer.get_hyperparameters()
    assert params["epochs"] == 50
    assert params["batch"] == 16


def test_get_dataset_config(trainer: Base1Trainer, config: dict[str, Any]) -> None:
    """Test get_dataset_config returns correct paths."""
    dataset_config = trainer.get_dataset_config()
    assert dataset_config["data_yaml_path"] == config["data_yaml_path"]
    assert dataset_config["model_weights"] == config["model_weights"]
    assert dataset_config["labels_path"] == config["labels_path"]
    assert dataset_config["images_dir"] == config["images_dir"]


def test_prepare_dataset_valid(trainer: Base1Trainer, tmp_path: Path) -> None:
    """Test prepare_dataset with valid zip and images dir."""
    labels_zip = Path(trainer.config["labels_path"])
    images_dir = Path(trainer.config["images_dir"])

    # Create dummy files
    labels_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(labels_zip, "w") as zf:
        zf.writestr("dummy.txt", "data")

    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "train").mkdir()
    (images_dir / "val").mkdir()

    yaml_path = trainer.prepare_dataset()

    assert yaml_path.name == "smart_dataset.yaml"
    workspace = Path(trainer.config["dataset_workspace"])
    assert (workspace / "labels").exists()
    assert (workspace / "images" / "train").exists()


def test_prepare_dataset_missing_zip(trainer: Base1Trainer) -> None:
    """Test prepare_dataset raises FileNotFoundError if zip is missing."""
    with pytest.raises(FileNotFoundError):
        trainer.prepare_dataset()


def test_prepare_dataset_pre_existing_workspace(
    trainer: Base1Trainer, tmp_path: Path
) -> None:
    """Test prepare_dataset skips if workspace is already populated."""
    yaml_path = Path(trainer.config["data_yaml_path"])
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.touch()

    labels_dir = yaml_path.parent / "labels"
    labels_dir.mkdir()
    (labels_dir / "dummy.txt").touch()

    images_dir = yaml_path.parent / "images"
    images_dir.mkdir()
    (images_dir / "dummy.jpg").touch()

    # If it tries to process zip, it will fail because zip doesn't exist
    # If it skips properly, it returns yaml_path
    result = trainer.prepare_dataset()
    assert result == yaml_path


def test_health_check_valid(trainer: Base1Trainer, tmp_path: Path) -> None:
    """Test health_check with valid zip, images dir, and Drive service."""
    labels_zip = Path(trainer.config["labels_path"])
    images_dir = Path(trainer.config["images_dir"])

    labels_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(labels_zip, "w") as zf:
        zf.writestr("dummy.txt", "data")

    images_dir.mkdir(parents=True, exist_ok=True)

    with patch(
        "src.training.base_training.torch.cuda.is_available", return_value=False
    ):
        health = trainer.health_check()

    assert health["passed"] is True
    assert health["details"]["labels"]["exists"] is True
    assert health["details"]["images_dir"]["exists"] is True
    assert health["details"]["drive_service"]["available"] is True


def test_health_check_corrupt_zip(trainer: Base1Trainer, tmp_path: Path) -> None:
    """Test health_check with a corrupt zip file."""
    labels_zip = Path(trainer.config["labels_path"])
    images_dir = Path(trainer.config["images_dir"])

    labels_zip.parent.mkdir(parents=True, exist_ok=True)
    labels_zip.write_text("not a zip file")
    images_dir.mkdir(parents=True, exist_ok=True)

    with patch(
        "src.training.base_training.torch.cuda.is_available", return_value=False
    ):
        health = trainer.health_check()

    assert health["passed"] is False
    assert health["details"]["labels"]["valid"] is False


def test_health_check_missing_images_dir(trainer: Base1Trainer) -> None:
    """Test health_check with missing images dir."""
    labels_zip = Path(trainer.config["labels_path"])
    labels_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(labels_zip, "w") as zf:
        zf.writestr("dummy.txt", "data")

    with patch(
        "src.training.base_training.torch.cuda.is_available", return_value=False
    ):
        health = trainer.health_check()

    assert health["passed"] is False
    assert health["details"]["images_dir"]["exists"] is False


# ===================================================================
# White-Box Tests
# ===================================================================


def test_get_hyperparameters_unallowed_override(trainer: Base1Trainer) -> None:
    """Test get_hyperparameters only accepts specific overrides."""
    trainer.config["mosaic"] = 1.0  # Not in override_keys
    params = trainer.get_hyperparameters()
    assert params["mosaic"] == 0.0  # Default value, not overridden


@patch("src.training.trainers.train_base_1.zipfile.ZipFile")
@patch("src.training.trainers.train_base_1.shutil.copytree")
@patch("src.training.trainers.train_base_1.Path.symlink_to")
def test_prepare_dataset_symlink_and_fallback(
    mock_symlink: MagicMock,
    mock_copytree: MagicMock,
    mock_zip: MagicMock,
    trainer: Base1Trainer,
    tmp_path: Path,
) -> None:
    """Test prepare_dataset calls extractall, symlink, and falls back to copytree on OSError."""
    labels_zip = Path(trainer.config["labels_zip_path"])
    images_dir = Path(trainer.config["images_dir"])

    labels_zip.parent.mkdir(parents=True, exist_ok=True)
    labels_zip.touch()

    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "train").mkdir()
    (images_dir / "val").mkdir()

    # Simulate OSError on the second symlink call (val)
    mock_symlink.side_effect = [None, OSError("Symlink not supported")]

    trainer.prepare_dataset()

    mock_zip.return_value.__enter__.return_value.extractall.assert_called_once()
    assert mock_symlink.call_count == 2
    assert mock_copytree.call_count == 1
    mock_copytree.assert_called_once_with(
        str(images_dir / "val"),
        str(Path(trainer.config["dataset_workspace"]) / "images" / "val"),
    )


def test_save_and_upload_error_branch(trainer: Base1Trainer, tmp_path: Path) -> None:
    """Test _save_and_upload handles Drive upload error."""
    local_path = tmp_path / "data.json"
    trainer.io_manager.upload_and_verify_file.side_effect = Exception("Upload failed")

    result = trainer._save_and_upload({"data": 1}, local_path, "folder_id")

    trainer.io_manager.save_json.assert_called_once_with({"data": 1}, local_path)
    assert result["local"] == str(local_path)
    assert result["drive_id"] is None


def test_upload_file_mime_types(trainer: Base1Trainer, tmp_path: Path) -> None:
    """Test _upload_file selects correct MIME types for different extensions."""
    test_files = [
        ("data.csv", "text/csv"),
        ("config.yaml", "text/yaml"),
        ("plot.png", "image/png"),
        ("model.pt", "application/octet-stream"),
        ("unknown.xyz", "application/octet-stream"),
    ]

    for fname, expected_mime in test_files:
        fpath = tmp_path / fname
        trainer._upload_file(fpath, "folder_id")
        trainer.io_manager.upload_and_verify_file.assert_called_with(
            fpath, "folder_id", mime_type=expected_mime
        )


def test_upload_file_strict_raises_on_drive_error(
    trainer: Base1Trainer, tmp_path: Path
) -> None:
    """A checkpoint-sync failure must stop training instead of being hidden."""
    weights_path = tmp_path / "last.pt"
    trainer.io_manager.upload_and_verify_file.side_effect = Exception("Drive offline")

    with pytest.raises(RuntimeError, match="Required Drive sync failed"):
        trainer._upload_file(weights_path, "folder_id", strict=True)


@patch("ultralytics.YOLO")
@patch.object(Base1Trainer, "detect_device")
@patch.object(Base1Trainer, "prepare_dataset")
@patch.object(Base1Trainer, "parse_results_csv")
def test_execute_success(
    mock_parse: MagicMock,
    mock_prepare: MagicMock,
    mock_detect_device: MagicMock,
    mock_yolo: MagicMock,
    trainer: Base1Trainer,
    tmp_path: Path,
) -> None:
    """Test execute method including results parsing and drive uploads."""
    mock_prepare.return_value = Path("dummy.yaml")
    mock_parse.return_value = [{"epoch": 1.0}]
    mock_detect_device.return_value = "0,1"

    # Setup YOLO mock to succeed on train
    mock_model = MagicMock()
    mock_model.train.return_value = None
    mock_yolo.return_value = mock_model

    # Create dummy output files
    train_dir = Path(trainer.config["output_dir"]) / "base1"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "results.csv").touch()
    (train_dir / "results.png").touch()
    (train_dir / "weights").mkdir()
    (train_dir / "weights" / "best.pt").touch()

    # Set drive IDs
    trainer.io_manager.upload_file_to_drive.return_value = "dummy_id"

    result = trainer.execute()

    # Assertions
    assert result["status"] == "success"
    assert "metrics" in result
    assert result["metrics"]["oom_fallback_triggered"] is False

    assert mock_model.train.call_count == 1
    mock_parse.assert_called_once()

    # Check uploads
    uploaded_files = [f["local"] for f in result["files"]]
    assert any("training_metrics.json" in f for f in uploaded_files)
    assert any("results.csv" in f for f in uploaded_files)
    assert any("results.png" in f for f in uploaded_files)
    assert any("best.pt" in f for f in uploaded_files)


def test_get_hyperparameters_fast_dev_run(tmp_path):
    """Black Box: Verify fast_dev_run sets epochs=1 and fraction=0.01 for smoke testing."""
    config = {
        "output_dir": str(tmp_path),
        "fast_dev_run": True,
    }
    trainer = Base1Trainer(config=config)
    params = trainer.get_hyperparameters()
    assert params["epochs"] == 1
    assert params["fraction"] == 0.01
    assert params["save_period"] == 1
