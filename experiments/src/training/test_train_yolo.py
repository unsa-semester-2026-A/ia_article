"""Unit tests for the YOLO OBB training orchestrator script."""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml
from src.training.train_yolo import load_config, main


def test_load_config() -> None:
    """Verifies that load_config successfully parses a YAML file and raises errors on missing file."""
    config_data = {
        "model": "yolo26s-obb.pt",
        "epochs": 100,
        "batch": 16,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        loaded = load_config(temp_path)
        assert loaded["model"] == "yolo26s-obb.pt"
        assert loaded["epochs"] == 100
        assert loaded["batch"] == 16
    finally:
        os.remove(temp_path)

    # Test FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config_file.yaml")


@patch("src.training.train_yolo.YOLO")
@patch("src.training.train_yolo.get_drive_service")
def test_main_execution(
    mock_get_drive_service: MagicMock, mock_yolo: MagicMock
) -> None:
    """Verifies that main training orchestrator loads args and triggers model.train."""
    config_data = {
        "model": "yolo26s-obb.pt",
        "epochs": 5,
        "batch": 4,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_config_path = f.name

    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance
    mock_results = MagicMock()
    mock_results.save_dir = "/tmp/runs/obb/test_run"
    mock_model_instance.train.return_value = mock_results

    mock_get_drive_service.return_value = (
        None  # Mock no Drive connection to skip API upload
    )

    test_args = [
        "train_yolo.py",
        "--config",
        temp_config_path,
        "--data",
        "test_dataset.yaml",
        "--project",
        "/tmp/runs/obb",
        "--name",
        "test_run",
    ]

    try:
        with patch.object(sys, "argv", test_args):
            main()

        # Verify YOLO was instantiated with model from config
        mock_yolo.assert_called_once_with("yolo26s-obb.pt")

        # Verify model.train was called with parameters
        mock_model_instance.train.assert_called_once()
        called_kwargs = mock_model_instance.train.call_args[1]
        assert called_kwargs["data"] == "test_dataset.yaml"
        assert called_kwargs["project"] == "/tmp/runs/obb"
        assert called_kwargs["name"] == "test_run"
        assert called_kwargs["epochs"] == 5
        assert called_kwargs["batch"] == 4
        assert called_kwargs["save_period"] == 10
    finally:
        os.remove(temp_config_path)
