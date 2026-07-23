"""Unit tests for IOManager module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.utils.io_manager import IOManager


@pytest.fixture
def io_mgr(tmp_path):
    """Fixture to instantiate IOManager without performing real Drive downloads."""
    with patch.object(IOManager, "_ensure_token_from_kaggle_secrets"):
        with patch.object(IOManager, "_get_drive_service", return_value=None):
            return IOManager(token_path=tmp_path / "dummy_token.json")


# ==========================================
# _ensure_token_from_kaggle_secrets Tests (White Box)
# ==========================================
def test_ensure_token_from_kaggle_secrets_when_missing(tmp_path):
    """White Box: Create token.json from Kaggle Secrets if missing."""
    import sys
    token_file = tmp_path / "token.json"
    assert not token_file.exists()

    mock_secrets_client = MagicMock()
    mock_secrets_client.get_secret.return_value = '{"token": "secret_token"}'
    mock_module = MagicMock()
    mock_module.UserSecretsClient.return_value = mock_secrets_client

    with patch.dict(sys.modules, {"kaggle_secrets": mock_module}):
        with patch.object(IOManager, "_get_drive_service", return_value=None):
            IOManager(token_path=token_file)

    assert token_file.exists()
    assert token_file.read_text() == '{"token": "secret_token"}'


def test_ensure_token_from_kaggle_secrets_raises_on_failure(tmp_path):
    """White Box: Program MUST fail (RuntimeError) if Kaggle Secrets fails to retrieve token."""
    import sys
    token_file = tmp_path / "token.json"
    assert not token_file.exists()

    mock_module = MagicMock()
    mock_module.UserSecretsClient.side_effect = Exception("Secrets error")

    with patch.dict(sys.modules, {"kaggle_secrets": mock_module}):
        with patch.object(IOManager, "_get_drive_service", return_value=None):
            with pytest.raises(RuntimeError, match="Failed to retrieve Kaggle secret"):
                IOManager(token_path=token_file)


# ==========================================
# _get_drive_service Tests (White Box)
# ==========================================
@patch("src.utils.io_manager.build")
@patch("src.utils.io_manager.Credentials")
def test_get_drive_service_uses_scopes_none(mock_creds, mock_build, tmp_path):
    """White Box: Verify Credentials uses scopes=None to prevent invalid_scope errors."""
    token_file = tmp_path / "token.json"
    token_file.write_text('{"token": "dummy"}')

    fake_creds = MagicMock()
    fake_creds.expired = False
    mock_creds.from_authorized_user_file.return_value = fake_creds

    with patch.object(IOManager, "_ensure_token_from_kaggle_secrets"):
        IOManager(token_path=token_file)

    mock_creds.from_authorized_user_file.assert_called_once_with(
        str(token_file), scopes=None
    )
    mock_build.assert_called_once_with("drive", "v3", credentials=fake_creds)


# ==========================================
# list_files_in_dir Tests (White & Black Box)
# ==========================================
def test_list_files_not_exists(io_mgr, tmp_path):
    """Black Box: Return empty list if directory does not exist."""
    assert io_mgr.list_files_in_dir(tmp_path / "nonexistent") == []


def test_list_files_with_extension(io_mgr, tmp_path):
    """White Box: Evaluate extension filtering and guaranteed order."""
    (tmp_path / "b.jpg").touch()
    (tmp_path / "c.txt").touch()
    (tmp_path / "a.jpg").touch()

    res = io_mgr.list_files_in_dir(tmp_path, extension=".jpg")
    names = [p.name for p in res]
    assert names == ["a.jpg", "b.jpg"]


def test_list_files_returns_paths(io_mgr, tmp_path):
    """Black Box: Verify output contains Path objects, not strings."""
    (tmp_path / "frame.jpg").touch()
    res = io_mgr.list_files_in_dir(tmp_path, extension=".jpg")
    assert all(isinstance(p, Path) for p in res)


# ==========================================
# load_csv Tests (White & Black Box)
# ==========================================
def test_load_csv_not_exists(io_mgr, tmp_path):
    """Black Box: Non-existent file returns empty list."""
    assert io_mgr.load_csv(tmp_path / "missing.csv") == []


def test_load_csv_valid(io_mgr, tmp_path):
    """White Box: Verify correct parsing of real CSV file."""
    csv_file = tmp_path / "meta.csv"
    csv_file.write_text("clip_id,split\nclip1,val\nclip2,train\n")

    res = io_mgr.load_csv(csv_file)
    assert len(res) == 2
    assert res[0]["clip_id"] == "clip1"
    assert res[0]["split"] == "val"


# ==========================================
# load_image Tests (White & Black Box)
# ==========================================
@patch("src.utils.io_manager.cv2.imread")
def test_load_image_not_found(mock_imread, io_mgr):
    """White Box: cv2.imread failure raises FileNotFoundError."""
    mock_imread.return_value = None
    with pytest.raises(FileNotFoundError):
        io_mgr.load_image("dummy.jpg")


@patch("src.utils.io_manager.cv2.imread")
def test_load_image_valid(mock_imread, io_mgr):
    """Black Box: Correct return of numpy array."""
    dummy_img = np.zeros((10, 10, 3))
    mock_imread.return_value = dummy_img
    res = io_mgr.load_image("dummy.jpg")
    assert np.array_equal(res, dummy_img)


# ==========================================
# save_json Tests (White Box)
# ==========================================
def test_save_json(io_mgr, tmp_path):
    """White Box: Verify JSON file persistence on disk."""
    import json

    data = {"key": "value", "number": 42}
    out_path = tmp_path / "subdir" / "data.json"

    result = io_mgr.save_json(data, out_path)

    assert isinstance(result, Path)
    assert result.exists()
    loaded = json.loads(result.read_text())
    assert loaded == data


# ==========================================
# upload_file_to_drive Tests (White Box)
# ==========================================
def test_upload_drive_without_service(io_mgr):
    """Black Box: Return None when Drive service is disabled."""
    assert io_mgr.upload_file_to_drive("dummy.json", "folder_123") is None


def test_upload_drive_with_service_new_file(io_mgr, tmp_path):
    """White Box: Create a new file when no existing file is found in target folder."""
    mock_service = MagicMock()
    mock_service.files().list().execute.return_value = {"files": []}
    mock_service.files().create().execute.return_value = {"id": "new_drive_id"}
    io_mgr.drive_service = mock_service

    dummy_file = tmp_path / "dummy.json"
    dummy_file.write_text("{}")

    res = io_mgr.upload_file_to_drive(dummy_file, "folder_123")
    assert res == "new_drive_id"


def test_upload_drive_with_service_update_existing_file(io_mgr, tmp_path):
    """White Box: Update (overwrite) existing file when file with same name exists in folder."""
    mock_service = MagicMock()
    mock_service.files().list().execute.return_value = {"files": [{"id": "existing_id", "name": "dummy.json"}]}
    mock_service.files().update().execute.return_value = {"id": "existing_id"}
    io_mgr.drive_service = mock_service

    dummy_file = tmp_path / "dummy.json"
    dummy_file.write_text("{}")

    res = io_mgr.upload_file_to_drive(dummy_file, "folder_123")
    assert res == "existing_id"


# ==========================================
# download_file_from_drive Tests (White Box)
# ==========================================
def test_download_drive_without_service(io_mgr):
    """Black Box: Return None when Drive service is disabled."""
    assert (
        io_mgr.download_file_from_drive("last.pt", "folder_123", "local_last.pt")
        is None
    )


def test_download_drive_not_found(io_mgr, tmp_path):
    """White Box: Return None when file is not found in Drive search."""
    mock_service = MagicMock()
    mock_service.files().list().execute.return_value = {"files": []}
    io_mgr.drive_service = mock_service

    res = io_mgr.download_file_from_drive("last.pt", "folder_123", tmp_path / "last.pt")
    assert res is None
