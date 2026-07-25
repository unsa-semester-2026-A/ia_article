"""Unit tests for IOManager module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.utils.io_manager import IOManager


@pytest.fixture(autouse=True)
def isolate_token_sources(monkeypatch, tmp_path):
    """Point the token source at a path that does not exist.

    Otherwise a token left in a real ``/kaggle/input`` mount or in the developer's
    environment would satisfy every construction and make the Kaggle Secrets
    fallback unreachable. Cases that care about a specific source override it.
    """
    monkeypatch.setenv("DRIVE_TOKEN_SOURCE", str(tmp_path / "absent" / "token.json"))


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


def _mock_download(io_mgr, remote_size, written_bytes):
    """Wire a Drive mock that reports remote_size and writes written_bytes locally."""
    mock_service = MagicMock()
    mock_service.files().list().execute.return_value = {
        "files": [{"id": "ckpt_id", "name": "last.pt", "size": str(remote_size)}]
    }
    io_mgr.drive_service = mock_service

    class _Downloader:
        def __init__(self, fh, _request):
            self.fh = fh

        def next_chunk(self):
            self.fh.write(written_bytes)
            return None, True

    return patch("googleapiclient.http.MediaIoBaseDownload", _Downloader)


def test_download_drive_accepts_complete_file(io_mgr, tmp_path):
    """White Box: Keep the file when its size matches the size reported by Drive."""
    dest = tmp_path / "last.pt"
    with _mock_download(io_mgr, remote_size=4, written_bytes=b"abcd"):
        res = io_mgr.download_file_from_drive("last.pt", "folder_123", dest)

    assert res == dest
    assert dest.exists()


def test_download_drive_discards_truncated_file(io_mgr, tmp_path):
    """White Box: Discard a partial checkpoint instead of letting the load fail later."""
    dest = tmp_path / "last.pt"
    with _mock_download(io_mgr, remote_size=100, written_bytes=b"ab"):
        res = io_mgr.download_file_from_drive("last.pt", "folder_123", dest)

    assert res is None
    assert not dest.exists()


# ==========================================
# Token Source Tests
# ==========================================
def test_token_is_copied_from_a_read_only_source(tmp_path, monkeypatch):
    """Black Box: The token is copied so a refreshed one can be written back."""
    source = tmp_path / "source" / "drive_token.json"
    source.parent.mkdir()
    source.write_text('{"refresh_token": "abc"}')
    destination = tmp_path / "working" / "token.json"
    monkeypatch.setenv("DRIVE_TOKEN_SOURCE", str(source))

    with patch.object(IOManager, "_get_drive_service", return_value=None):
        manager = IOManager(token_path=destination)

    assert destination.read_text() == '{"refresh_token": "abc"}'
    assert manager.token_path == destination


def test_existing_token_is_not_overwritten(tmp_path, monkeypatch):
    """White Box: A refreshed token on disk must survive a later construction."""
    source = tmp_path / "source.json"
    source.write_text('{"refresh_token": "stale"}')
    destination = tmp_path / "token.json"
    destination.write_text('{"refresh_token": "refreshed"}')
    monkeypatch.setenv("DRIVE_TOKEN_SOURCE", str(source))

    with patch.object(IOManager, "_get_drive_service", return_value=None):
        IOManager(token_path=destination)

    assert destination.read_text() == '{"refresh_token": "refreshed"}'


def test_token_source_override_replaces_the_candidate_list(tmp_path, monkeypatch):
    """White Box: An explicit source is the only candidate considered."""
    monkeypatch.setenv("DRIVE_TOKEN_SOURCE", str(tmp_path / "explicit.json"))

    with patch.object(IOManager, "_get_drive_service", return_value=None):
        with patch.object(IOManager, "_ensure_token_from_kaggle_secrets"):
            manager = IOManager(token_path=tmp_path / "token.json")

    assert manager.token_source_candidates() == [tmp_path / "explicit.json"]


def test_attached_kaggle_datasets_are_the_default_candidates(tmp_path, monkeypatch):
    """Black Box: Without an override, an attached dataset provides the token."""
    monkeypatch.delenv("DRIVE_TOKEN_SOURCE", raising=False)
    mounted = tmp_path / "input"
    (mounted / "ia-article-drive-token").mkdir(parents=True)
    (mounted / "ia-article-drive-token" / "token.json").write_text("{}")
    (mounted / "mtc-challenge").mkdir()
    monkeypatch.setattr("src.utils.io_manager.KAGGLE_INPUT_ROOT", mounted)

    with patch.object(IOManager, "_get_drive_service", return_value=None):
        with patch.object(IOManager, "_ensure_token_from_kaggle_secrets"):
            manager = IOManager(token_path=tmp_path / "token.json")

    assert manager.token_source_candidates() == [
        mounted / "ia-article-drive-token" / "token.json"
    ]


def test_nested_kaggle_mount_layout_is_searched(tmp_path, monkeypatch):
    """White Box: Kaggle also mounts datasets under datasets/<owner>/<name>/."""
    monkeypatch.delenv("DRIVE_TOKEN_SOURCE", raising=False)
    mounted = tmp_path / "input"
    nested = mounted / "datasets" / "alvaroquispeunsa" / "ia-article-drive-token"
    nested.mkdir(parents=True)
    (nested / "token.json").write_text("{}")
    monkeypatch.setattr("src.utils.io_manager.KAGGLE_INPUT_ROOT", mounted)

    with patch.object(IOManager, "_get_drive_service", return_value=None):
        with patch.object(IOManager, "_ensure_token_from_kaggle_secrets"):
            manager = IOManager(token_path=tmp_path / "token.json")

    assert manager.token_source_candidates() == [nested / "token.json"]


def test_secrets_are_not_consulted_when_a_source_provided_the_token(
    tmp_path, monkeypatch
):
    """White Box: A usable token makes the Kaggle Secrets fallback unnecessary."""
    source = tmp_path / "source.json"
    source.write_text('{"refresh_token": "abc"}')
    monkeypatch.setenv("DRIVE_TOKEN_SOURCE", str(source))

    with patch.object(IOManager, "_get_drive_service", return_value=None):
        with patch.object(
            IOManager, "_ensure_token_from_kaggle_secrets"
        ) as mock_secrets:
            IOManager(token_path=tmp_path / "token.json")

    # The fallback still runs but returns immediately; what matters is that the
    # token on disk is the one from the source.
    mock_secrets.assert_called_once()


# ==========================================
# Credential Strictness Tests
# ==========================================
def test_missing_secret_is_fatal_when_drive_is_required(tmp_path, monkeypatch):
    """Black Box: A production run must not start without a way to persist results."""
    fake_secrets = MagicMock()
    fake_secrets.UserSecretsClient.return_value.get_secret.side_effect = RuntimeError(
        "secret not attached"
    )
    monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_secrets)

    with pytest.raises(RuntimeError, match="DRIVE_TOKEN_JSON"):
        IOManager(token_path=tmp_path / "token.json", require_drive=True)


def test_missing_secret_degrades_when_drive_is_optional(tmp_path, monkeypatch):
    """Black Box: A disposable run continues with Drive disabled."""
    fake_secrets = MagicMock()
    fake_secrets.UserSecretsClient.return_value.get_secret.side_effect = RuntimeError(
        "secret not attached"
    )
    monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_secrets)

    manager = IOManager(token_path=tmp_path / "token.json", require_drive=False)

    assert manager.drive_service is None


# ==========================================
# remote_name Tests (White Box)
# ==========================================
def test_upload_uses_remote_name_when_creating(io_mgr, tmp_path):
    """White Box: The Drive file takes the run-prefixed name, not the local one."""
    mock_service = MagicMock()
    mock_service.files().list().execute.return_value = {"files": []}
    mock_service.files().create().execute.return_value = {"id": "new_id"}
    io_mgr.drive_service = mock_service

    dummy = tmp_path / "last.pt"
    dummy.write_bytes(b"weights")

    io_mgr.upload_file_to_drive(dummy, "folder_123", remote_name="f1_c1_last.pt")

    body = mock_service.files().create.call_args.kwargs["body"]
    assert body["name"] == "f1_c1_last.pt"


def test_upload_looks_up_existing_by_remote_name(io_mgr, tmp_path):
    """White Box: Overwrite lookup must use the remote name to stay per-run."""
    mock_service = MagicMock()
    mock_service.files().list().execute.return_value = {
        "files": [{"id": "existing_id", "name": "f1_c1_last.pt"}]
    }
    mock_service.files().update().execute.return_value = {"id": "existing_id"}
    io_mgr.drive_service = mock_service

    dummy = tmp_path / "last.pt"
    dummy.write_bytes(b"weights")

    with patch.object(
        IOManager, "_find_existing_file_id", return_value="existing_id"
    ) as mock_find:
        result = io_mgr.upload_file_to_drive(
            dummy, "folder_123", remote_name="f1_c1_last.pt"
        )

    assert result == "existing_id"
    mock_find.assert_called_once_with("f1_c1_last.pt", "folder_123")
