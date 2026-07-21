import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from experiments.src.utils.io_manager import IOManager


@pytest.fixture
def io_mgr(tmp_path):
    """Instancia de prueba de IOManager evitando descargas automáticas durante tests."""
    with patch.object(IOManager, "_ensure_token_exists"):
        with patch.object(IOManager, "_get_drive_service", return_value=None):
            return IOManager(token_path=tmp_path / "dummy_token.json")


# ==========================================
# Pruebas _ensure_token_exists (Caja Blanca)
# ==========================================
@patch("urllib.request.urlretrieve")
def test_ensure_token_exists_downloads_when_missing(mock_urlretrieve, tmp_path):
    """Caja Blanca: Si token.json no existe, IOManager lo descarga automáticamente."""
    token_file = tmp_path / "token.json"
    assert not token_file.exists()

    with patch.object(IOManager, "_get_drive_service", return_value=None):
        mgr = IOManager(token_path=token_file, drive_token_file_id="fake_id_123")

    mock_urlretrieve.assert_called_once_with(
        "https://drive.google.com/uc?export=download&id=fake_id_123",
        str(token_file),
    )


# ==========================================
# Pruebas list_files_in_dir (Caja Blanca/Negra)
# ==========================================
def test_list_files_not_exists(io_mgr, tmp_path):
    """Caja Negra: Debe retornar lista vacía si el directorio no existe."""
    assert io_mgr.list_files_in_dir(tmp_path / "nonexistent") == []


def test_list_files_with_extension(io_mgr, tmp_path):
    """Caja Blanca: Evalúa filtrado por extensión y orden garantizado."""
    (tmp_path / "b.jpg").touch()
    (tmp_path / "c.txt").touch()
    (tmp_path / "a.jpg").touch()

    res = io_mgr.list_files_in_dir(tmp_path, extension=".jpg")
    names = [p.name for p in res]
    assert names == ["a.jpg", "b.jpg"]


def test_list_files_returns_paths(io_mgr, tmp_path):
    """Caja Negra: Verifica que retorne objetos Path, no strings."""
    (tmp_path / "frame.jpg").touch()
    res = io_mgr.list_files_in_dir(tmp_path, extension=".jpg")
    assert all(isinstance(p, Path) for p in res)


# ==========================================
# Pruebas load_csv (Caja Blanca/Negra)
# ==========================================
def test_load_csv_not_exists(io_mgr, tmp_path):
    """Caja Negra: Fichero inexistente retorna lista vacía."""
    assert io_mgr.load_csv(tmp_path / "missing.csv") == []


def test_load_csv_valid(io_mgr, tmp_path):
    """Caja Blanca: Verifica lectura correcta de un CSV real en disco."""
    csv_file = tmp_path / "meta.csv"
    csv_file.write_text("clip_id,split\nclip1,val\nclip2,train\n")

    res = io_mgr.load_csv(csv_file)
    assert len(res) == 2
    assert res[0]["clip_id"] == "clip1"
    assert res[0]["split"] == "val"


# ==========================================
# Pruebas load_image (Caja Blanca/Negra)
# ==========================================
@patch("experiments.src.utils.io_manager.cv2.imread")
def test_load_image_not_found(mock_imread, io_mgr):
    """Caja Blanca: cv2.imread falla -> dispara FileNotFoundError."""
    mock_imread.return_value = None
    with pytest.raises(FileNotFoundError):
        io_mgr.load_image("dummy.jpg")


@patch("experiments.src.utils.io_manager.cv2.imread")
def test_load_image_valid(mock_imread, io_mgr):
    """Caja Negra: Retorno correcto de una matriz numpy."""
    dummy_img = np.zeros((10, 10, 3))
    mock_imread.return_value = dummy_img
    res = io_mgr.load_image("dummy.jpg")
    assert np.array_equal(res, dummy_img)


# ==========================================
# Pruebas save_json (Caja Blanca)
# ==========================================
def test_save_json(io_mgr, tmp_path):
    """Caja Blanca: Verifica persistencia real en disco y retorno de Path."""
    import json

    data = {"key": "value", "number": 42}
    out_path = tmp_path / "subdir" / "data.json"

    result = io_mgr.save_json(data, out_path)

    assert isinstance(result, Path)
    assert result.exists()
    loaded = json.loads(result.read_text())
    assert loaded == data


# ==========================================
# Pruebas upload_file_to_drive (Caja Blanca)
# ==========================================
def test_upload_drive_without_service(io_mgr):
    """Caja Negra: Sin servicio Drive inicializado, retorna None."""
    assert io_mgr.upload_file_to_drive("dummy.json", "folder_123") is None


def test_upload_drive_with_service(io_mgr):
    """Caja Blanca: Evalúa la llamada correcta a la API de Google inyectando un servicio Mock."""
    mock_service = MagicMock()
    mock_service.files().create().execute.return_value = {"id": "fake_drive_id"}
    io_mgr.drive_service = mock_service

    res = io_mgr.upload_file_to_drive("dummy.json", "folder_123")
    assert res == "fake_drive_id"
