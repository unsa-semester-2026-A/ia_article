import pytest
from src.data_preparation.parser import convert_obb_to_corners, parse_csv


def test_convert_obb_to_corners() -> None:
    """Verifies that OBB corner normalization maps inside [0.0, 1.0]."""
    cx, cy, w, h, angle_deg = 960.0, 540.0, 100.0, 50.0, 30.0
    W, H = 1920, 1080

    corners = convert_obb_to_corners(cx, cy, w, h, angle_deg, W, H)

    assert len(corners) == 8
    for coordinate in corners:
        assert 0.0 <= coordinate <= 1.0


def test_parse_csv_file_not_found() -> None:
    """Verifies FileNotFoundError is raised when annotations file does not exist."""
    non_existent_path = "fictional/path/to/missing_file.csv"

    with pytest.raises(FileNotFoundError):
        parse_csv(non_existent_path)
