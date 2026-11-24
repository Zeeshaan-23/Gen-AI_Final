"""
Pytest configuration and shared fixtures for catalogupdate-ai test suite.
"""

import pytest
from PIL import Image, ImageDraw

from src.batch import MockCatalogEditor


@pytest.fixture
def mock_editor():
    """Provides a lightweight CPU mock editor for testing."""
    return MockCatalogEditor(device="cpu")


@pytest.fixture
def test_input_dir(tmp_path):
    """Creates a temporary input directory with 3 synthetic product images."""
    in_dir = tmp_path / "input"
    in_dir.mkdir(parents=True, exist_ok=True)

    colors = [
        ((230, 230, 230), (50, 50, 60), (180, 180, 180)),
        ((240, 240, 240), (80, 40, 40), (140, 140, 140)),
        ((250, 250, 250), (40, 70, 50), (160, 160, 160)),
    ]

    for idx, (bg, seat, leg) in enumerate(colors):
        img = Image.new("RGB", (256, 256), bg)
        draw = ImageDraw.Draw(img)
        # Seat/cushion
        draw.rounded_rectangle([70, 50, 186, 150], radius=8, fill=seat)
        # Legs
        draw.rectangle([80, 150, 95, 220], fill=leg)
        draw.rectangle([161, 150, 176, 220], fill=leg)
        img.save(in_dir / f"chair_{idx + 1}.png")

    return in_dir


@pytest.fixture
def test_output_dir(tmp_path):
    """Creates an empty temporary output directory."""
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
