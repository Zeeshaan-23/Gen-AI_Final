"""
Unit tests for ControlNet structure preservation, make_control_image,
manifest fields, CLI flags, and compare grid creation.
"""

import sys
import json
import csv
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

from src.engine import make_control_image
from src.batch import run_batch, MockCatalogEditor
from src.compare import compare_controlnet


def test_make_control_image_shape_and_dtype():
    """
    Verifies that make_control_image returns a 3-channel RGB PIL Image
    of exact dimensions 512x512 with uint8 array representations across
    various input aspect ratios (square, landscape, portrait).
    """
    # 1. Square image
    square_img = Image.new("RGB", (256, 256), (200, 100, 50))
    ctrl_sq = make_control_image(square_img)
    assert ctrl_sq.size == (512, 512)
    assert ctrl_sq.mode == "RGB"
    arr_sq = np.array(ctrl_sq)
    assert arr_sq.shape == (512, 512, 3)
    assert arr_sq.dtype == np.uint8

    # 2. Landscape image (e.g. 640x360)
    land_img = Image.new("RGB", (640, 360), (50, 120, 220))
    ctrl_land = make_control_image(land_img, low_threshold=80, high_threshold=160)
    assert ctrl_land.size == (512, 512)
    assert ctrl_land.mode == "RGB"
    arr_land = np.array(ctrl_land)
    assert arr_land.shape == (512, 512, 3)
    assert arr_land.dtype == np.uint8

    # 3. Portrait image (e.g. 300x500)
    port_img = Image.new("RGB", (300, 500), (30, 30, 30))
    ctrl_port = make_control_image(port_img)
    assert ctrl_port.size == (512, 512)
    assert ctrl_port.mode == "RGB"


def test_edit_returns_control_image():
    """
    Verifies that edit() returns a dictionary containing 'mask', 'result', and 'control'.
    """
    mock_editor = MockCatalogEditor(device="cpu")
    img = Image.new("RGB", (200, 200), (240, 240, 240))
    out = mock_editor.edit(
        image=img,
        part_text="chair legs",
        new_spec_prompt="walnut wood legs",
        use_controlnet=True,
        controlnet_scale=0.85,
    )

    assert "mask" in out
    assert "result" in out
    assert "control" in out
    assert out["control"].size == (512, 512)
    assert out["control"].mode == "RGB"


def test_batch_manifest_controlnet_fields(test_input_dir, test_output_dir):
    """
    Verifies that run_batch correctly records use_controlnet and controlnet_scale
    in manifest.json and outputs control edge maps when use_controlnet=True.
    """
    params = {
        "use_controlnet": True,
        "controlnet_scale": 0.9,
        "mock": True,
    }

    manifest = run_batch(
        input_dir=test_input_dir,
        output_dir=test_output_dir,
        part_text="chair legs",
        spec_prompt="walnut legs",
        params=params,
    )

    assert len(manifest) > 0
    for record in manifest:
        assert "use_controlnet" in record
        assert "controlnet_scale" in record
        assert record["use_controlnet"] is True
        assert record["controlnet_scale"] == 0.9

    # Verify manifest.json on disk
    manifest_file = test_output_dir / "manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == len(manifest)
    assert data[0]["use_controlnet"] is True
    assert data[0]["controlnet_scale"] == 0.9

    # Verify control image output folder
    control_dir = test_output_dir / "control"
    assert control_dir.exists()
    ctrl_files = list(control_dir.glob("*.png"))
    assert len(ctrl_files) == len(manifest)


def test_batch_cli_controlnet_flags(test_input_dir, tmp_path):
    """
    Tests invoking python -m src.batch with --controlnet and --controlnet-scale flags.
    """
    out_dir = tmp_path / "cli_out"
    cmd = [
        sys.executable,
        "-m",
        "src.batch",
        "--input",
        str(test_input_dir),
        "--output",
        str(out_dir),
        "--part",
        "chair legs",
        "--prompt",
        "walnut legs",
        "--controlnet",
        "--controlnet-scale",
        "0.75",
        "--mock",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI error: {res.stderr}"

    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) > 0
    assert data[0]["use_controlnet"] is True
    assert data[0]["controlnet_scale"] == 0.75


def test_compare_controlnet_pipeline(test_input_dir, tmp_path):
    """
    Verifies that compare_controlnet generates the 4-column comparison_grid.png
    and comparison_metrics.csv with all expected columns.
    """
    out_dir = tmp_path / "compare_out"

    results = compare_controlnet(
        input_dir=test_input_dir,
        output_dir=out_dir,
        part_text="chair legs",
        spec_prompt="solid walnut wood",
        params={"controlnet_scale": 0.85},
        mock=True,
    )

    grid_path = Path(results["grid_path"])
    csv_path = Path(results["csv_path"])

    assert grid_path.exists()
    assert csv_path.exists()

    # Verify grid image properties: 4 columns of cell_size=512 -> 2048 width
    with Image.open(grid_path) as grid_img:
        assert grid_img.width == 4 * 512
        assert grid_img.height > 512

    # Verify CSV metrics contents
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0
    expected_fields = {
        "image_name",
        "edge_ssim_without_controlnet",
        "edge_ssim_with_controlnet",
        "structure_gain",
        "controlnet_scale",
        "time_s",
    }
    assert expected_fields.issubset(set(rows[0].keys()))
    assert float(rows[0]["controlnet_scale"]) == 0.85


def test_compare_cli(test_input_dir, tmp_path):
    """
    Tests invoking python -m src.compare CLI with --mock.
    """
    out_dir = tmp_path / "compare_cli_out"
    cmd = [
        sys.executable,
        "-m",
        "src.compare",
        "--input",
        str(test_input_dir),
        "--output",
        str(out_dir),
        "--part",
        "chair legs",
        "--prompt",
        "walnut legs",
        "--controlnet-scale",
        "0.8",
        "--mock",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Compare CLI error: {res.stderr}"

    assert (out_dir / "comparison_grid.png").exists()
    assert (out_dir / "comparison_metrics.csv").exists()
