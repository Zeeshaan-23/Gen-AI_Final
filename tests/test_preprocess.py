"""
Tests for the image preprocessing stage:
- RGBA transparency compositing onto solid white.
- Aspect ratio preserving square padding.
- Fault-tolerant corrupted/unreadable file handling in directory batch.
- preprocess_dir report schema and CSV output.
- Near-uniform light background normalization.
- In-memory preprocessing integration with run_batch.
"""

import csv
import json
import numpy as np
from PIL import Image, ImageDraw

from src.preprocess import (
    load_and_clean,
    normalize_background,
    pad_to_square,
    resize_for_model,
    preprocess_dir,
)
from src.batch import run_batch


def test_rgba_to_white_composite(tmp_path):
    """Verifies transparent pixels in RGBA images are composited onto pure white."""
    rgba = Image.new("RGBA", (50, 50), (0, 0, 0, 0))  # 100% transparent
    # Draw an opaque colored box in the center
    draw = ImageDraw.Draw(rgba)
    draw.rectangle([15, 15, 35, 35], fill=(30, 140, 200, 255))

    img_path = tmp_path / "transparent_sample.png"
    rgba.save(img_path)

    cleaned = load_and_clean(img_path)

    assert cleaned.mode == "RGB"
    assert cleaned.size == (50, 50)
    # Corner (formerly transparent) must be pure white (255, 255, 255)
    assert cleaned.getpixel((0, 0)) == (255, 255, 255)
    # Center (opaque) must retain its color
    assert cleaned.getpixel((25, 25)) == (30, 140, 200)


def test_padding_is_square():
    """Verifies that rectangular images are padded to square preserving aspect ratio."""
    rect = Image.new("RGB", (320, 160), (200, 100, 50))
    square = pad_to_square(rect)

    assert square.size == (320, 320)
    # Center must retain original image color
    assert square.getpixel((160, 160)) == (200, 100, 50)
    # Top and bottom padding areas must be white
    assert square.getpixel((160, 10)) == (255, 255, 255)
    assert square.getpixel((160, 310)) == (255, 255, 255)

    # Verify resize_for_model
    model_img = resize_for_model(rect, size=512)
    assert model_img.size == (512, 512)


def test_corrupt_file_is_skipped(tmp_path):
    """Verifies that corrupt or unreadable files are logged and skipped rather than crashing."""
    in_dir = tmp_path / "inputs"
    out_dir = tmp_path / "outputs"
    in_dir.mkdir()

    # Create a valid image
    valid_img = Image.new("RGB", (100, 100), (255, 255, 255))
    valid_img.save(in_dir / "good.jpg")

    # Create a corrupted file with an image extension
    corrupt_file = in_dir / "broken.jpg"
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_JPEG_HEADER_RANDOM_CORRUPT_BYTES_1234567890")

    report = preprocess_dir(in_dir, out_dir)

    assert len(report) == 2
    statuses = {r["name"]: r["status"] for r in report}

    assert statuses["good.jpg"] == "success"
    assert statuses["broken.jpg"].startswith("corrupt_skipped")

    # Output directory must contain good.png and NOT crash on broken.jpg
    assert (out_dir / "good.png").exists()
    assert not (out_dir / "broken.png").exists()


def test_preprocess_dir_report_rows(tmp_path):
    """Verifies that preprocess_dir produces valid report dicts and preprocess_report.csv."""
    in_dir = tmp_path / "catalog_in"
    out_dir = tmp_path / "catalog_out"
    in_dir.mkdir()

    # Create 2 test images
    img1 = Image.new("RGB", (800, 600), (245, 245, 245))  # Off-white background
    img1.save(in_dir / "chair1.jpg")

    img2 = Image.new("RGB", (400, 400), (40, 40, 40))      # Dark background
    img2.save(in_dir / "chair2.png")

    report = preprocess_dir(in_dir, out_dir, size=500)

    assert len(report) == 2
    for r in report:
        assert "name" in r
        assert "orig_size" in r
        assert "new_size" in r
        assert "bg_normalized" in r
        assert "status" in r
        assert r["status"] == "success"

    # chair1 was 800x600, scaled down to max 500 -> 500x375
    c1 = next(r for r in report if r["name"] == "chair1.jpg")
    assert c1["orig_size"] == (800, 600)
    assert c1["new_size"] == (500, 375)
    assert c1["bg_normalized"] is True

    # Verify CSV file
    csv_file = out_dir / "preprocess_report.csv"
    assert csv_file.exists()

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["status"] == "success"


def test_normalize_background():
    """Verifies background normalization on light vs dark vs non-uniform borders."""
    # 1. Off-white uniform background -> normalized to pure white
    offwhite = Image.new("RGB", (100, 100), (242, 242, 240))
    draw = ImageDraw.Draw(offwhite)
    draw.rectangle([30, 30, 70, 70], fill=(60, 30, 20))
    res1 = normalize_background(offwhite, tol=18)

    assert res1.getpixel((0, 0)) == (255, 255, 255)
    assert res1.getpixel((50, 50)) == (60, 30, 20)

    # 2. Dark background -> remains unchanged
    dark = Image.new("RGB", (100, 100), (50, 50, 50))
    res2 = normalize_background(dark, tol=18)
    assert np.array_equal(np.array(dark), np.array(res2))

    # 3. Non-uniform border -> remains unchanged
    var = Image.new("RGB", (100, 100), (255, 255, 255))
    ImageDraw.Draw(var).line([(0, 0), (99, 0)], fill=(0, 0, 0), width=6)
    res3 = normalize_background(var, tol=18)
    assert np.array_equal(np.array(var), np.array(res3))


def test_batch_integration_preprocess(tmp_path):
    """Verifies batch run records orig_size and bg_normalized in manifest.json."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    sample = Image.new("RGB", (120, 80), (245, 245, 245))
    sample.save(in_dir / "test_item.jpg")

    manifest = run_batch(
        input_dir=in_dir,
        output_dir=out_dir,
        part_text="chair legs",
        spec_prompt="walnut legs",
        mock=True,
    )

    assert len(manifest) == 1
    rec = manifest[0]
    assert "orig_size" in rec
    assert rec["orig_size"] == [120, 80]
    assert "bg_normalized" in rec
    assert rec["bg_normalized"] is True

    # Check manifest.json on disk
    with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
        disk_manifest = json.load(f)
        assert disk_manifest[0]["orig_size"] == [120, 80]
        assert disk_manifest[0]["bg_normalized"] is True
