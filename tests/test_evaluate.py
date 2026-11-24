"""
Unit tests for src/evaluate.py SSIM metrics and batch evaluation.
"""

import csv
from PIL import Image
import numpy as np

from src.evaluate import ssim_outside_mask, evaluate_run
from src.batch import run_batch


def test_ssim_outside_mask_identical():
    """Verifies that ssim_outside_mask returns exactly 1.0 when unmasked regions are identical."""
    np.random.seed(42)
    orig_arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    orig_img = Image.fromarray(orig_arr, mode="RGB")

    # Inpainted image differs inside the center box
    edit_arr = orig_arr.copy()
    edit_arr[20:44, 20:44] = np.random.randint(0, 256, (24, 24, 3), dtype=np.uint8)
    edit_img = Image.fromarray(edit_arr, mode="RGB")

    # Mask covers the edited center box
    mask_arr = np.zeros((64, 64), dtype=np.uint8)
    mask_arr[20:44, 20:44] = 255
    mask_img = Image.fromarray(mask_arr, mode="L")

    # Outside the mask, orig_img and edit_img are byte-identical
    score = ssim_outside_mask(orig_img, edit_img, mask_img)
    assert score == 1.0


def test_ssim_outside_mask_differing():
    """Verifies that ssim_outside_mask returns < 1.0 when unmasked regions differ."""
    orig_arr = np.full((64, 64, 3), 120, dtype=np.uint8)
    orig_img = Image.fromarray(orig_arr, mode="RGB")

    # Inpainted image modified outside the mask
    edit_arr = orig_arr.copy()
    edit_arr[0:5, 0:5] = 200  # Corruption in top-left unmasked corner
    edit_img = Image.fromarray(edit_arr, mode="RGB")

    # Mask covers only center
    mask_arr = np.zeros((64, 64), dtype=np.uint8)
    mask_arr[20:44, 20:44] = 255
    mask_img = Image.fromarray(mask_arr, mode="L")

    score = ssim_outside_mask(orig_img, edit_img, mask_img)
    assert score < 1.0


def test_ssim_outside_mask_full_mask():
    """Verifies edge case where mask covers 100% of the image."""
    orig_img = Image.new("RGB", (32, 32), (10, 10, 10))
    edit_img = Image.new("RGB", (32, 32), (200, 200, 200))
    full_mask = Image.new("L", (32, 32), 255)

    assert ssim_outside_mask(orig_img, edit_img, full_mask) == 1.0


def test_evaluate_run_writes_csv(test_input_dir, test_output_dir):
    """Verifies that evaluate_run reads manifest.json and writes evaluation.csv with proper fields."""
    run_batch(
        input_dir=str(test_input_dir),
        output_dir=str(test_output_dir),
        part_text="chair legs",
        spec_prompt="walnut wood chair legs",
        mock=True,
    )

    manifest_path = test_output_dir / "manifest.json"
    records = evaluate_run(
        manifest_path=str(manifest_path),
        input_dir=str(test_input_dir),
        output_dir=str(test_output_dir),
    )

    assert len(records) == 3
    csv_file = test_output_dir / "evaluation.csv"
    assert csv_file.exists()

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 3
    for row in reader:
        assert row["status"] == "success"
        assert float(row["outside_ssim"]) == 1.0
        assert float(row["mask_coverage_pct"]) > 0
