"""
Unit tests for src/batch.py pipeline.
"""

import json
import zipfile
from pathlib import Path
from PIL import Image

from src.batch import run_batch, zip_outputs, MockCatalogEditor


def test_batch_processing_dummy_images(test_input_dir, test_output_dir):
    """Verifies that run_batch processes images, creates masks, outputs, and manifest.json."""
    manifest = run_batch(
        input_dir=str(test_input_dir),
        output_dir=str(test_output_dir),
        part_text="chair legs",
        spec_prompt="walnut wood chair legs",
        mock=True,
    )

    assert len(manifest) == 3
    manifest_file = test_output_dir / "manifest.json"
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 3

    for item in data:
        assert item["status"] == "success"
        assert item["output_path"] is not None
        assert Path(item["output_path"]).exists()
        mask_file = test_output_dir / "masks" / f"{Path(item['name']).stem}_mask.png"
        assert mask_file.exists()
        assert item["mask_coverage_pct"] > 0


def test_batch_empty_mask_no_part_found(test_input_dir, test_output_dir):
    """Verifies that when mask coverage is below cutoff, status is set to 'no_part_found'."""
    # When part_text is "empty", MockCatalogEditor returns an all-black mask (0.0% coverage)
    manifest = run_batch(
        input_dir=str(test_input_dir),
        output_dir=str(test_output_dir),
        part_text="empty",
        spec_prompt="walnut wood chair legs",
        mock=True,
    )

    assert len(manifest) == 3
    for item in manifest:
        assert item["status"] == "no_part_found"
        assert item["output_path"] is None
        assert item["mask_coverage_pct"] == 0.0


def test_batch_per_image_exception_recovery(test_input_dir, test_output_dir):
    """Verifies that if an exception occurs during inpainting for one image, others still complete."""
    class FlakyEditor(MockCatalogEditor):
        def inpaint(self, image, mask, prompt, **kwargs):
            # Fail on chair_2
            if getattr(image, "filename", None) and "chair_2" in image.filename:
                raise RuntimeError("Simulated inpainting hardware fault")
            return super().inpaint(image, mask, prompt, **kwargs)

    # To pass filename through Image.open, we monkeypatch or wrap inpaint
    editor = MockCatalogEditor(device="cpu")
    original_inpaint = editor.inpaint

    call_count = 0
    def flaky_inpaint(image, mask, prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Simulated failure on image 2")
        return original_inpaint(image, mask, prompt, **kwargs)

    editor.inpaint = flaky_inpaint

    manifest = run_batch(
        input_dir=str(test_input_dir),
        output_dir=str(test_output_dir),
        part_text="chair legs",
        spec_prompt="walnut wood chair legs",
        params={"editor": editor},
    )

    assert len(manifest) == 3
    assert manifest[0]["status"] == "success"
    assert "error: Simulated failure on image 2" in manifest[1]["status"]
    assert manifest[1]["output_path"] is None
    assert manifest[2]["status"] == "success"


def test_batch_retry_with_lower_threshold(test_input_dir, test_output_dir):
    """Verifies that low initial mask triggers retry with lower threshold and succeeds."""
    class RetryEditor(MockCatalogEditor):
        def generate_mask(self, image, part_text, threshold=0.4, dilate_px=12):
            w, h = image.size
            mask = Image.new("L", (w, h), 0)
            # If standard threshold, make a tiny dot (<0.5% coverage on 256x256)
            if threshold > 0.25:
                # 256*256 = 65536. 0.5% is 327 pixels. 10x10 is 100 pixels (<0.5%)
                for x in range(10):
                    for y in range(10):
                        mask.putpixel((x, y), 255)
            else:
                # Retry threshold: make full rectangle (>0.5%)
                for x in range(50):
                    for y in range(50):
                        mask.putpixel((x, y), 255)
            return mask

    editor = RetryEditor(device="cpu")
    manifest = run_batch(
        input_dir=str(test_input_dir),
        output_dir=str(test_output_dir),
        part_text="chair legs",
        spec_prompt="walnut wood chair legs",
        params={"editor": editor, "retry_larger_mask": True},
    )

    assert len(manifest) == 3
    for item in manifest:
        assert item["status"] == "success (retried_mask)"
        assert item["output_path"] is not None


def test_zip_outputs(test_input_dir, test_output_dir):
    """Verifies that zip_outputs packages the results directory into a valid .zip archive."""
    run_batch(
        input_dir=str(test_input_dir),
        output_dir=str(test_output_dir),
        part_text="chair legs",
        spec_prompt="walnut wood chair legs",
        mock=True,
    )

    zip_path_str = zip_outputs(str(test_output_dir))
    zip_path = Path(zip_path_str)
    assert zip_path.exists()
    assert zip_path.suffix == ".zip"

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "manifest.json" in namelist
        assert any(n.endswith("_edited.png") for n in namelist)
        assert any("masks" in n for n in namelist)
