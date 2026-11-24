"""
Unit tests for src/engine.py logic and paste-back compositing.
"""

from PIL import Image
import numpy as np

from src.engine import paste_back, CatalogEditor


def test_paste_back_byte_identical():
    """
    Verifies that paste_back leaves pixels outside mask strictly byte-identical to original.
    """
    np.random.seed(123)
    # Generate random 128x128 RGB image
    orig_arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    orig_img = Image.fromarray(orig_arr, mode="RGB")

    # Inpainted replacement (solid gray)
    fake_inpaint_img = Image.fromarray(np.full((128, 128, 3), 100, dtype=np.uint8), mode="RGB")

    # Mask: 40x40 square in center
    mask_arr = np.zeros((128, 128), dtype=np.uint8)
    mask_arr[44:84, 44:84] = 255
    mask_img = Image.fromarray(mask_arr, mode="L")

    # Composite using paste_back
    final_img = paste_back(orig_img, fake_inpaint_img, mask_img)
    final_arr = np.array(final_img)

    # Pixel difference strictly outside mask
    diff_outside = np.abs(
        final_arr[mask_arr == 0].astype(int) - orig_arr[mask_arr == 0].astype(int)
    )

    assert np.max(diff_outside) == 0
    assert np.array_equal(final_arr[mask_arr == 0], orig_arr[mask_arr == 0])

    # Pixels inside mask should be the inpainted image
    assert np.array_equal(final_arr[mask_arr == 255], np.array(fake_inpaint_img)[mask_arr == 255])


def test_catalog_editor_lazy_loading():
    """
    Verifies that initializing CatalogEditor does not load models or download weights.
    """
    editor = CatalogEditor(device="cpu")
    assert editor.device == "cpu"
    assert editor._clipseg_processor is None
    assert editor._clipseg_model is None
    assert editor._inpaint_pipe is None
