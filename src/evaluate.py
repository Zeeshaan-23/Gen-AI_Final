"""
CatalogUpdate AI - Evaluation and Quality Control Metrics.

Computes unmasked Structural Similarity Index (SSIM) to guarantee non-target
background and furniture parts remain byte-identical and uncorrupted.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:
    ssim_fn = None


def ssim_outside_mask(
    original: Union[Image.Image, np.ndarray],
    edited: Union[Image.Image, np.ndarray],
    mask: Union[Image.Image, np.ndarray],
) -> float:
    """
    Calculates the Structural Similarity Index (SSIM) strictly on pixels outside the mask.
    
    Returns:
        1.0 if pixels outside mask are identical.
        <1.0 if any pixels outside mask differ.
    """
    if isinstance(original, Image.Image):
        orig_arr = np.array(original.convert("L"), dtype=np.uint8)
    else:
        orig_arr = np.array(original, dtype=np.uint8)
        if orig_arr.ndim == 3:
            orig_arr = orig_arr.mean(axis=2).astype(np.uint8)

    if isinstance(edited, Image.Image):
        edit_arr = np.array(edited.convert("L"), dtype=np.uint8)
    else:
        edit_arr = np.array(edited, dtype=np.uint8)
        if edit_arr.ndim == 3:
            edit_arr = edit_arr.mean(axis=2).astype(np.uint8)

    if isinstance(mask, Image.Image):
        mask_arr = np.array(mask.convert("L"), dtype=np.uint8)
    else:
        mask_arr = np.array(mask, dtype=np.uint8)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr.mean(axis=2).astype(np.uint8)

    if orig_arr.shape != edit_arr.shape:
        raise ValueError(f"Original shape {orig_arr.shape} does not match edited shape {edit_arr.shape}")
    if orig_arr.shape != mask_arr.shape:
        raise ValueError(f"Image shape {orig_arr.shape} does not match mask shape {mask_arr.shape}")

    unmasked = (mask_arr == 0)

    # Edge case: mask covers 100% of the image (no unmasked pixels)
    if np.count_nonzero(unmasked) == 0:
        return 1.0

    # Strict byte identity fast-path
    if np.array_equal(orig_arr[unmasked], edit_arr[unmasked]):
        return 1.0

    if ssim_fn is not None:
        # Full SSIM difference map
        _, diff = ssim_fn(orig_arr, edit_arr, full=True)
        val = float(np.clip(diff[unmasked].mean(), 0.0, 1.0))
        # Ensure that if pixels differ, returned SSIM is strictly < 1.0
        return min(val, 0.9999)
    else:
        # Fallback normalized MSE similarity
        mse = np.mean((orig_arr[unmasked].astype(float) - edit_arr[unmasked].astype(float)) ** 2)
        score = float(max(0.0, 1.0 - (mse / (255.0 ** 2))))
        return min(score, 0.9999)


def evaluate_run(
    manifest_path: Union[str, Path],
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluates a batch run by reading manifest.json, computing outside-mask SSIM,
    and writing evaluation.csv.
    """
    m_path = Path(manifest_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Manifest not found: {m_path}")

    with open(m_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    in_path = Path(input_dir)
    out_path = Path(output_dir)
    csv_path = Path(output_csv) if output_csv else out_path / "evaluation.csv"

    results = []
    for item in manifest:
        name = item.get("name", "")
        status = item.get("status", "")
        output_file = item.get("output_path")
        orig_file = in_path / name
        mask_file = out_path / "masks" / f"{Path(name).stem}_mask.png"

        outside_ssim = 1.0
        if "success" in status and output_file and Path(output_file).exists() and orig_file.exists():
            from src.preprocess import load_and_clean, normalize_background
            clean_img = load_and_clean(orig_file)
            if item.get("bg_normalized", False):
                clean_img = normalize_background(clean_img)
            edit_img = Image.open(output_file)
            mask_img = Image.open(mask_file) if mask_file.exists() else Image.new("L", clean_img.size, 0)
            outside_ssim = round(ssim_outside_mask(clean_img, edit_img, mask_img), 4)

        rec = {
            "name": name,
            "status": status,
            "mask_coverage_pct": item.get("mask_coverage_pct", 0.0),
            "time_s": item.get("time_s", 0.0),
            "outside_ssim": outside_ssim,
            "output_path": output_file or "None",
        }
        results.append(rec)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "status", "mask_coverage_pct", "time_s", "outside_ssim", "output_path"]
        )
        writer.writeheader()
        writer.writerows(results)

    return results
