"""
CatalogUpdate AI - ControlNet Comparison and Structure Preservation Analysis.

Runs dual inpainting (with and without ControlNet Canny) across identical images
using identical random seeds, generates a 4-column side-by-side comparison grid:
Original | Mask | Without ControlNet | With ControlNet,
and calculates full-image edge-map SSIM metrics saved to a CSV report.
"""

import os
import csv
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
from PIL import Image, ImageDraw

from src.config import (
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE,
    DEFAULT_SEED,
    DEFAULT_CONTROLNET_SCALE,
    DEFAULT_CANNY_LOW,
    DEFAULT_CANNY_HIGH,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_MASK_DILATION_PX,
)
from src.prompts import FURNITURE_PRESETS, build_prompt
from src.engine import make_control_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare_pipeline")


def compute_edge_map_ssim(
    edge_orig: Image.Image,
    edge_eval: Image.Image,
) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two Canny edge maps
    across the entire image area.
    """
    arr1 = np.array(edge_orig.convert("L"), dtype=np.uint8)
    arr2 = np.array(edge_eval.convert("L"), dtype=np.uint8)

    if arr1.shape != arr2.shape:
        arr2 = np.array(Image.fromarray(arr2).resize((arr1.shape[1], arr1.shape[0]), Image.Resampling.BILINEAR))

    if np.array_equal(arr1, arr2):
        return 1.0

    try:
        from skimage.metrics import structural_similarity as skimage_ssim
        score = float(skimage_ssim(arr1, arr2))
        return round(float(np.clip(score, 0.0, 1.0)), 4)
    except ImportError:
        # Normalized MSE similarity fallback
        mse = float(np.mean((arr1.astype(float) - arr2.astype(float)) ** 2))
        score = max(0.0, 1.0 - (mse / (255.0 ** 2)))
        return round(float(score), 4)


def create_comparison_grid(
    rows_data: List[Dict[str, Any]],
    output_path: Path,
    cell_size: int = 512,
) -> Path:
    """
    Creates a 4-column comparison grid:
    Original | Mask | Without ControlNet | With ControlNet.
    Includes column header labels and per-row image names.
    """
    if not rows_data:
        raise ValueError("No comparison rows provided for grid generation.")

    header_height = 40
    num_rows = len(rows_data)
    grid_w = cell_size * 4
    grid_h = header_height + (cell_size * num_rows)

    grid = Image.new("RGB", (grid_w, grid_h), (25, 25, 30))
    draw = ImageDraw.Draw(grid)

    headers = [
        "Original Product",
        "CLIPSeg Mask",
        "Without ControlNet (Standard Inpaint)",
        "With ControlNet (Canny Structure Preserved)",
    ]

    for col_idx, title in enumerate(headers):
        x = col_idx * cell_size + cell_size // 2
        y = header_height // 2
        draw.text((x, y), title, fill=(240, 240, 240), anchor="mm")

    for row_idx, row in enumerate(rows_data):
        y_offset = header_height + (row_idx * cell_size)

        cells = [
            row["orig"].resize((cell_size, cell_size), Image.Resampling.LANCZOS),
            row["mask"].convert("RGB").resize((cell_size, cell_size), Image.Resampling.BILINEAR),
            row["no_cn"].resize((cell_size, cell_size), Image.Resampling.LANCZOS),
            row["with_cn"].resize((cell_size, cell_size), Image.Resampling.LANCZOS),
        ]

        for col_idx, cell in enumerate(cells):
            x_offset = col_idx * cell_size
            grid.paste(cell, (x_offset, y_offset))
            # Column borders
            draw.rectangle(
                [x_offset, y_offset, x_offset + cell_size, y_offset + cell_size],
                outline=(50, 50, 60),
                width=1,
            )

        # Label image filename in first column
        draw.text(
            (10, y_offset + 10),
            row["name"],
            fill=(255, 255, 0),
        )

    grid_path = output_path / "comparison_grid.png"
    grid.save(grid_path)
    logger.info("Saved comparison grid: %s", grid_path)
    return grid_path


def compare_controlnet(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    part_text: str,
    spec_prompt: str,
    negative_prompt: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    mock: bool = False,
) -> Dict[str, Any]:
    """
    Executes comparison between standard inpainting and ControlNet structure preservation.

    Runs inference across all images in input_dir twice with identical random seeds:
      1) use_controlnet = False (baseline)
      2) use_controlnet = True  (ControlNet Canny)

    Computes whole-image edge-map SSIM against original image to quantify structure preservation.
    Saves a 4-column visual comparison grid and comparison_metrics.csv.

    Returns:
        Dict with keys: 'metrics', 'grid_path', 'csv_path', 'output_dir'.
    """
    in_path = Path(input_dir).resolve()
    out_path = Path(output_dir).resolve()
    no_cn_dir = out_path / "without_controlnet"
    with_cn_dir = out_path / "with_controlnet"
    masks_dir = out_path / "masks"

    out_path.mkdir(parents=True, exist_ok=True)
    no_cn_dir.mkdir(parents=True, exist_ok=True)
    with_cn_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    params = params or {}
    steps = params.get("steps", DEFAULT_STEPS)
    guidance = params.get("guidance", DEFAULT_GUIDANCE)
    seed = params.get("seed", DEFAULT_SEED)
    controlnet_scale = params.get("controlnet_scale", DEFAULT_CONTROLNET_SCALE)
    threshold = params.get("threshold", DEFAULT_MASK_THRESHOLD)
    dilate_px = params.get("dilate_px", DEFAULT_MASK_DILATION_PX)
    canny_low = params.get("canny_low", DEFAULT_CANNY_LOW)
    canny_high = params.get("canny_high", DEFAULT_CANNY_HIGH)
    device = params.get("device", "cuda")
    mock = mock or params.get("mock", False)

    # Initialize editor
    if "editor" in params and params["editor"] is not None:
        editor = params["editor"]
    elif mock or os.environ.get("USE_MOCK_ENGINE", "0").strip().lower() in ("1", "true", "yes"):
        from src.batch import MockCatalogEditor
        editor = MockCatalogEditor(device="cpu")
    else:
        try:
            from src.engine import CatalogEditor
            editor = CatalogEditor(device=device)
        except Exception as e:
            logger.warning("Falling back to MockCatalogEditor: %s", e)
            from src.batch import MockCatalogEditor
            editor = MockCatalogEditor(device="cpu")

    supported_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = sorted([f for f in in_path.iterdir() if f.is_file() and f.suffix.lower() in supported_exts])
    if not image_files:
        raise ValueError(f"No valid images found in {in_path}")

    logger.info("Found %d images to compare in %s", len(image_files), in_path)

    metrics_list: List[Dict[str, Any]] = []
    rows_data: List[Dict[str, Any]] = []

    for idx, img_file in enumerate(image_files):
        t0 = time.time()
        logger.info("[%d/%d] Processing comparison for %s...", idx + 1, len(image_files), img_file.name)

        with Image.open(img_file) as raw:
            orig_img = raw.convert("RGB")

        stem = img_file.stem
        mask = editor.generate_mask(
            orig_img,
            part_text=part_text,
            threshold=threshold,
            dilate_px=dilate_px,
        )
        mask.save(masks_dir / f"{stem}_mask.png")

        # 1. Without ControlNet (Standard inpaint)
        res_no_cn = editor.inpaint(
            image=orig_img,
            mask=mask,
            prompt=spec_prompt,
            negative_prompt=negative_prompt or "",
            steps=steps,
            guidance=guidance,
            seed=seed,
            use_controlnet=False,
        )
        res_no_cn.save(no_cn_dir / f"{stem}_no_controlnet.png")

        # 2. With ControlNet (Canny edge preservation)
        res_with_cn = editor.inpaint(
            image=orig_img,
            mask=mask,
            prompt=spec_prompt,
            negative_prompt=negative_prompt or "",
            steps=steps,
            guidance=guidance,
            seed=seed,
            use_controlnet=True,
            controlnet_scale=controlnet_scale,
            canny_low=canny_low,
            canny_high=canny_high,
        )
        res_with_cn.save(with_cn_dir / f"{stem}_with_controlnet.png")

        # 3. Compute Edge Maps and SSIM structure retention
        edge_orig = make_control_image(orig_img, low_threshold=canny_low, high_threshold=canny_high)
        edge_no_cn = make_control_image(res_no_cn, low_threshold=canny_low, high_threshold=canny_high)
        edge_with_cn = make_control_image(res_with_cn, low_threshold=canny_low, high_threshold=canny_high)

        ssim_no_cn = compute_edge_map_ssim(edge_orig, edge_no_cn)
        ssim_with_cn = compute_edge_map_ssim(edge_orig, edge_with_cn)
        gain = round(ssim_with_cn - ssim_no_cn, 4)

        elapsed = round(time.time() - t0, 3)

        metric_record = {
            "image_name": img_file.name,
            "edge_ssim_without_controlnet": ssim_no_cn,
            "edge_ssim_with_controlnet": ssim_with_cn,
            "structure_gain": gain,
            "controlnet_scale": controlnet_scale,
            "time_s": elapsed,
        }
        metrics_list.append(metric_record)

        rows_data.append({
            "name": img_file.name,
            "orig": orig_img,
            "mask": mask,
            "no_cn": res_no_cn,
            "with_cn": res_with_cn,
        })

        logger.info(
            "[%d/%d] %s: Edge SSIM without ControlNet = %.4f, with ControlNet = %.4f (Gain: %+.4f)",
            idx + 1, len(image_files), img_file.name, ssim_no_cn, ssim_with_cn, gain
        )

    # 4. Generate visual comparison grid
    grid_path = create_comparison_grid(rows_data, out_path)

    # 5. Write CSV report
    csv_path = out_path / "comparison_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_name",
            "edge_ssim_without_controlnet",
            "edge_ssim_with_controlnet",
            "structure_gain",
            "controlnet_scale",
            "time_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)

    logger.info("Saved comparison metrics CSV: %s", csv_path)

    return {
        "metrics": metrics_list,
        "grid_path": str(grid_path),
        "csv_path": str(csv_path),
        "output_dir": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description="CatalogUpdate AI - ControlNet Comparison CLI")
    parser.add_argument("--input", default="data/input", help="Path to input catalog images")
    parser.add_argument("--output", default="data/output_compare", help="Path to comparison output directory")
    parser.add_argument("--preset", default=None, choices=list(FURNITURE_PRESETS.keys()), help="Preset key")
    parser.add_argument("--part", default=None, help="Furniture part to mask (e.g. 'chair legs')")
    parser.add_argument("--prompt", default=None, help="Replacement spec prompt")
    parser.add_argument("--negative-prompt", default="", help="Negative guidance prompt")
    parser.add_argument("--controlnet-scale", type=float, default=DEFAULT_CONTROLNET_SCALE, help="ControlNet scale (default: 0.8)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for both runs")
    parser.add_argument("--mock", action="store_true", help="Run with mock engine on CPU without loading heavy models")

    args = parser.parse_args()

    part = args.part
    prompt = args.prompt

    if args.preset and args.preset in FURNITURE_PRESETS:
        p_info = FURNITURE_PRESETS[args.preset]
        if not part:
            part = p_info["part_text"]
        if not prompt:
            pos, neg = build_prompt(part, p_info["change"])
            prompt = pos
            if not args.negative_prompt:
                args.negative_prompt = neg

    if not part:
        part = "chair legs"
    if not prompt:
        prompt = "solid walnut wood legs, dark satin finish, studio photo"

    params = {
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "controlnet_scale": args.controlnet_scale,
        "mock": args.mock,
    }

    results = compare_controlnet(
        input_dir=args.input,
        output_dir=args.output,
        part_text=part,
        spec_prompt=prompt,
        negative_prompt=args.negative_prompt,
        params=params,
        mock=args.mock,
    )

    print("\nComparison Summary:")
    print(f"  Grid Image: {results['grid_path']}")
    print(f"  Metrics CSV: {results['csv_path']}")
    for m in results["metrics"]:
        print(f"  - {m['image_name']}: without_CN={m['edge_ssim_without_controlnet']:.4f}, with_CN={m['edge_ssim_with_controlnet']:.4f}, Gain={m['structure_gain']:+.4f}")


if __name__ == "__main__":
    main()
