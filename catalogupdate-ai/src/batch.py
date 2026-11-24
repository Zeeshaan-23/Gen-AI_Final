"""
CatalogUpdate AI - Batch Processing Pipeline.

Processes folders of product catalog imagery, applying auto-masking and inpainting
with status tracking, error handling, mask coverage filtering, automatic retry with
larger masks, and manifest generation.
"""

import os
import time
import json
import zipfile
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union

import numpy as np
from PIL import Image, ImageDraw

from src.config import (
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE,
    DEFAULT_STRENGTH,
    DEFAULT_SEED,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_RETRY_MASK_THRESHOLD,
    DEFAULT_MASK_DILATION_PX,
    MIN_MASK_COVERAGE_PCT,
)
from src.prompts import build_prompt, FURNITURE_PRESETS, DEFAULT_NEGATIVE_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch_pipeline")


class MockCatalogEditor:
    """Mock editor for structural validation on CPU without loading heavy models."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def generate_mask(
        self,
        image: Image.Image,
        part_text: str,
        threshold: float = DEFAULT_MASK_THRESHOLD,
        dilate_px: int = DEFAULT_MASK_DILATION_PX,
    ) -> Image.Image:
        w, h = image.size
        mask = Image.new("L", (w, h), 0)
        if part_text in ("empty", "nonexistent"):
            return mask
        draw = ImageDraw.Draw(mask)
        # If threshold is low (retry), make a slightly larger area
        scale = 0.25 if threshold <= 0.25 else 0.2
        x0, y0 = int(w * (0.5 - scale / 2)), int(h * (0.5 - scale / 2))
        x1, y1 = int(w * (0.5 + scale / 2)), int(h * (0.5 + scale / 2))
        draw.rectangle([x0, y0, x1, y1], fill=255)
        return mask

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        **kwargs,
    ) -> Image.Image:
        res = image.convert("RGB").copy()
        walnut_tint = Image.new("RGB", image.size, (110, 60, 30))
        res.paste(walnut_tint, (0, 0), mask.convert("L"))
        return res

    def edit(
        self,
        image: Image.Image,
        part_text: str,
        new_spec_prompt: str,
        **kwargs,
    ) -> Dict[str, Image.Image]:
        mask = self.generate_mask(image, part_text)
        result = self.inpaint(image, mask, new_spec_prompt, **kwargs)
        return {"mask": mask, "result": result}


def run_batch(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    part_text: str,
    spec_prompt: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    mock: bool = False,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Executes batch auto-masking and inpainting across all images in input_dir.

    Args:
        input_dir: Directory containing input catalog images.
        output_dir: Directory to save edited images, masks, and manifest.json.
        part_text: Specification part to detect (e.g. 'chair legs').
        spec_prompt: Replacement specification prompt (free-text or composed).
        negative_prompt: Optional negative guidance prompt.
        params: Optional hyperparameters dict:
                - threshold, retry_threshold, retry_larger_mask, dilate_px,
                  steps, guidance, strength, seed, device, mock, editor, change.
        progress_cb: Optional callback func(current_index, total_count, filename).

    Returns:
        List of manifest records describing per-image results.
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    masks_path = output_path / "masks"

    output_path.mkdir(parents=True, exist_ok=True)
    masks_path.mkdir(parents=True, exist_ok=True)

    params = params or {}
    threshold = params.get("threshold", DEFAULT_MASK_THRESHOLD)
    retry_threshold = params.get("retry_threshold", DEFAULT_RETRY_MASK_THRESHOLD)
    retry_larger_mask = params.get("retry_larger_mask", True)
    dilate_px = params.get("dilate_px", DEFAULT_MASK_DILATION_PX)
    steps = params.get("steps", DEFAULT_STEPS)
    guidance = params.get("guidance", DEFAULT_GUIDANCE)
    strength = params.get("strength", DEFAULT_STRENGTH)
    seed = params.get("seed", DEFAULT_SEED)
    device = params.get("device", "cuda")
    mock = mock or params.get("mock", False)

    # Resolve structured prompt if change dict or structured fields are passed
    change_dict = params.get("change")
    if (not spec_prompt or not spec_prompt.strip()) and change_dict:
        composed_pos, composed_neg = build_prompt(part_text, change_dict)
        spec_prompt = composed_pos
        if not negative_prompt:
            negative_prompt = composed_neg
    elif not negative_prompt:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    # Obtain or instantiate editor (reused across the entire batch)
    if "editor" in params and params["editor"] is not None:
        editor = params["editor"]
    elif mock or os.environ.get("USE_MOCK_ENGINE", "0").strip().lower() in ("1", "true", "yes"):
        editor = MockCatalogEditor(device="cpu")
    else:
        try:
            from src.engine import CatalogEditor
            editor = CatalogEditor(device=device)
        except Exception as e:
            logger.warning("Could not load real CatalogEditor, falling back to MockCatalogEditor: %s", e)
            editor = MockCatalogEditor(device="cpu")

    supported_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = sorted([f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in supported_exts])
    total = len(image_files)

    logger.info("Found %d images in %s to process.", total, input_path)
    manifest: List[Dict[str, Any]] = []

    for idx, img_file in enumerate(image_files):
        if progress_cb:
            progress_cb(idx + 1, total, img_file.name)

        t0 = time.time()
        stem = img_file.stem
        edited_file = output_path / f"{stem}_edited.png"
        mask_file = masks_path / f"{stem}_mask.png"
        retried = False

        try:
            with Image.open(img_file) as raw_img:
                img = raw_img.convert("RGB")

            # 1. Initial Mask Generation
            mask = editor.generate_mask(
                image=img,
                part_text=part_text,
                threshold=threshold,
                dilate_px=dilate_px,
            )

            mask_arr = np.array(mask.convert("L"))
            mask_pixels = int(np.count_nonzero(mask_arr > 0))
            total_pixels = int(mask_arr.size)
            coverage_pct = round((mask_pixels / total_pixels) * 100.0, 2)

            # 2. Check coverage cutoff (< 0.5%) & Retry with larger mask if enabled
            if coverage_pct < MIN_MASK_COVERAGE_PCT and retry_larger_mask:
                logger.info(
                    "[%d/%d] %s: Mask coverage (%.2f%%) < %.1f%%. Retrying once with lower threshold %.2f...",
                    idx + 1, total, img_file.name, coverage_pct, MIN_MASK_COVERAGE_PCT, retry_threshold
                )
                retry_mask = editor.generate_mask(
                    image=img,
                    part_text=part_text,
                    threshold=retry_threshold,
                    dilate_px=dilate_px + 4,
                )
                retry_arr = np.array(retry_mask.convert("L"))
                retry_coverage = round((int(np.count_nonzero(retry_arr > 0)) / total_pixels) * 100.0, 2)

                if retry_coverage >= MIN_MASK_COVERAGE_PCT:
                    logger.info("[%d/%d] %s: Retry succeeded with %.2f%% coverage!", idx + 1, total, img_file.name, retry_coverage)
                    mask = retry_mask
                    coverage_pct = retry_coverage
                    retried = True

            # Save the final mask for inspection
            mask.save(mask_file)

            if coverage_pct < MIN_MASK_COVERAGE_PCT:
                status = "no_part_found"
                output_path_str = None
                logger.info("[%d/%d] %s: Skipped (no part found, coverage: %.2f%%)", idx + 1, total, img_file.name, coverage_pct)
            else:
                # 3. Inpaint using fixed seed for consistency
                edited = editor.inpaint(
                    image=img,
                    mask=mask,
                    prompt=spec_prompt,
                    negative_prompt=negative_prompt or "",
                    steps=steps,
                    guidance=guidance,
                    strength=strength,
                    seed=seed,
                )
                edited.save(edited_file)
                status = "success (retried_mask)" if retried else "success"
                output_path_str = str(edited_file)
                logger.info("[%d/%d] %s: Successfully edited (%s, coverage: %.2f%%)", idx + 1, total, img_file.name, status, coverage_pct)

            elapsed = round(time.time() - t0, 3)
            record = {
                "name": img_file.name,
                "status": status,
                "mask_coverage_pct": coverage_pct,
                "time_s": elapsed,
                "output_path": output_path_str,
                "part_text": part_text,
                "prompt": spec_prompt,
            }

        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            logger.error("[%d/%d] %s: Failed with error: %s", idx + 1, total, img_file.name, e)
            record = {
                "name": img_file.name,
                "status": f"error: {str(e)}",
                "mask_coverage_pct": 0.0,
                "time_s": elapsed,
                "output_path": None,
                "part_text": part_text,
                "prompt": spec_prompt,
            }

        manifest.append(record)

        # Clear PyTorch caching allocator and run garbage collection between batch images
        try:
            import torch
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        import gc
        gc.collect()

    # Write manifest.json in output_dir
    manifest_file = output_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Batch completed. Manifest saved to %s", manifest_file)
    return manifest


def zip_outputs(output_dir: Union[str, Path]) -> str:
    """
    Compresses the output directory contents (images, masks, manifest) into a zip archive.
    """
    target_dir = Path(output_dir).resolve()
    if not target_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {target_dir}")

    zip_path = target_dir.parent / f"{target_dir.name}_archive.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(target_dir):
            for file in files:
                file_full = Path(root) / file
                if file_full.resolve() == zip_path.resolve():
                    continue
                arcname = file_full.relative_to(target_dir)
                zipf.write(file_full, arcname)

    logger.info("Created output archive: %s", zip_path)
    return str(zip_path)


def main():
    parser = argparse.ArgumentParser(description="CatalogUpdate AI - Batch Processing CLI")
    parser.add_argument("--input", default="data/input", help="Path to input image directory")
    parser.add_argument("--output", default="data/output", help="Path to output directory")
    parser.add_argument("--preset", default=None, choices=list(FURNITURE_PRESETS.keys()), help="Preset key (e.g. chair_walnut_legs)")
    parser.add_argument("--part", default=None, help="Furniture part to mask (e.g. 'chair legs')")
    parser.add_argument("--prompt", default=None, help="Replacement spec prompt")
    parser.add_argument("--color", default=None, help="Spec color attribute")
    parser.add_argument("--material", default=None, help="Spec material attribute")
    parser.add_argument("--finish", default=None, help="Spec finish attribute")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MASK_THRESHOLD, help="Mask probability threshold")
    parser.add_argument("--no-retry", action="store_true", help="Disable automatic retry with larger mask on low coverage")
    parser.add_argument("--dilate", type=int, default=DEFAULT_MASK_DILATION_PX, help="Mask dilation radius in pixels")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Inpainting inference steps")
    parser.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE, help="Guidance scale")
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH, help="Inpainting strength")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--zip", action="store_true", help="Create a zip archive of the output directory")
    parser.add_argument("--mock", action="store_true", help="Run with mock engine on CPU without loading models")

    args = parser.parse_args()

    part = args.part
    prompt = args.prompt
    change_dict = {}

    if args.preset and args.preset in FURNITURE_PRESETS:
        preset_info = FURNITURE_PRESETS[args.preset]
        if not part:
            part = preset_info["part_text"]
        change_dict = preset_info["change"].copy()

    if args.color:
        change_dict["color"] = args.color
    if args.material:
        change_dict["material"] = args.material
    if args.finish:
        change_dict["finish"] = args.finish

    if not part:
        part = "chair legs"

    if not prompt and change_dict:
        pos, neg = build_prompt(part, change_dict)
        prompt = pos
        if not args.negative_prompt:
            args.negative_prompt = neg
    elif not prompt:
        prompt = "dark walnut wood legs"

    params = {
        "threshold": args.threshold,
        "retry_larger_mask": not args.no_retry,
        "dilate_px": args.dilate,
        "steps": args.steps,
        "guidance": args.guidance,
        "strength": args.strength,
        "seed": args.seed,
        "device": args.device,
        "mock": args.mock,
    }

    manifest = run_batch(
        input_dir=args.input,
        output_dir=args.output,
        part_text=part,
        spec_prompt=prompt,
        negative_prompt=args.negative_prompt,
        params=params,
    )

    if args.zip:
        zip_path = zip_outputs(args.output)
        print(f"Archive created: {zip_path}")

    print(f"\nBatch finished successfully. {len(manifest)} items logged in manifest.json.")


if __name__ == "__main__":
    main()
