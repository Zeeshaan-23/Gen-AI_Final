"""
CatalogUpdate AI - Image Preprocessing Pipeline.

Provides explicit catalog image preprocessing:
- EXIF orientation correction, metadata stripping, and RGBA-to-white background compositing.
- Near-uniform light background normalization via flood-fill to pure white.
- Aspect ratio preserving square padding and resolution standardization.
- Directory-level batch preprocessing with fault-tolerant corrupted file skipping and CSV reporting.
"""

import csv
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps, ImageDraw

from src.config import (
    DEFAULT_PREPROCESS_TOL,
    DEFAULT_PREPROCESS_SIZE,
    INPAINT_SIZE,
    PADDING_COLOR,
)

logger = logging.getLogger("preprocess")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_and_clean(path: Union[str, Path, Image.Image]) -> Image.Image:
    """
    Loads an image from path (or processes an existing PIL Image), cleans it, and returns
    a standard RGB PIL Image:
    - Handles WebP, PNG, JPG, and RGBA / LA modes.
    - Composites transparent alpha pixels onto a solid white background.
    - Resolves EXIF orientation tags using PIL ImageOps.exif_transpose.
    - Strips all non-essential metadata, EXIF blocks, and embedded profiles.

    Args:
        path: File path (str or Path) or PIL Image instance.

    Returns:
        Clean, metadata-stripped RGB PIL Image.
    """
    if isinstance(path, Image.Image):
        raw_img = path
    else:
        raw_img = Image.open(path)

    # 1. Correct EXIF orientation transpose
    try:
        transposed = ImageOps.exif_transpose(raw_img)
        img = transposed if transposed is not None else raw_img
    except Exception as e:
        logger.debug("EXIF transpose skipped or failed: %s", e)
        img = raw_img

    # 2. Composite transparency / RGBA onto white background
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        img_rgba = img.convert("RGBA")
        white_canvas = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(white_canvas, img_rgba)
        clean_rgb = composite.convert("RGB")
    else:
        clean_rgb = img.convert("RGB")

    # 3. Strip all metadata by pasting into a brand new PIL Image canvas
    stripped = Image.new("RGB", clean_rgb.size)
    stripped.paste(clean_rgb, (0, 0))
    stripped.info.clear()

    return stripped


def normalize_background(img: Image.Image, tol: int = DEFAULT_PREPROCESS_TOL) -> Image.Image:
    """
    Normalizes off-white or near-uniform light catalog studio backdrops to pure white.
    If the perimeter border of the image is near-uniform and light, flood-fills it
    to pure white (255, 255, 255). Otherwise, returns the image unchanged.

    Args:
        img: Input PIL Image.
        tol: Floodfill color distance tolerance threshold (default: 18).

    Returns:
        PIL Image with normalized white background or unchanged image.
    """
    rgb_img = img.convert("RGB")
    w, h = rgb_img.size

    if w < 4 or h < 4:
        return img

    arr = np.array(rgb_img, dtype=np.float32)
    # Extract 1-pixel perimeter border pixels
    border_pixels = np.concatenate([
        arr[0, :, :],       # top edge
        arr[h - 1, :, :],   # bottom edge
        arr[:, 0, :],       # left edge
        arr[:, w - 1, :],   # right edge
    ], axis=0)

    border_gray = border_pixels.mean(axis=1)
    mean_brightness = float(np.mean(border_gray))
    std_brightness = float(np.std(border_gray))

    # A catalog studio background qualifies for normalization if:
    # 1. Average border brightness is high (>= 200 out of 255)
    # 2. Border standard deviation is relatively uniform (<= tol * 1.5)
    if mean_brightness < 200.0 or std_brightness > (tol * 1.5):
        return img

    img_copy = rgb_img.copy()
    seeds = [
        (0, 0),
        (w - 1, 0),
        (0, h - 1),
        (w - 1, h - 1),
        (w // 2, 0),
        (w // 2, h - 1),
        (0, h // 2),
        (w - 1, h // 2),
    ]

    for seed in seeds:
        seed_px = img_copy.getpixel(seed)
        seed_lightness = sum(seed_px) / 3.0
        if seed_lightness >= 190.0 and abs(seed_lightness - mean_brightness) <= (tol * 2.0):
            ImageDraw.floodfill(img_copy, seed, (255, 255, 255), thresh=tol)

    # Return original instance if no pixels actually changed
    if np.array_equal(np.array(rgb_img), np.array(img_copy)):
        return img

    return img_copy


def pad_to_square(
    img: Image.Image,
    size: Optional[int] = None,
    fill: Tuple[int, int, int] = PADDING_COLOR,
) -> Image.Image:
    """
    Pads a rectangular image to square dimensions with a neutral fill color,
    preserving exact furniture aspect ratio. Optionally resizes to (size, size).

    Args:
        img: Input PIL Image.
        size: Optional target square dimension.
        fill: RGB tuple for background padding (default: (255, 255, 255)).

    Returns:
        Square PIL Image.
    """
    w, h = img.size
    max_dim = max(w, h)

    if w == max_dim and h == max_dim:
        square = img.copy()
    else:
        square = Image.new("RGB", (max_dim, max_dim), fill)
        pad_x = (max_dim - w) // 2
        pad_y = (max_dim - h) // 2
        square.paste(img.convert("RGB"), (pad_x, pad_y))

    if size is not None and (square.size[0] != size or square.size[1] != size):
        square = square.resize((size, size), Image.Resampling.LANCZOS)

    return square


def resize_for_model(img: Image.Image, size: int = INPAINT_SIZE[0]) -> Image.Image:
    """
    Pads image to square and resizes to (size, size) for model ingestion (e.g. 512x512).

    Args:
        img: Input PIL Image.
        size: Target model square dimension (default: 512).

    Returns:
        Square PIL Image of size (size, size).
    """
    return pad_to_square(img, size=size, fill=PADDING_COLOR)


def preprocess_dir(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    size: Optional[int] = DEFAULT_PREPROCESS_SIZE,
    tol: int = DEFAULT_PREPROCESS_TOL,
) -> List[Dict[str, Any]]:
    """
    Batch processes an entire directory of catalog images:
    - Loads and cleans each image (transparency composite, EXIF fix, metadata strip).
    - Normalizes near-uniform light backgrounds to pure white.
    - Proportionally scales images down if their longest edge exceeds `size`.
    - Saves cleaned images as PNG to output_dir.
    - Writes output_dir/preprocess_report.csv.
    - Skips and logs corrupt/unreadable files without halting the batch.

    Args:
        input_dir: Path to directory of raw images.
        output_dir: Path to directory where cleaned PNG images and report will be written.
        size: Maximum edge dimension for downscaling (default: 1024; None to keep original size).
        tol: Background normalization tolerance (default: 18).

    Returns:
        List of dicts: [{"name": str, "orig_size": tuple, "new_size": tuple,
                         "bg_normalized": bool, "status": str}, ...]
    """
    in_path = Path(input_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    supported_exts = {".jpg", ".jpeg", ".png", ".webp"}
    raw_files = sorted([
        f for f in in_path.iterdir()
        if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in supported_exts
    ])

    report: List[Dict[str, Any]] = []

    for file_path in raw_files:
        orig_size = None
        new_size = None
        bg_normalized = False

        try:
            with Image.open(file_path) as raw_probe:
                orig_size = raw_probe.size

            cleaned = load_and_clean(file_path)
            norm = normalize_background(cleaned, tol=tol)
            bg_normalized = bool(not np.array_equal(np.array(cleaned), np.array(norm)))

            if size is not None:
                cur_w, cur_h = norm.size
                if max(cur_w, cur_h) > size:
                    scale = size / max(cur_w, cur_h)
                    new_w = max(1, int(round(cur_w * scale)))
                    new_h = max(1, int(round(cur_h * scale)))
                    norm = norm.resize((new_w, new_h), Image.Resampling.LANCZOS)

            new_size = norm.size
            out_file = out_path / f"{file_path.stem}.png"
            norm.save(out_file, "PNG")

            rec = {
                "name": file_path.name,
                "orig_size": orig_size,
                "new_size": new_size,
                "bg_normalized": bg_normalized,
                "status": "success",
            }
            logger.info("Preprocessed %s: %s -> %s (bg_normalized: %s)", file_path.name, orig_size, new_size, bg_normalized)

        except Exception as e:
            logger.warning("Skipping unreadable or corrupt file %s: %s", file_path.name, e)
            rec = {
                "name": file_path.name,
                "orig_size": orig_size,
                "new_size": None,
                "bg_normalized": False,
                "status": f"corrupt_skipped: {str(e)}",
            }

        report.append(rec)

    # Write report CSV
    csv_file = out_path / "preprocess_report.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "orig_size", "new_size", "bg_normalized", "status"]
        )
        writer.writeheader()
        writer.writerows(report)

    logger.info("Preprocessing complete. Processed %d images. Report saved to %s", len(report), csv_file)
    return report


def main():
    parser = argparse.ArgumentParser(description="CatalogUpdate AI - Image Preprocessing CLI")
    parser.add_argument("--input", default="data/input", help="Directory containing raw catalog images")
    parser.add_argument("--output", default="data/clean", help="Directory to save preprocessed PNG images")
    parser.add_argument("--size", type=int, default=DEFAULT_PREPROCESS_SIZE, help=f"Max edge dimension (default: {DEFAULT_PREPROCESS_SIZE})")
    parser.add_argument("--tol", type=int, default=DEFAULT_PREPROCESS_TOL, help=f"Background normalization tolerance (default: {DEFAULT_PREPROCESS_TOL})")

    args = parser.parse_args()
    report = preprocess_dir(input_dir=args.input, output_dir=args.output, size=args.size, tol=args.tol)
    success_count = sum(1 for r in report if r["status"] == "success")
    print(f"\nPreprocessing finished: {success_count}/{len(report)} images cleaned into {args.output}.")


if __name__ == "__main__":
    main()
