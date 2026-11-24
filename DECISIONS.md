# Technical Decisions Log

This document records technical decisions, model choices, parameter choices, and trade-offs.

---

## Architectural & Engine Decisions
- **Inpainting Model**: Selected `stable-diffusion-v1-5/stable-diffusion-inpainting` in FP16 because it runs reliably within free Google Colab/Kaggle 16GB T4 VRAM limits with established diffusers support.
- **Zero-Shot Auto-Masking Model**: Selected `CIDAS/clipseg-rd64-refined` because it allows zero-shot text-prompted segmentation of furniture sub-parts without fine-tuning.
- **Local Development Environment Strategy**: Designed code with lazy loading and conditional device selection (`cuda` vs `cpu`) so modules can be imported and unit-tested on local CPU without downloading 5GB+ weights or crashing.
- **Mask Post-Processing**: Used binary thresholding followed by dilation (`scikit-image` / `ImageFilter.MaxFilter`) and light Gaussian blur to ensure the mask fully covers the component boundary and blends smoothly.
- **Quality Evaluation Metric**: Used Structural Similarity Index (SSIM) computed strictly outside the mask area to mathematically prove background and non-target furniture parts remain untouched.
- **Dual-Mode UI**: Implemented Gradio with two distinct tabs (Batch Processing and Single-Image Manual Brush) to satisfy both high-throughput automation and manual edge-case overrides.
- **Repository Structure**: Encapsulated the project under `catalogupdate-ai/` as the self-contained root folder ready for Git repository initialization and Colab cloning.
- **Dependency Pinning**: Pinned dependencies in `requirements.txt` with `numpy>=1.24.4,<2.0.0` to prevent binary ABI incompatibilities on Colab while maintaining full compatibility with PyTorch 2.1+ and Diffusers 0.29+.
- **Import Strategy for Deep Learning Frameworks**: Designed `src/engine.py` with top-level try/except blocks and deferred model instantiation (`_load_clipseg` and `_load_inpaint`) so `import src.engine` succeeds instantly on any local CPU workstation without downloading weights or throwing CUDA errors.
- **Aspect Ratio Preservation via Square Padding**: In `CatalogEditor.inpaint`, padded rectangular images to square dimensions with a neutral white background before 512x512 resizing, ensuring furniture proportions are preserved during diffusion without horizontal or vertical stretching.
- **Selective Alpha-Mask Composition**: In `CatalogEditor.inpaint`, pasted inpainted results strictly over the original image using the feathered mask as an alpha channel, guaranteeing zero background degradation and identical unmasked pixel fidelity.
- **Batch Pipeline Fault Tolerance**: Wrapped per-image processing in `run_batch` with try/except blocks, logging failures and recording errors in the manifest without terminating the entire batch.
- **Mask Coverage Cutoff Threshold**: Set a 0.5% area cutoff in `run_batch` to automatically catch cases where CLIPSeg fails to locate the requested part, marking status as `no_part_found` and saving GPU compute by skipping diffusion inpainting.
- **Reused Engine & Consistent Seed**: Instantiated `CatalogEditor` once per batch run and used a fixed random seed across all images to ensure visual material texture consistency across product catalog variants.
- **Mock Engine for Local Testing & USE_MOCK_ENGINE Environment Variable**: Embedded a `MockCatalogEditor` in `src/batch.py` enabled via `--mock` CLI flag or `USE_MOCK_ENGINE=1` environment variable. This allows end-to-end structural testing of CLI arguments, file pipelines, Gradio web UI interactions, and manifest generation on CPU without requiring GPU hardware or multi-gigabyte model downloads. The mock editor returns a tinted walnut copy of the image over the targeted component mask.
- **Single Global Editor Instance in UI**: In `app.py`, created a single global `CatalogEditor` instance at module load with deferred weight initialization to avoid multi-gigabyte VRAM reallocation per button click while keeping `import app` fast on CPU.
- **Interactive Layer Mask Extraction**: In `app.py`, extracted user-painted brush strokes from `gr.ImageEditor` RGBA layers into binary masks, allowing seamless combining of CLIPSeg auto-masking and manual brush refinement.
- **Non-Blocking UI Feedback**: Used `gr.Warning()` notifications for missing inputs (empty image upload, blank prompt, low mask coverage < 0.5%) to provide clear, actionable user feedback without crashing or showing cryptic stack traces.
- **Automated Colab Share Links**: Implemented environment detection (`is_colab()`) in `app.py` to automatically launch with `share=True` when running inside Google Colab.
- **Centralized Hyperparameter Configuration**: Established `src/config.py` as the single source of truth for all diffusion, inpainting, and segmentation defaults.
- **Structured Spec Prompt Engineering**: `build_prompt` in `src/prompts.py` decouples specification changes (`color`, `material`, `finish`) from free-text while enforcing standardized studio catalog lighting keywords.
- **Automatic Mask Recovery (Retry with Lower Threshold)**: Added a single-retry recovery mechanism in `src/batch.py` that lowers the CLIPSeg threshold to 0.2 and widens dilation when initial mask coverage is below 0.5%, recovering thin furniture legs and angled parts without manual intervention.

---

## Tuning Notes

The following optimal defaults were calibrated for furniture catalog inpainting:

1. **Inference Steps (`steps = 30`)**:
   - 30 diffusion steps provides the sweet spot for SD 1.5 inpainting on Colab T4.
   - At < 20 steps, intricate material textures (such as natural wood grain, marble veining, and woven linen) remain soft and under-resolved.
   - At > 40 steps, texture quality reaches diminishing returns while doubling latency per product image.

2. **Guidance Scale (`guidance = 7.5`)**:
   - 7.5 maintains strict adherence to the specified material and finish without producing high-contrast halos, burnt saturation, or diffusion artifacts.
   - Higher values (> 10.0) cause color oversaturation and unnatural dark outlines at mask boundaries.

3. **Inpainting Strength (`strength = 1.0`)**:
   - Inpainting strength is set to 1.0 because spec changes (e.g. metal legs -> dark walnut wood) require complete replacement of the original component's texture and color rather than subtle blending.

4. **Mask Threshold (`threshold = 0.4`, `retry_threshold = 0.2`)**:
   - 0.4 ensures high-precision localization from CLIPSeg's sigmoid probability map, avoiding bleeding onto adjacent upholstery or rugs.
   - For slim, tapered, or angled furniture legs where 0.4 produces < 0.5% coverage, the automated retry at 0.2 reliably captures the full component geometry.

5. **Mask Dilation Radius (`dilate_px = 12`)**:
   - A 12-pixel dilation expands the mask beyond the raw segmentation outline, fully engulfing edge bevels, shadows, and contact seams.
   - This prevents remnants of the old material from leaking as thin halos along the perimeter of the new component.

6. **Mask Feathering / Blur (`radius = 2`)**:
   - A 2-pixel Gaussian blur creates a soft alpha gradient along the mask boundary.
   - When combined with PIL alpha-compositing, this produces a natural sub-pixel transition that eliminates visible cut lines while keeping all unmasked background pixels strictly identical.

7. **Square Aspect Ratio Padding (`padding_color = (255, 255, 255)`)**:
   - Commercial furniture catalog imagery is often rectangular (e.g. 800x600 or 600x900). Direct resizing to 512x512 distorts product proportions.
   - Centering the product within a padded square (using neutral studio white) preserves exact furniture geometry during diffusion. Unpadding restores original dimensions and crops.

8. **Random Seed Reproducibility (`seed = 42`)**:
   - Reusing a fixed random seed across all images in a batch guarantees consistent wood grain tones, metal sheen, and fabric weave patterns across different angles of the same furniture line.

---

## Duplicate Resolution & Repository Cleanup (Phase 5 Clean-Up)
During repository audit and reorganization from the nested `catalogupdate-ai/` layout to the clean root layout `Gen-AI_Final/`, duplicate files existed between the root and `catalogupdate-ai/`. Every duplicate was diffed and resolved as follows:

1. **`app.py`**:
   - *Root version*: Minimal 21-line forwarder importing `catalogupdate-ai.app`.
   - *catalogupdate-ai version*: Authoritative 20KB implementation containing full dual-tab Gradio Blocks UI (Batch + Single Image manual brush mask), `USE_MOCK_ENGINE` support, layer extraction, progress handling, and Colab share link detection.
   - *Decision*: Kept `catalogupdate-ai/app.py` and moved to root `app.py`.

2. **`src/__init__.py`**:
   - *Root version*: Minimal exports.
   - *catalogupdate-ai version*: Complete exports (`CatalogEditor`, `run_batch`, `zip_outputs`, `build_prompt`, `FURNITURE_PRESETS`, `evaluate_pair`, `evaluate_batch`).
   - *Decision*: Kept `catalogupdate-ai/src/__init__.py` -> `src/__init__.py`.

3. **`src/batch.py`**:
   - *Root version*: Basic 89-line script without retry or mock support.
   - *catalogupdate-ai version*: 15KB robust batch processing pipeline with automatic mask retry (<0.5% coverage recovery), `MockCatalogEditor` for CPU testing, CLI argument parsing, manifest generation, and ZIP packaging.
   - *Decision*: Kept `catalogupdate-ai/src/batch.py` -> `src/batch.py`.

4. **`src/config.py`**:
   - *Root version*: Missing or incomplete defaults.
   - *catalogupdate-ai version*: Centralized hyperparameter configuration (`DEFAULT_STEPS=30`, `DEFAULT_GUIDANCE=7.5`, `DEFAULT_MASK_THRESHOLD=0.4`, `DEFAULT_RETRY_MASK_THRESHOLD=0.2`, `DEFAULT_MASK_DILATION_PX=12`, etc.).
   - *Decision*: Kept `catalogupdate-ai/src/config.py` -> `src/config.py`.

5. **`src/engine.py`**:
   - *Root version*: Basic 68-line engine without device fallbacks or paste-back pixel preservation.
   - *catalogupdate-ai version*: 11KB audit-hardened engine featuring FP16 VAE upcast (prevents black image artifacts), attention slicing, aspect ratio padding, lazy model loading, and strict alpha paste-back ensuring unmasked pixels remain 100% identical.
   - *Decision*: Kept `catalogupdate-ai/src/engine.py` -> `src/engine.py`.

6. **`src/prompts.py`**:
   - *Root version*: Partial prompt builder.
   - *catalogupdate-ai version*: Full spec-aware prompt engineering with 12 furniture presets and negative prompts.
   - *Decision*: Kept `catalogupdate-ai/src/prompts.py` -> `src/prompts.py`.

7. **Tracking and Documentation Files (`PROJECT.md`, `PLAN.md`, `PROGRESS.md`, `DECISIONS.md`, `README.md`)**:
   - *Root versions*: Older scaffolds.
   - *catalogupdate-ai versions*: Complete progress logs, technical decisions, tuning notes, and phase documentation.
   - *Decision*: Moved `catalogupdate-ai` files to repository root.

8. **Caches, Bytecode, and Generated Artifacts**:
   - Untracked and removed all `.pyc` files, `__pycache__/` directories, `.pytest_cache/`, `*.zip` archives, and generated output images from git. Updated `.gitignore` to prevent future tracking.
