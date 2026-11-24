# Project Implementation Plan: CatalogUpdate AI

This document outlines the 5 execution phases of CatalogUpdate AI. Each phase includes a checklist of tasks and an acceptance criterion.

---

## Phase 1: Scaffold + Core Engine (Inpaint + CLIPSeg Masking) + Colab Notebook
- [x] Create project repository structure and tracking files (PROJECT.md, PLAN.md, PROGRESS.md, DECISIONS.md)
- [x] Create placeholder files for application, modules, notebook, and data folders
- [x] Define project dependencies in `requirements.txt`
- [x] Implement `src/engine.py` with device detection (CUDA/CPU), CLIPSeg segmentation, and SD inpainting pipeline setup
- [x] Implement mask post-processing utilities (thresholding, binary dilation, blurring/feathering) in `src/engine.py`
- [x] Build `notebooks/colab_run.ipynb` with repository cloning, dependency installation, and GPU sanity checks
- [x] Verify clean module imports on CPU without downloading heavy model weights locally

**Acceptance Criterion**: `src/engine.py` imports cleanly on local CPU with no syntax/import errors. `colab_run.ipynb` can execute on Google Colab T4 GPU, load both models in FP16, generate a CLIPSeg mask for a test query, and execute an inpainting step successfully.

---

## Phase 2: Batch Pipeline (Folder In, Folder Out, Results Manifest)
- [x] Implement `src/batch.py` to ingest images from an input folder (`data/input/`)
- [x] Add image format validation (PNG, JPG, JPEG, WEBP) and error handling
- [x] Integrate core engine to process images sequentially with progress reporting
- [x] Save processed images to `data/output/` with deterministic naming (`{stem}_edited.png` and `masks/{stem}_mask.png`)
- [x] Implement mask coverage threshold check (< 0.5% triggers `no_part_found` status)
- [x] Output a structured results manifest (`manifest.json`) detailing name, status, mask_coverage_pct, time_s, output_path, part_text, and prompt
- [x] Implement output packaging utility `zip_outputs(output_dir)`
- [x] Implement CLI interface (`python -m src.batch ...`) with mock testing support on CPU
- [x] Add batch execution and before/after visualization grid cells to `notebooks/colab_run.ipynb`

**Acceptance Criterion**: CLI runs end to end structurally (argument parsing, file handling) on CPU with a mocked engine; real run is verified on Colab.

---

## Phase 3: Gradio UI (Batch Tab + Single-Image Manual-Mask Tab)
- [x] Construct Gradio Blocks interface in `app.py`
- [x] Implement Tab 1: Batch Processing (multi-file upload, part textbox, new spec textbox, advanced accordion, gr.Progress, before/after gallery, mask preview gallery, manifest status table, downloadable ZIP)
- [x] Implement Tab 2: Single-Image Interactive (gr.ImageEditor with manual brush painting, CLIPSeg Auto-mask button, spec textbox, before/after output views)
- [x] Configure global `CatalogEditor` loaded once at app start with deferred weight loading
- [x] Implement readable UI error notifications (`gr.Warning` for empty uploads, empty prompts, no part found)
- [x] Configure `launch(share=True)` automatic Colab detection and update notebook launch cell

**Acceptance Criterion**: `python -c "import app"` works on CPU with model loading deferred; UI runs on Colab.

---

## Phase 4: Spec-Aware Prompt Templates + Furniture Presets + Quality Tuning
- [x] Create `src/prompts.py` with curated furniture material presets (12 presets covering wood, fabric, marble, brass, leather, velvet, rattan, steel, chrome)
- [x] Implement `build_prompt(part, change)` combining structured spec attributes (color, material, finish, shape/style, extra) with catalog studio photography terms
- [x] Establish shared negative prompt suppressing diffusion artifacts, seams, edge bleeding, and visual noise
- [x] Centralize all default tuning parameters in `src/config.py` (steps, guidance, mask threshold, retry threshold, dilation, blur, padding, seed, strength)
- [x] Implement automatic retry with larger mask (lower threshold) in `src/batch.py` when initial coverage < 0.5%
- [x] Update `app.py` and `batch.py` with structured spec fields and quick preset dropdowns
- [x] Add detailed Tuning Notes section to `DECISIONS.md`

**Acceptance Criterion**: Preset prompts print correctly via `python -m src.prompts` demo; `python -c "import app"` imports cleanly on CPU.

---

## Phase 5: Evaluation (SSIM Outside Mask, Results Grid), README, Demo Script
- [x] Implement `src/evaluate.py` to compute Structural Similarity Index (SSIM) exclusively on unmasked pixels between original and output images
- [x] Build visualization utility in `src/evaluate.py` to generate side-by-side comparison grids (Original, Auto-Mask, Inpainted Result)
- [x] Create a lightweight demo verification script (`demo.py` or sample runner) / `DEMO.md`
- [x] Write detailed `README.md` covering project overview, architecture, quickstart, Colab instructions, and sample outputs

**Acceptance Criterion**: SSIM evaluation script validates background preservation on sample pairs, side-by-side comparison grids are generated, and README provides end-to-end instructions for reproducing results.
