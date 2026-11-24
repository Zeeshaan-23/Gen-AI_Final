# CatalogUpdate AI - Demo & Presentation Guide

This guide provides both a structured **3-Minute Presentation Script** for live evaluation and comprehensive technical instructions for running and testing **CatalogUpdate AI** across Google Colab T4 GPU and local CPU environments.

---

## 1. 3-Minute Live Presentation Script

Use this script during a live presentation, defense, or screencast walkthrough.

```
+-----------------------------------------------------------------------------------------+
|                                3-MINUTE PRESENTATION TIMELINE                           |
|                                                                                         |
|   [ 0:00 - 0:45 ]  Problem, Retailer Challenges & Automated Preprocessing               |
|   [ 0:45 - 1:30 ]  Character-Level LSTM Spec Generation & Parser Auto-Fill              |
|   [ 1:30 - 2:15 ]  Batch Inpainting + ControlNet Structure Preservation                 |
|   [ 2:15 - 3:00 ]  Mathematical SSIM Proof, Quality Metrics & Fallback Strategy         |
+-----------------------------------------------------------------------------------------+
```

### [0:00 – 0:45] The Problem, Real-World Catalog Challenges, and Preprocessing
- **What to say**:
  > "In commercial furniture e-commerce, updating product catalogs for new seasonal materials—such as changing metal chair legs to solid walnut wood—traditionally requires either expensive studio reshoots or tedious manual retouching in Photoshop.
  > 
  > However, applying naive diffusion models to vendor imagery creates major problems: vendor photos come with transparent RGBA channels that render as pitch black, camera EXIF rotations, and inconsistent off-white studio backdrops.
  > 
  > To solve this, CatalogUpdate AI begins with an automated **Catalog Preprocessing Engine** (`src/preprocess.py`). It composites transparent alphas onto clean white, rectifies EXIF tilt, and uses a seeded flood-fill to normalize off-white cycloramas to pure RGB white (`tol=18`), ensuring clean inputs before neural inpainting begins."
- **What to show**:
  - Show the 4-image preprocessing strip from Colab Cell 5: Raw input with off-white/transparent background vs. sanitized pure-white normalized output.

---

### [0:45 – 1:30] Character-Level LSTM Spec Generation & Parser Auto-Fill
- **What to say**:
  > "Rather than requiring users to manually craft complex diffusion prompts, we built an autoregressive **Character-Level LSTM** (`src/text/model.py`) that generates domain-specific furniture specification updates.
  > 
  > We chose character-level modeling over word-level tokens because furniture specifications follow a compact, formulaic syntax. This completely eliminates out-of-vocabulary errors, requires zero external tokenization packages, and trains to a validation perplexity of 1.39 in under 40 seconds on a free Colab T4 GPU.
  > 
  > In our Gradio UI, the user clicks 'Generate & Fill Spec'. An autoregressive sample like `'Update: chair legs changed to dark walnut wood, matte finish.'` is generated, and our regex spec parser instantly decomposes it into structured fields (`part`, `color`, `material`, `finish`) and injects studio catalog photography keywords."
- **What to show**:
  - Open Gradio Tab 1 -> Expand the **LSTM Spec Generator** accordion -> Show generation with temperature slider -> Click **Generate & Fill Spec** and show the form fields auto-populating.

---

### [1:30 – 2:15] Batch Inpainting + ControlNet Structure Preservation
- **What to say**:
  > "Next, our core inpainting engine takes over. First, zero-shot **CLIPSeg** (`CIDAS/clipseg-rd64-refined`) segments the targeted component from the text query. We apply morphological dilation (12px) and Gaussian feathering, plus an automated retry mechanism that drops threshold to 0.2 if initial coverage is under 0.5%.
  > 
  > Now, notice the key innovation: **ControlNet Structure Preservation**.
  > Standard Stable Diffusion inpainting frequently suffers from structural drift—chair legs bend, warp, or lose joint alignment.
  > By integrating ControlNet Canny edge guidance (`lllyasviel/control_v11p_sd15_canny`) at a calibrated conditioning scale of 0.8, we lock the exact geometric boundaries and silhouette of the furniture, replacing strictly the surface texture with rich walnut grain."
- **What to show**:
  - Show Colab Cell 11 comparison grid (`Original | Mask | Without ControlNet | With ControlNet`) or toggle ControlNet on/off in the Gradio UI to demonstrate how leg straightness and joint integrity are preserved.

---

### [2:15 – 3:00] Mathematical SSIM Proof, Quality Metrics & Fallback Strategy
- **What to say**:
  > "How do we prove to retailers that the non-target parts of the product and background were not corrupted?
  > In `src/engine.py`, we execute an **alpha-blended paste-back**. The diffusion output is composited strictly through the feathered mask over the original cleaned image.
  > Outside the mask, pixels are mathematically byte-identical (`np.array_equal == True`), yielding an outside-mask SSIM of exactly **1.0000 by construction**. Furthermore, our edge SSIM metric confirms structural preservation within the modified zone.
  > 
  > If a live Colab GPU session ever disconnects or runs out of compute quota during a presentation, our system features a seamless **Local Mock Fallback**: setting `USE_MOCK_ENGINE=1` runs the full Gradio interface and CLI pipelines on any local CPU laptop without requiring GPU hardware or downloading heavy weights."
- **What to show**:
  - Show the manifest table and outside SSIM = 1.0000 in the UI; show downloadable ZIP output; demonstrate clean CPU mock execution.

---

## 2. Running on Google Colab (Full Neural Inpainting & GPU Inference)

Google Colab provides a free 16GB Nvidia T4 GPU, which is the target production hardware for CatalogUpdate AI.

### Step-by-Step Instructions:
1. Open [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) in [Google Colaboratory](https://colab.research.google.com/).
2. Select Runtime: **Runtime** -> **Change runtime type** -> **T4 GPU** -> **Save**.
3. Run all 17 cells sequentially:
   - **Cell 1**: GPU Sanity Check (`nvidia-smi` + PyTorch CUDA check).
   - **Cell 2**: Repository clone (`git clone https://github.com/Zeeshaan-23/Gen-AI_Final.git`) & requirements installation.
   - **Cell 3**: Environment & import verification.
   - **Cell 4**: Synthetic catalog sample image generation.
   - **Cell 5**: Catalog image preprocessing (`preprocess_dir`) across formats with 4-image Before/After strip.
   - **Cell 6 & 7**: Single-image smoke test (`chair legs` -> `walnut wood`) with Before / Mask / After visualization.
   - **Cell 8 & 9**: Batch pipeline execution across catalog images with results grid.
   - **Cell 10**: ControlNet batch execution with structure preservation.
   - **Cell 11**: Comparative evaluation (`src.compare`) with 4-column comparison grid (`Original | Mask | Without ControlNet | With ControlNet`) and edge SSIM table.
   - **Cell 12**: Text corpus generation (`make_corpus`) and text preprocessing (`preprocess_text`).
   - **Cell 13**: `CharLSTM` model training (30 epochs, ~30s on T4) with loss curve plot.
   - **Cell 14**: Autoregressive sampling and quantitative evaluation (`evaluate_text`) with benchmark table.
   - **Cell 15**: Output archive packaging (`catalogupdate_outputs.zip`).
   - **Cell 16**: Gradio web app launch with public `share=True` live link.
   - **Cell 17**: Offline local mock engine verification.

---

## 3. Running Locally on CPU (Development & Mock Mode)

For development, testing, and continuous integration on machines without a dedicated GPU, CatalogUpdate AI includes a lightweight `MockCatalogEditor` that simulates inpainting without downloading multi-gigabyte models.

### 3.1 Preprocessing CLI
```bash
# Clean and normalize raw catalog photos
python -m src.preprocess --input data/input --output data/clean --size 1024 --tol 18
```

### 3.2 Batch Inpainting CLI (Mock Mode)
```bash
# Run batch update using a preset material on sample catalog images
python -m src.batch --input data/input --output data/output --preset chair_walnut_legs --mock --zip
```

Or with custom parameters and ControlNet enabled:
```bash
python -m src.batch \
  --input data/input \
  --output data/output \
  --part "chair legs" \
  --prompt "solid dark walnut wood legs, rich grain, studio photo" \
  --controlnet \
  --controlnet-scale 0.8 \
  --threshold 0.4 \
  --dilate 12 \
  --seed 42 \
  --mock \
  --zip
```

### 3.3 Comparative Evaluation CLI
```bash
# Run comparative evaluation (CPU / Mock mode)
python -m src.compare \
  --input data/input \
  --output data/output_compare \
  --part "chair legs" \
  --prompt "dark walnut wood legs" \
  --mock
```

### 3.4 Text Generation Pipeline
```bash
# 1. Generate realistic raw corpus (492 lines)
python -m src.text.make_corpus --lines 500 --seed 42

# 2. Clean text, extract character vocabulary, and save dataset stats
python -m src.text.preprocess_text --input data/text/furniture_specs_raw.txt --output outputs/text

# 3. Train CharLSTM (fast on CPU for smoke test ~5-10s)
python -m src.text.train --epochs 3 --batch 128 --lr 2e-3

# 4. Generate specification updates with temperature control
python -m src.text.generate --seed "Update: chair legs changed to" --temperature 0.8 --one-line

# 5. Run quantitative evaluation (150 samples across 5 seeds x 3 temperatures)
python -m src.text.evaluate_text --checkpoint outputs/text/charlstm.pt --output-dir outputs/text
```

### 3.5 Gradio Web Application
```bash
# Windows PowerShell
$env:USE_MOCK_ENGINE="1"; python app.py

# Linux / macOS
USE_MOCK_ENGINE=1 python app.py
```
Open `http://127.0.0.1:7860` in any web browser.

---

## 4. Test Suite & Static Quality Checks

All 32 tests run cleanly on local CPU in under 15 seconds without network downloads:

```bash
# Run unit test suite
pytest -q

# Run static analysis
python -m pyflakes app.py src tests scripts

# Validate Colab notebook structure and cell schema
python scripts/validate_colab.py
```
