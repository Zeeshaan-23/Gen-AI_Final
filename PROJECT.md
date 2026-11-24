# CatalogUpdate AI

## 1. Project Goal
CatalogUpdate AI is a batch product-image editor designed specifically for furniture catalogs. It enables commercial furniture retailers and designers to apply specification changes (e.g., "chair legs -> walnut wood", "cushion fabric -> beige linen") across a set of product catalog images automatically. The tool uses zero-shot open-vocabulary segmentation to detect and mask the targeted furniture component and inpainting diffusion models to render the new material specification seamlessly without requiring expensive physical reshoots.

---

## 2. Architecture & Workflow

### 2.1 Core Pipeline
1. **Catalog Preprocessing**: Ingests raw product photos (`data/input/` or uploaded via UI), handles RGBA transparency by compositing onto pure white, fixes EXIF orientation rotations, strips metadata, and normalizes near-uniform light studio backdrops to pure white via seeded flood-filling.
2. **Text-Prompted Auto-Masking**: Uses `CIDAS/clipseg-rd64-refined` to generate a probability heatmap of the target furniture part from the text prompt, followed by thresholding and morphological dilation to produce a clean binary mask.
3. **Spec Inpainting**: Feeds the cleaned image, generated mask, and a spec-aware positive/negative prompt into `stable-diffusion-v1-5/stable-diffusion-inpainting` (running in FP16 on GPU), with optional ControlNet Canny edge guidance.
4. **Output Generation & Manifest**: Saves modified images to `data/output/` and logs metadata, prompts, timestamps, original size, background normalization status, and status into a results manifest (`manifest.json` / `manifest.csv`).

### 2.2 Dual Interface
- **Batch Processing Tab**: High-throughput processing for directories of catalog imagery with preset material selection and automated manifest generation.
- **Single Image / Manual Mask Tab**: Interactive Gradio canvas allowing users to inspect or manually paint/touch up masks when automated segmentation needs fine control.

### 2.3 Evaluation & Quality Control
- **SSIM Calculation**: Measures Structural Similarity Index (SSIM) strictly on unmasked pixels to guarantee non-target furniture parts and backgrounds remain completely unchanged.
- **Side-by-Side Visual Inspection**: Side-by-side comparison grids (Original Image vs. Auto-Mask vs. Inpainted Output).

---

## 3. Tech Stack
- **Python Version**: Python 3.10+
- **Deep Learning Framework**: `torch` (PyTorch)
- **Generative AI & Inpainting**: `diffusers` (`StableDiffusionInpaintPipeline`), `accelerate`
- **Zero-Shot Segmentation**: `transformers` (`CLIPSegProcessor`, `CLIPSegForImageSegmentation`)
- **User Interface**: `gradio` (Blocks interface)
- **Image Processing & Math**: `Pillow` (PIL), `numpy`, `scikit-image`

---

## 4. Models
- **Inpainting Model**: `stable-diffusion-v1-5/stable-diffusion-inpainting` (FP16 on CUDA, CPU fallback for import verification)
- **Auto-Masking Model**: `CIDAS/clipseg-rd64-refined`
- **ControlNet Model**: `lllyasviel/control_v11p_sd15_canny` (FP16 on CUDA, optional structure preservation)
- **Text Generation Model**: Custom 2-layer Character-Level LSTM (`CharLSTM`, hidden 256, embedding 64) for autoregressive furniture specification generation

---

## 5. Constraints & Environment
- **Execution Target**: Free Google Colab / Kaggle T4 GPU (16GB VRAM).
- **Local Machine**: CPU only (no local GPU inference; all code must import cleanly on CPU without heavy weight downloads during local testing).
- **Cost**: $0 / No paid API keys or proprietary services. Strictly open-source models executed locally.
- **Timeline**: 2 days strict deadline. Minimal complexity, no over-engineering.
- **Delivery**: GitHub repository containing code and `notebooks/colab_run.ipynb` for 1-click Colab cloning, dependency installation, and launching Gradio with `share=True`.

---

## 6. Repository Structure
```
Gen-AI_Final/
  PROJECT.md
  PLAN.md
  PROGRESS.md
  DECISIONS.md
  DEMO.md
  REPORT.md
  CHANGELOG.md
  README.md
  requirements.txt
  .gitignore
  app.py
  src/
    __init__.py
    engine.py
    preprocess.py
    batch.py
    compare.py
    prompts.py
    evaluate.py
    config.py
    text/
      __init__.py
      make_corpus.py
      preprocess_text.py
      model.py
      train.py
      generate.py
      evaluate_text.py
      spec_parser.py
  tests/
    conftest.py
    test_batch.py
    test_controlnet.py
    test_engine.py
    test_evaluate.py
    test_preprocess.py
    test_prompts.py
    test_text.py
  docs/
    figures/
      README.md
  notebooks/
    colab_run.ipynb
  data/
    input/
    output/
    text/
  verification/
    VERIFICATION.md
    logs/
    screenshots/
```
