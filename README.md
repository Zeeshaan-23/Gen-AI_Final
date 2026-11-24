# CatalogUpdate AI

A batch product-image editor designed for furniture catalogs. The tool automatically masks target furniture parts using text prompts and inpaints new material specifications (e.g., "chair legs -> walnut wood") with zero physical reshoots.

---

## How to Run on Google Colab (Free T4 GPU)

1. **Open Google Colab**:
   Open [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) in [Google Colaboratory](https://colab.research.google.com/).

2. **Select Hardware Accelerator**:
   * Navigate to `Runtime` > `Change runtime type`.
   * Set **Hardware accelerator** to **T4 GPU**.
   * Click **Save**.

3. **Run the Notebook Cells**:
   * **Cell 1**: Verifies GPU availability (`nvidia-smi` and PyTorch CUDA).
   * **Cell 2**: Clones the GitHub repository and installs dependencies:
     ```bash
     !git clone https://github.com/Zeeshaan-23/Gen-AI_Final.git
     %cd Gen-AI_Final
     !pip install -q -r requirements.txt
     ```
   * **Cell 3**: Verifies requirements installation.
   * **Cell 4**: Prepares a sample catalog chair image in `data/input/`.
   * **Cell 5**: Runs the catalog image preprocessing pipeline (`preprocess_dir`) and displays a 4-image Before/After strip (`Raw Input` vs `Cleaned Output`).
   * **Cell 6 & 7**: Runs the smoke test using `CatalogEditor` and visualizes the side-by-side comparison (`Original` vs `Auto-Mask` vs `Inpainted Result`).
   * **Cell 8**: Runs the batch pipeline across `data/input/`.
   * **Cell 9**: Displays the batch results grid (Before vs. Mask vs. After).
   * **Cell 10**: Runs the structure-preserving batch pipeline with ControlNet Canny.
   * **Cell 11**: Runs `compare_controlnet` and displays the side-by-side 4-column comparison grid and edge-map SSIM metrics.
   * **Cell 12**: Packages and downloads the output archive.
   * **Cell 13**: Launches the Gradio web interface with `share=True` for interactive batch editing.

---

## Repository Structure
```
Gen-AI_Final/
  PROJECT.md        # Goals, architecture, tech stack, constraints
  PLAN.md           # Execution plan and task checklist
  PROGRESS.md       # Append-only chronological engineering log
  DECISIONS.md      # Architectural & technical decisions
  DEMO.md           # Quickstart and demo run guide
  README.md         # Project documentation and Colab setup guide
  requirements.txt  # Pinned dependencies
  .gitignore        # Git ignore rules
  app.py            # Gradio web interface (batch + single-image manual mask + ControlNet)
  src/
    __init__.py     # Module exports
    engine.py       # Core inpainting, CLIPSeg segmentation & ControlNet Canny engine
    preprocess.py   # Catalog image loading, RGBA compositing, background normalization & resizing
    batch.py        # Batch folder processor, mock engine, and manifest writer
    compare.py      # Dual comparison pipeline (without vs with ControlNet) and edge SSIM
    prompts.py      # Spec-aware prompt templates & 12 furniture presets
    evaluate.py     # Evaluation metrics (SSIM outside mask)
    config.py       # Centralized hyperparameters & tuning configuration
  tests/            # Pytest test suite with CPU mock engine
    conftest.py     # Shared test fixtures
    test_batch.py   # Batch processing tests
    test_controlnet.py # ControlNet edge map, manifest & compare tests
    test_engine.py  # Engine padding, clipping, and CPU safety tests
    test_evaluate.py# SSIM evaluation metric tests
    test_preprocess.py # RGBA composite, padding, background normalization & report tests
    test_prompts.py # Prompt builder & preset validation tests
  notebooks/
    colab_run.ipynb # Google Colab T4 runner & smoke test
  data/
    input/          # Input product catalog images (e.g. sample_chair.png)
    output/         # Inpainted catalog images, masks, and manifest.json
  verification/     # Verification report, test logs, and UI screenshots
    VERIFICATION.md # Full audit and verification report
    logs/           # Raw verification logs
    screenshots/    # UI verification screenshots
```
