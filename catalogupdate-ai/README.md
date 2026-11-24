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
   * **Cell 2**: Clones the GitHub repository and changes directory into `catalogupdate-ai`.
   * **Cell 3**: Installs dependencies via `pip install -r requirements.txt`.
   * **Cell 4**: Prepares a sample catalog chair image in `data/input/`.
   * **Cell 5 & 6**: Runs the smoke test using `CatalogEditor` and visualizes the side-by-side comparison (`Original` vs `Auto-Mask` vs `Inpainted Result`).
   * **Cell 7**: Launches the Gradio web interface with `share=True` for interactive batch editing.

---

## Repository Structure
```
catalogupdate-ai/
  PROJECT.md        # Goals, architecture, tech stack, constraints
  PLAN.md           # 5-phase execution plan and task checklist
  PROGRESS.md       # Append-only chronological engineering log
  DECISIONS.md      # Architectural & technical decisions
  requirements.txt  # Pinned dependencies
  app.py            # Gradio web interface
  src/
    engine.py       # Core inpainting & CLIPSeg segmentation engine
    batch.py        # Batch folder processor and manifest writer
    prompts.py      # Spec-aware prompt templates & material presets
    evaluate.py     # Evaluation metrics (SSIM outside mask)
  notebooks/
    colab_run.ipynb # Google Colab T4 runner & smoke test
  data/
    input/          # Input product catalog images
    output/         # Inpainted catalog images & manifests
```
