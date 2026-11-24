# CatalogUpdate AI - Visual Figure & Artifact Guide

This document defines the 8 required visual figures and evaluation artifacts to capture when running [`notebooks/colab_run.ipynb`](../../notebooks/colab_run.ipynb) on a Google Colab T4 GPU environment.

Each figure corresponds to an exact placeholder in [`REPORT.md`](../../REPORT.md) and provides visual and quantitative proof of system execution.

---

## 1. Directory Structure

Save all captured figures in this directory (`docs/figures/`) with the exact filenames listed below:

```
docs/figures/
  01_gpu_check.png
  02_single_smoke_test.png
  03_batch_results_grid.png
  04_controlnet_comparison_grid.png
  05_text_loss_curve.png
  06_text_eval_summary.png
  07_gradio_ui_batch.png
  08_gradio_ui_lstm.png
```

---

## 2. Artifact Specification & Capture Instructions

| File Name | Notebook Source | Description | Capture Instructions |
|---|---|---|---|
| `01_gpu_check.png` | **Cell 1** | Nvidia T4 GPU sanity check, CUDA availability, and VRAM capacity. | Run Cell 1. Capture the terminal output showing `nvidia-smi` driver table, GPU name (`Tesla T4`), total memory (~15-16 GB), and PyTorch `torch.cuda.is_available() == True`. |
| `02_single_smoke_test.png` | **Cell 7** | Single-image inpainting smoke test comparison strip. | Run Cell 7. Right-click or snip the 3-panel matplotlib visualization: `Original Image` vs. `Auto-Mask (CLIPSeg)` vs. `Inpainted Result (Walnut Wood)`. |
| `03_batch_results_grid.png` | **Cell 9** | Multi-image batch processing gallery across diverse catalog items. | Run Cell 9 (`run_batch`). Capture the multi-row Before / Mask / Inpainted Result grid showing batch execution across `data/input/`. |
| `04_controlnet_comparison_grid.png` | **Cell 11** | 4-column side-by-side structural comparison with and without ControlNet. | Run Cell 11 (`src.compare`). Capture the 4-column figure: `Original` \| `Mask` \| `Without ControlNet (Drift)` \| `With ControlNet (Preserved)`. |
| `05_text_loss_curve.png` | **Cell 13** | CharLSTM training vs. validation loss trajectory across 30 epochs. | Run Cell 13 (`src.text.train`). Download or snip the generated plot displaying the smooth loss drop from ~3.5 to ~0.4. (Also auto-saved to `outputs/text/loss_curve.png`). |
| `06_text_eval_summary.png` | **Cell 14** | Quantitative text generation evaluation summary table. | Run Cell 14 (`src.text.evaluate_text`). Snip the markdown summary table showing Perplexity, Format Validity, Vocabulary Validity, and Diversity metrics across temperatures `0.5`, `0.8`, `1.1`. |
| `07_gradio_ui_batch.png` | **Cell 16** | Gradio web interface Tab 1 ("Batch Catalog Update") in browser. | Open the generated `https://*.gradio.live` link in a browser. Upload images, select preset `chair_walnut_legs`, run the batch, and capture the full interface with result galleries and manifest table. |
| `08_gradio_ui_lstm.png` | **Cell 16** | Gradio UI Tab 1 showing the expanded LSTM Spec Generator accordion. | In the Gradio app, expand "Generate a spec update (LSTM)", select temperature 0.8, click "Generate & Fill Spec", and take a focused screenshot showing the generated specification string and auto-populated fields. |

---

## 3. Best Practices for Figure Capture

1. **Resolution & Clarity**:
   - Capture at 100% browser zoom or export directly using matplotlib `plt.savefig(..., dpi=200, bbox_inches='tight')`.
   - Ensure axis labels, legends, and table text are sharp and legible.
2. **Deterministic Outputs**:
   - All batch and text training scripts use fixed seeds (`seed = 42`) for full reproducibility across runs.
3. **Automated File Recovery**:
   - Cell 15 of `colab_run.ipynb` creates a consolidated `catalogupdate_outputs.zip` archive containing generated images, Canny maps, CSV reports, and model checkpoints, which can be downloaded in one click from the Colab file browser.
