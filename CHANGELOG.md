# CatalogUpdate AI - Changelog & Commit History

This document chronicles the development history of **CatalogUpdate AI**, mapping git commits directly to project implementation phases, detailing architectural milestones, and providing guidelines for future version control hygiene.

---

## 1. Commit History by Phase

### Phase 8: Character-Level LSTM Text Generation Module
- **Commit**: `930432b`
- **Date**: `2026-11-24 22:51:06 +0530`
- **Author**: Zeeshaan-23
- **Summary**: `Add character-level LSTM text generation module, preprocessing, evaluation, and UI integration`
- **Key Deliverables**:
  - `src/text/make_corpus.py`: Generates 492 lines of raw furniture specification change sentences from vocabulary lists with intentional realistic noise (casing, stray punctuation, duplicates, blank lines).
  - `src/text/preprocess_text.py`: Cleans raw text, extracts character vocabulary (`outputs/text/vocab.json`), constructs sliding window tensors (`seq_len=40, stride=3`), and logs dataset stats to `outputs/text/dataset_stats.json`.
  - `src/text/model.py`: Implements 2-layer stacked `CharLSTM` PyTorch model (Embedding 64 -> LSTM hidden 256 -> Linear vocab_size).
  - `src/text/train.py`: Trains CharLSTM with loss curve generation (`outputs/text/loss_curve.png`), training log CSV, and best checkpoint saving.
  - `src/text/generate.py`: Autoregressive sampling with temperature scaling (0.2–1.5), top-k filtering, and newline termination.
  - `src/text/evaluate_text.py`: Evaluates 150 samples across 5 seeds and 3 temperatures for validation perplexity (1.39), format validity (100%), vocabulary validity (95%), relevance (93.3%), and distinct-1/2 diversity.
  - `src/text/spec_parser.py`: Regex template matching and safe fallback mapping natural language strings to structured attributes (`part`, `color`, `material`, `finish`).
  - `app.py`: Integrated "Generate a spec update (LSTM)" accordion in Tab 1 with automatic field population.
  - `tests/test_text.py`: 7 unit tests covering text cleaning, vocab persistence, sequence extraction, forward pass, CPU training smoke test (<10s), generation, and parsing.

---

### Phase 7: Catalog Image Preprocessing Stage
- **Commit**: `051b4e4`
- **Date**: `2026-11-24 22:30:54 +0530`
- **Author**: Zeeshaan-23
- **Summary**: `Add explicit catalog image preprocessing stage, RGBA compositing, and background normalization`
- **Key Deliverables**:
  - `src/preprocess.py`: Implements `load_and_clean` (RGBA transparency composite onto white canvas, EXIF rotation correction, metadata stripping), `normalize_background` (seeded flood-fill to pure white for light studio cycloramas), `pad_to_square`, `resize_for_model`, and `preprocess_dir`.
  - `src/batch.py`: Added `--preprocess` flag (default on); records `orig_size` and `bg_normalized` in `manifest.json`.
  - `src/evaluate.py`: Updated outside-mask SSIM calculation to evaluate against cleaned images, eliminating dimension/channel mismatches.
  - `tests/test_preprocess.py`: 6 unit tests covering RGBA compositing, square padding, corrupt file skipping, CSV report rows, and batch integration.
  - `notebooks/colab_run.ipynb`: Added Cell 5 rendering a 4-image Before/After preprocessing strip (`Raw Input` vs `Cleaned Output`).

---

### Phase 6: ControlNet Structure Preservation & Comparative Evaluation
- **Commit**: `7e2e17f`
- **Date**: `2026-11-24 22:16:42 +0530`
- **Author**: Zeeshaan-23
- **Summary**: `Add ControlNet structure preservation, Canny edge pipeline, and comparative evaluation`
- **Key Deliverables**:
  - `src/engine.py`: Added `make_control_image` generating 512x512 3-channel Canny edge maps; lazy-loaded `StableDiffusionControlNetInpaintPipeline` (`lllyasviel/control_v11p_sd15_canny`) in unified FP16.
  - `src/config.py`: Centralized ControlNet defaults (`DEFAULT_USE_CONTROLNET=False`, `DEFAULT_CONTROLNET_SCALE=0.8`, `DEFAULT_CANNY_LOW=100`, `DEFAULT_CANNY_HIGH=200`).
  - `src/compare.py`: Built comparative evaluation pipeline generating 4-column visual grids (`Original | Mask | Without ControlNet | With ControlNet`) and computing edge-map SSIM metrics to quantify geometric fidelity.
  - `app.py`: Added ControlNet checkbox, conditioning scale slider (0.0–1.5), and Canny edge map preview components.
  - `tests/test_controlnet.py`: 6 unit tests verifying edge extraction, mock engine handling, manifest fields, CLI arguments, and compare grids.

---

### Colab T4 Runtime Fix: VAE Dtype Mismatch
- **Commit**: `5f71e7f`
- **Date**: `2026-11-24 15:06:40 +0530`
- **Author**: Zeeshaan-23
- **Summary**: `Fix Colab T4 RuntimeError: remove VAE float32 upcast to keep inpaint pipeline unified in fp16`
- **Key Deliverables**:
  - `src/engine.py`: Removed VAE float32 upcasting in `_load_inpaint`, keeping the entire pipeline unified in FP16 on CUDA.
  - Resolved `RuntimeError: Input type (c10::Half) and bias type (float) should be the same` during VAE decoding on Google Colab T4 GPU.

---

### Clean-Up & Layout Reorganization
- **Commit**: `d1529b0`
- **Date**: `2026-11-24 14:19:54 +0530`
- **Author**: Zeeshaan-23
- **Summary**: `Clean repo layout: single project root, remove caches, fix paths`
- **Key Deliverables**:
  - Consolidated nested `catalogupdate-ai/` layout into unified repository root `Gen-AI_Final/`.
  - Untracked and purged all bytecode caches (`__pycache__`, `*.pyc`), test artifacts, and zip archives from git history.
  - Updated `notebooks/colab_run.ipynb` Cell 2 to directly clone `https://github.com/Zeeshaan-23/Gen-AI_Final.git` and `%cd Gen-AI_Final`.
  - Comprehensive `.gitignore` configuration preventing tracking of binary checkpoints, virtual environments, and caches.

---

### Phase 1–5 Initial Upload & Scaffolding
- **Commit**: `bdd83f4`
- **Date**: `2026-11-24 14:04:19 +0530`
- **Author**: Zeeshaan-23
- **Summary**: `Initial project upload`
- **Key Deliverables**:
  - Core inpainting diffusion engine with lazy loading and CPU fallback (`src/engine.py`).
  - Text-prompted zero-shot component masking with CLIPSeg (`CIDAS/clipseg-rd64-refined`).
  - Batch catalog processing pipeline, status tracking, mask coverage cutoff (<0.5%), and automatic retry (`src/batch.py`).
  - Dual-tab Gradio web interface (`app.py`).
  - Spec-aware prompt templates and 12 furniture presets (`src/prompts.py`).
  - Outside-mask SSIM quality evaluation metric (`src/evaluate.py`).
  - Comprehensive unit test suite in `tests/`.

---

## 2. Commit Quality Review & Future Improvements

### Current Commit Quality Assessment
- **Imperative Mood**: Commit titles consistently use imperative present tense (`Add...`, `Fix...`, `Clean...`), aligning with Git best practices.
- **Granularity & Separation of Concerns**: Each major milestone represents a coherent functional feature (ControlNet, Preprocessing, LSTM Text Generation) rather than fragmented micro-edits.
- **Traceability**: Commit logs link directly to phase documentation in `PLAN.md` and `PROGRESS.md`.

### Recommendations for Future Commits
1. **Conventional Commits Specification**: Adopt standardized prefix conventions:
   - `feat:` for new capabilities (e.g. `feat(text): add character-level LSTM language model`)
   - `fix:` for bug fixes (e.g. `fix(engine): remove VAE float32 upcast to resolve fp16 bias crash`)
   - `test:` for test additions and fixtures
   - `docs:` for markdown reports and user guides
   - `refactor:` for code restructuring without behavioral changes
2. **Body Context & Breaking Changes**: Include bullet points in the commit body referencing specific PRs, issue numbers, or constraints.
3. **Automated Pre-Commit Hooks**: Configure pre-commit hooks to run `pyflakes`, `pytest -q`, and schema validators automatically prior to staging commits.
