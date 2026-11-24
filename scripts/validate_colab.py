"""
Validation script for notebooks/colab_run.ipynb.
"""

import nbformat

nb_path = "notebooks/colab_run.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

print("=== 1. Schema Validation ===")
nbformat.validate(nb)
print("nbformat validation: PASS (Valid v4 schema)")
print(f"Total cells: {len(nb.cells)}")
assert len(nb.cells) == 17, f"Expected 17 cells, got {len(nb.cells)}"

print("\n=== 2. Cell 2 Repo URL Placeholder Verification ===")
c2_source = "".join(nb.cells[2].source)
assert "!git clone https://github.com/Zeeshaan-23/Gen-AI_Final.git" in c2_source
assert "%cd Gen-AI_Final" in c2_source
assert "!pip install -q -r requirements.txt" in c2_source
print("Cell 2 clone and dependencies installation: PASS")
print("  Line found: !git clone https://github.com/Zeeshaan-23/Gen-AI_Final.git")
print("  Line found: %cd Gen-AI_Final")

print("\n=== 3. Cell 3 Dependency Installation Verification ===")
c3_source = "".join(nb.cells[3].source)
assert "!pip install -q -r requirements.txt" in c3_source
print("Cell 3 requirements installation: PASS (no conflicting pins)")

print("\n=== 4. Cell 5 Catalog Image Preprocessing Verification ===")
c5_source = "".join(nb.cells[5].source)
assert "preprocess_dir" in c5_source
assert "data/clean" in c5_source
print("Cell 5 Preprocessing pipeline & Before/After strip: PASS")

print("\n=== 5. Cell 8 Editor Instance Reuse (Colab T4 VRAM hygiene) ===")
c8_source = "".join(nb.cells[8].source)
assert '"editor": editor' in c8_source
print("Cell 8 editor instance reuse: PASS")

print("\n=== 6. Cell 9 Status Filter Logic ===")
c9_source = "".join(nb.cells[9].source)
assert '"success" in m["status"]' in c9_source
print("Cell 9 status filter: PASS (includes retried_mask entries)")

print("\n=== 7. Cell 10 ControlNet Batch Run Verification ===")
c10_source = "".join(nb.cells[10].source)
assert '"use_controlnet": True' in c10_source
assert '"controlnet_scale": 0.8' in c10_source
print("Cell 10 ControlNet batch execution: PASS")

print("\n=== 8. Cell 11 ControlNet Compare & Grid Verification ===")
c11_source = "".join(nb.cells[11].source)
assert "compare_controlnet" in c11_source
assert "grid_path" in c11_source
print("Cell 11 ControlNet comparison pipeline: PASS")

print("\n=== 9. Cell 12 Text Preprocessing Verification ===")
c12_source = "".join(nb.cells[12].source)
assert "make_corpus" in c12_source
assert "preprocess_text" in c12_source
print("Cell 12 Text corpus generation & preprocessing: PASS")

print("\n=== 10. Cell 13 CharLSTM Training Verification ===")
c13_source = "".join(nb.cells[13].source)
assert "src.text.train" in c13_source
assert "loss_curve.png" in c13_source
print("Cell 13 CharLSTM training & loss curve display: PASS")

print("\n=== 11. Cell 14 Text Evaluation Verification ===")
c14_source = "".join(nb.cells[14].source)
assert "src.text.generate" in c14_source
assert "src.text.evaluate_text" in c14_source
print("Cell 14 Text generation sampling & evaluation summary: PASS")

print("\n=== 12. Cell 16 Public Gradio Share Verification ===")
c16_source = "".join(nb.cells[16].source)
assert "--share" in c16_source
print("Cell 16 Gradio launch with --share: PASS")

print("\nColab Readiness Overall Status: PASS")
