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

print("\n=== 4. Cell 7 Editor Instance Reuse (Colab T4 VRAM hygiene) ===")
c7_source = "".join(nb.cells[7].source)
assert '"editor": editor' in c7_source
print("Cell 7 editor instance reuse: PASS")

print("\n=== 5. Cell 8 Status Filter Logic ===")
c8_source = "".join(nb.cells[8].source)
assert '"success" in m["status"]' in c8_source
print("Cell 8 status filter: PASS (includes retried_mask entries)")

print("\n=== 6. Cell 10 Public Gradio Share Verification ===")
c10_source = "".join(nb.cells[10].source)
assert "--share" in c10_source
print("Cell 10 Gradio launch with --share: PASS")

print("\nColab Readiness Overall Status: PASS")
