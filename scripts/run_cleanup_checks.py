"""
Run all Step 7 Cleanup & Verification Checks and log to verification/logs/cleanup_after.txt.
"""

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
LOG_FILE = WORKSPACE / "verification" / "logs" / "cleanup_after.txt"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg=""):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run(cmd, env_extra=None, capture=True):
    log(f"$ {cmd}")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    res = subprocess.run(
        cmd,
        cwd=str(WORKSPACE),
        shell=True,
        capture_output=capture,
        text=True,
        env=env,
    )
    if res.stdout:
        log(res.stdout.strip())
    if res.stderr:
        log(res.stderr.strip())
    log(f"Exit code: {res.returncode}")
    if res.returncode != 0:
        log(f"ERROR: Command failed with exit code {res.returncode}")
    return res


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("CatalogUpdate AI - Step 7 Cleanup & Verification Log\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("============================================================\n\n")

    log("--- CHECK 1: Clean Imports ---")
    r1 = run('python -c "import src.engine, src.batch, src.prompts, src.evaluate, app; print(\'Clean import: PASS\')"')
    assert r1.returncode == 0, "Check 1 Failed"

    log("\n--- CHECK 2: Static Analysis (Pyflakes) ---")
    r2 = run("python -m pyflakes app.py src tests scripts")
    assert r2.returncode == 0, "Check 2 Failed"
    log("Pyflakes static analysis: PASS (0 errors)")

    log("\n--- CHECK 3: Unit Tests (pytest) ---")
    r3 = run("pytest -q")
    assert r3.returncode == 0, "Check 3 Failed"

    log("\n--- CHECK 4: Colab Notebook Schema & Setup Validation ---")
    r4a = run("python -c \"import nbformat; nb = nbformat.read('notebooks/colab_run.ipynb', as_version=4); nbformat.validate(nb); print('Notebook schema valid: PASS')\"")
    assert r4a.returncode == 0, "Check 4a Failed"
    r4b = run("python scripts/validate_colab.py")
    assert r4b.returncode == 0, "Check 4b Failed"

    log("\n--- CHECK 5: Mock Batch CLI Run ---")
    test_out = WORKSPACE / "data" / "output_test"
    if test_out.exists():
        shutil.rmtree(test_out)
    r5 = run(
        'python -m src.batch --input data/input --output data/output_test --part "chair legs" --prompt "walnut wood" --mock',
        env_extra={"USE_MOCK_ENGINE": "1"},
    )
    assert r5.returncode == 0, "Check 5 Failed"
    log("Generated files in data/output_test:")
    for p in sorted(test_out.rglob("*")):
        if p.is_file():
            log(f"  {p.relative_to(WORKSPACE)} ({p.stat().st_size} bytes)")
    shutil.rmtree(test_out)
    log("Cleaned up data/output_test: PASS")

    log("\n--- CHECK 6: Git Status Check ---")
    run("git status --short")

    log("\n============================================================")
    log("ALL VERIFICATION CHECKS PASSED")
    log("============================================================")


if __name__ == "__main__":
    main()
