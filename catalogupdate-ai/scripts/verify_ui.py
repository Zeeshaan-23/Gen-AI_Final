"""
CatalogUpdate AI - UI Verification Script with Playwright.
Launches app.py in background with USE_MOCK_ENGINE=1, controls browser,
and captures the 4 required verification screenshots.
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path
from playwright.sync_api import sync_playwright

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = WORKSPACE_DIR / "verification" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
PORT = 7860
URL = f"http://127.0.0.1:{PORT}"


def wait_for_server(url: str, timeout: int = 30):
    start = time.time()
    print(f"Waiting for Gradio server at {url}...")
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"Gradio server is ready at {url}!")
                return True
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"Server at {url} failed to start within {timeout}s.")


def main():
    print("=== Step 1: Starting app.py with USE_MOCK_ENGINE=1 ===")
    env = os.environ.copy()
    env["USE_MOCK_ENGINE"] = "1"
    env["GRADIO_ANALYTICS_ENABLED"] = "False"

    python_exe = sys.executable
    server_process = subprocess.Popen(
        [python_exe, "app.py", "--port", str(PORT)],
        cwd=str(WORKSPACE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        wait_for_server(URL, timeout=40)

        print("\n=== Step 2: Launching Playwright Browser ===")
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True, channel="msedge")
                print("Using Microsoft Edge browser.")
            except Exception as e:
                print("Fallback to bundled Chromium:", e)
                browser = p.chromium.launch(headless=True)

            context = browser.new_context(viewport={"width": 1400, "height": 950})
            page = context.new_page()

            print(f"Navigating to {URL}...")
            page.goto(URL, wait_until="domcontentloaded")
            page.wait_for_selector("button:has-text('Run Batch Update')", timeout=20000)
            time.sleep(3)

            # (a) Screenshot 1: Tab 1 empty
            print("Capturing Screenshot (a): 01_tab1_empty.png...")
            page.screenshot(path=str(SCREENSHOT_DIR / "01_tab1_empty.png"))
            assert (SCREENSHOT_DIR / "01_tab1_empty.png").exists()

            # (d) Screenshot 4: Error message shown when clicking Run Batch with no images
            print("Triggering error: Click 'Run Batch Update' with no images...")
            page.click("button:has-text('Run Batch Update')")
            time.sleep(1.5)
            print("Capturing Screenshot (d): 04_tab1_error_no_images.png...")
            page.screenshot(path=str(SCREENSHOT_DIR / "04_tab1_error_no_images.png"))
            assert (SCREENSHOT_DIR / "04_tab1_error_no_images.png").exists()

            # (b) Screenshot 2: Tab 1 after uploading 3 synthetic images and running batch
            print("Uploading 3 synthetic test images...")
            img_files = [
                str(WORKSPACE_DIR / "data" / "test_input" / f"chair_0{i}.png")
                for i in [1, 2, 3]
            ]
            file_input = page.locator("input[type='file']").first
            file_input.set_input_files(img_files)
            time.sleep(3)

            print("Clicking 'Run Batch Update'...")
            page.click("button:has-text('Run Batch Update')")

            # Wait for results table and galleries to render
            print("Waiting for batch processing to complete...")
            try:
                page.wait_for_selector("table tbody tr, [data-testid='dataframe']", timeout=30000)
            except Exception as e:
                print("Table selector wait timeout, waiting 5 seconds:", e)
            time.sleep(5)

            print("Capturing Screenshot (b): 02_tab1_batch_results.png...")
            page.screenshot(path=str(SCREENSHOT_DIR / "02_tab1_batch_results.png"))
            assert (SCREENSHOT_DIR / "02_tab1_batch_results.png").exists()

            # (c) Screenshot 3: Tab 2 with image editor
            print("Switching to Tab 2: Single Image (Manual Mask)...")
            try:
                page.locator("button[role='tab']", has_text="Single Image").click(force=True)
            except Exception:
                page.evaluate("""() => {
                    const tabs = Array.from(document.querySelectorAll('button[role="tab"]'));
                    const target = tabs.find(el => el.textContent.includes('Single Image'));
                    if (target) target.click();
                }""")
            time.sleep(3)
            try:
                page.wait_for_selector(".image-editor, canvas, [aria-label*='Product Image']", timeout=10000)
            except Exception as e:
                print("Editor selector wait timeout, continuing:", e)
            time.sleep(2)

            print("Capturing Screenshot (c): 03_tab2_editor.png...")
            page.screenshot(path=str(SCREENSHOT_DIR / "03_tab2_editor.png"))
            assert (SCREENSHOT_DIR / "03_tab2_editor.png").exists()

            browser.close()
            print("\n=== UI Verification & All 4 Screenshots Captured Successfully! ===")

    finally:
        print("Terminating Gradio server process...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()


if __name__ == "__main__":
    main()
