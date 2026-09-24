"""
capture_screenshots.py
Uses Playwright to open the REAL live Streamlit application and
capture actual screenshots at each step of the user journey.

Run:
    python reports/capture_screenshots.py
Output:
    reports/screenshots/  (real app screenshots, used by assemble_tutorial_video.py)
"""

import sys, time, shutil
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
SHOT_DIR   = BASE / "reports" / "screenshots"
SAMPLE_MRI = BASE / "data" / "test_samples" / "non_1263.jpg"
LIVE_URL   = "https://alzheimer-xai-vijay.streamlit.app"

SHOT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def wait_for_streamlit(page, timeout=30000):
    """Wait until Streamlit's spinner disappears (app fully loaded)."""
    try:
        # Wait for the stSpinner to disappear, or just wait a moment
        page.wait_for_load_state("networkidle", timeout=timeout)
    except PWTimeout:
        pass
    time.sleep(2)  # extra buffer for Streamlit's re-render


def scroll_smooth(page, direction="down", amount=300):
    """Smoothly scroll the page."""
    page.evaluate(f"window.scrollBy({{top: {amount if direction=='down' else -amount}, behavior: 'smooth'}})")
    time.sleep(0.8)


def move_mouse_to(page, selector, pause=0.5):
    """Move mouse to a visible element."""
    try:
        el = page.locator(selector).first
        el.scroll_into_view_if_needed()
        el.hover()
        time.sleep(pause)
    except Exception:
        pass


def shot(page, name, caption=""):
    """Take a screenshot and save it."""
    path = SHOT_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=False)
    print(f"  [shot] {name}.png  {caption}")
    return path


def click_nav(page, label):
    """Click a Streamlit sidebar navigation item by its text."""
    # Streamlit uses radio buttons or anchors for navigation
    for selector in [
        f"[data-testid='stSidebarNavItems'] >> text={label}",
        f"[data-testid='stSidebarContent'] >> text={label}",
        f"section[data-testid='stSidebar'] >> text={label}",
        f"text={label}",
    ]:
        try:
            el = page.locator(selector).first
            if el.is_visible(timeout=3000):
                el.click()
                time.sleep(3)
                wait_for_streamlit(page, 15000)
                return True
        except Exception:
            continue
    print(f"  [warn] Could not find nav item: {label}")
    return False


# ──────────────────────────────────────────────
# MAIN CAPTURE SEQUENCE
# ──────────────────────────────────────────────

def capture_all():
    with sync_playwright() as p:
        print("Launching Chromium browser...")
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--force-device-scale-factor=1",
            ]
        )

        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1,
        )
        page = ctx.new_page()

        # ── STEP 1: Open URL ──────────────────────────────
        print("\n[Step 1] Opening live application...")
        page.goto(LIVE_URL, wait_until="domcontentloaded", timeout=60000)
        time.sleep(2)
        shot(page, "01a_browser_url", "Browser showing URL before load")
        wait_for_streamlit(page, 45000)
        time.sleep(3)
        shot(page, "01b_home_loaded", "App fully loaded - home page")

        # ── STEP 2: Explore home page ────────────────────
        print("[Step 2] Exploring home page...")
        time.sleep(1)
        shot(page, "02a_home_overview", "Home page overview")
        scroll_smooth(page, "down", 400)
        shot(page, "02b_home_scrolled", "Home page scrolled down")
        scroll_smooth(page, "up", 400)

        # ── STEP 3: Navigate to MRI Analysis ─────────────
        print("[Step 3] Navigating to MRI Analysis...")
        nav_ok = click_nav(page, "MRI Analysis")
        if not nav_ok:
            # Try clicking directly on any sidebar link
            try:
                page.click("text=MRI Analysis", timeout=10000)
                time.sleep(3)
                wait_for_streamlit(page, 20000)
            except Exception as e:
                print(f"  [warn] Nav click failed: {e}")
        shot(page, "03a_mri_page", "MRI Analysis page loaded")

        # ── STEP 4: Select/Upload MRI ─────────────────────
        print("[Step 4] Handling MRI input...")
        # Look for the "Load Curated Research Sample" option
        for label in ["Load Curated Research Sample", "Curated Research Sample",
                       "load curated", "demo"]:
            try:
                el = page.locator(f"text={label}").first
                if el.is_visible(timeout=3000):
                    el.click()
                    time.sleep(2)
                    print(f"  Clicked: {label}")
                    break
            except Exception:
                continue

        wait_for_streamlit(page, 15000)
        shot(page, "04a_mri_input_selected", "MRI input mode selected")

        # Try to use file uploader if visible
        file_uploader = page.locator("[data-testid='stFileUploaderDropzone']").first
        try:
            if file_uploader.is_visible(timeout=3000):
                print("  File uploader visible — uploading MRI sample...")
                file_uploader.set_input_files(str(SAMPLE_MRI))
                time.sleep(4)
                wait_for_streamlit(page, 20000)
                shot(page, "04b_mri_uploaded", "MRI image uploaded and previewed")
        except Exception as e:
            print(f"  [info] File uploader: {e}")

        # Look for a sample selector (selectbox or radio)
        try:
            # Streamlit selectbox
            selects = page.locator("select").all()
            for sel in selects:
                if sel.is_visible(timeout=2000):
                    sel.select_option(index=0)
                    time.sleep(2)
                    break
        except Exception:
            pass

        wait_for_streamlit(page, 15000)
        shot(page, "04c_mri_with_sample", "MRI sample displayed")
        scroll_smooth(page, "down", 300)
        shot(page, "04d_mri_scrolled", "MRI analysis section scrolled")

        # ── STEP 5: Model selection ───────────────────────
        print("[Step 5] Model selection...")
        scroll_smooth(page, "up", 600)
        # Look for model radio buttons
        for model_name in ["EfficientNet-B0", "ResNet-18", "MobileNetV2"]:
            try:
                el = page.locator(f"text={model_name}").first
                if el.is_visible(timeout=2000):
                    el.click()
                    time.sleep(1.5)
                    print(f"  Selected model: {model_name}")
                    break
            except Exception:
                continue
        shot(page, "05a_model_selected", "Model selected")

        # ── STEP 6: Run Analysis (if button exists) ───────
        print("[Step 6] Attempting to run analysis...")
        for btn_text in ["Analyze", "Run Analysis", "Analyze MRI",
                          "Run", "Submit", "Predict"]:
            try:
                btn = page.locator(f"button:has-text('{btn_text}')").first
                if btn.is_visible(timeout=2000):
                    btn.click()
                    print(f"  Clicked button: {btn_text}")
                    time.sleep(2)
                    wait_for_streamlit(page, 30000)
                    break
            except Exception:
                continue

        shot(page, "06a_after_analyze", "After analysis triggered")
        scroll_smooth(page, "down", 300)
        shot(page, "06b_results_area", "Results area visible")

        # ── STEP 7: Prediction results ────────────────────
        print("[Step 7] Capturing prediction results...")
        scroll_smooth(page, "down", 400)
        shot(page, "07a_prediction", "Prediction result shown")
        scroll_smooth(page, "down", 400)
        shot(page, "07b_probabilities", "Class probabilities shown")

        # ── STEP 8: Grad-CAM ──────────────────────────────
        print("[Step 8] Capturing Grad-CAM...")
        scroll_smooth(page, "down", 500)
        shot(page, "08a_gradcam_area", "Grad-CAM area")
        # Look for Grad-CAM tab or section
        for label in ["Grad-CAM", "Explainability", "grad_cam", "GRAD-CAM"]:
            try:
                el = page.locator(f"text={label}").first
                if el.is_visible(timeout=2000):
                    el.click()
                    time.sleep(2)
                    wait_for_streamlit(page, 10000)
                    shot(page, "08b_gradcam_clicked", "Grad-CAM section opened")
                    break
            except Exception:
                continue
        scroll_smooth(page, "down", 400)
        shot(page, "08c_gradcam_display", "Grad-CAM displayed")

        # ── STEP 9: Model Comparison page ─────────────────
        print("[Step 9] Navigating to Model Comparison...")
        click_nav(page, "Model Comparison")
        shot(page, "09a_model_comparison", "Model Comparison page")
        scroll_smooth(page, "down", 500)
        shot(page, "09b_comparison_charts", "Model comparison charts")

        # ── STEP 10: Evaluation page ──────────────────────
        print("[Step 10] Navigating to Evaluation...")
        click_nav(page, "Evaluation")
        shot(page, "10a_evaluation", "Evaluation results page")
        scroll_smooth(page, "down", 500)
        shot(page, "10b_evaluation_charts", "Evaluation charts")

        # ── STEP 11: Error & Robustness ───────────────────
        print("[Step 11] Error & Robustness page...")
        click_nav(page, "Error & Robustness")
        shot(page, "11a_robustness", "Error & Robustness page")

        # ── STEP 12: Methodology ──────────────────────────
        print("[Step 12] Methodology page...")
        click_nav(page, "Methodology")
        shot(page, "12a_methodology", "Methodology page")

        # ── STEP 13: About ────────────────────────────────
        print("[Step 13] About page...")
        click_nav(page, "About")
        shot(page, "13a_about", "About page")

        # ── STEP 14: Back to MRI Analysis (final recap) ───
        print("[Step 14] Final — back to MRI Analysis...")
        click_nav(page, "MRI Analysis")
        shot(page, "14a_final_mri", "Final MRI page recap")
        scroll_smooth(page, "up", 999)
        shot(page, "14b_final_top", "Final top of MRI page")

        browser.close()
        print(f"\nCapture complete. Screenshots saved to: {SHOT_DIR}")
        shots = sorted(SHOT_DIR.glob("*.png"))
        print(f"Total screenshots: {len(shots)}")
        for s in shots:
            print(f"  {s.name}  ({s.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    capture_all()
