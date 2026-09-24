import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")

SAMPLE_MRI = Path("data/test_samples/mild_33.jpg").resolve()
print("Sample MRI path:", SAMPLE_MRI, "exists:", SAMPLE_MRI.exists())

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto("https://alzheimer-xai-vijay.streamlit.app/~/+/", timeout=60000)
    page.wait_for_timeout(6000)
    
    # 1. Click MRI Analysis
    print("Clicking MRI Analysis...")
    page.locator("label:has-text('MRI Analysis')").click()
    page.wait_for_timeout(3000)
    
    # 2. Check input file element
    file_input = page.locator("input[type='file']")
    print("File input count:", file_input.count())
    if file_input.count() > 0:
        print("Uploading sample MRI...")
        file_input.set_input_files(str(SAMPLE_MRI))
        page.wait_for_timeout(8000)
        
        # Check text on page
        text = page.locator("body").inner_text()
        print("Has 'Input validation passed':", "INPUT VALIDATION PASSED" in text or "passed" in text.lower())
        print("Has 'MODEL PREDICTION':", "MODEL PREDICTION" in text or "Prediction" in text)
        print("Has 'Grad-CAM':", "Grad-CAM" in text or "GRAD-CAM" in text)
        
        # Take a screenshot to verify
        page.screenshot(path="reports/screenshots/test_live_upload.png", full_page=False)
        print("Screenshot saved to reports/screenshots/test_live_upload.png")
    else:
        # Check if 'Load Curated Research Sample' radio exists
        print("Looking for Curated sample radio...")
        sample_radio = page.locator("label:has-text('Load Curated Research Sample')")
        if sample_radio.count() > 0:
            sample_radio.click()
            page.wait_for_timeout(5000)
            text = page.locator("body").inner_text()
            print("Curated mode loaded text len:", len(text))
            page.screenshot(path="reports/screenshots/test_live_curated.png", full_page=False)
            print("Screenshot saved to reports/screenshots/test_live_curated.png")
            
    browser.close()
