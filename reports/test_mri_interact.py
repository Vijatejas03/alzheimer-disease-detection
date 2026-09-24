import sys, time
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    print("Loading app frame directly...")
    page.goto("https://alzheimer-xai-vijay.streamlit.app/~/+/", timeout=60000)
    page.wait_for_timeout(6000)
    print("Page title:", page.title())
    
    # Click MRI Analysis
    mri_label = page.locator("label:has-text('MRI Analysis')")
    print("MRI Analysis locator count:", mri_label.count())
    mri_label.click()
    page.wait_for_timeout(4000)
    
    print("MRI Page loaded!")
    # Check what controls exist
    radios = page.locator("div[role='radiogroup']").all()
    print("Radiogroups on MRI page:", len(radios))
    for r in radios:
        print("  Radio:", r.inner_text().replace("\n", " | "))
        
    buttons = page.locator("button").all()
    print("Buttons on MRI page:", len(buttons))
    for b in buttons:
        print("  Btn:", b.inner_text())
        
    browser.close()
