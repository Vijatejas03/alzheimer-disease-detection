import time
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "https://alzheimer-xai-vijay.streamlit.app"
iphone_ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1"

print("Waiting 15 seconds for Streamlit Community Cloud to refresh from git commit...")
time.sleep(15)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    
    # 1. Desktop verification
    print("\n[VERIFY 1] Desktop (1920x1080)...")
    d_ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    d_page = d_ctx.new_page()
    d_page.goto(URL, timeout=50000)
    d_page.wait_for_timeout(8000)
    d_page.screenshot(path="reports/verify_desktop_fixed.png")
    d_body = d_page.inner_text("body")
    print("Desktop title:", d_page.title())
    print("Desktop screenshot saved to reports/verify_desktop_fixed.png")
    d_ctx.close()
    
    # 2. Mobile iPhone verification
    print("\n[VERIFY 2] Mobile iPhone 14 (390x844)...")
    m_ctx = browser.new_context(
        viewport={"width": 390, "height": 844},
        device_scale_factor=2,
        is_mobile=True,
        has_touch=True,
        user_agent=iphone_ua
    )
    m_page = m_ctx.new_page()
    m_page.goto(URL, timeout=50000)
    m_page.wait_for_timeout(8000)
    m_page.screenshot(path="reports/verify_mobile_fixed.png")
    print("Mobile title:", m_page.title())
    print("Mobile screenshot saved to reports/verify_mobile_fixed.png")
    
    # Check if app frame has hero title visible
    frames = [f for f in m_page.frames if "/~/+/" in f.url]
    if frames:
        f = frames[0]
        text = f.inner_text("body")
        print("Has 'Explainable Alzheimer's MRI Analysis':", "Explainable Alzheimer's MRI Analysis" in text)
        print("Has 'Analyze MRI Scan':", "Analyze MRI Scan" in text)
        
        # Test clicking MRI Analysis
        print("Testing navigation to MRI Analysis on mobile...")
        # Open sidebar if collapsed
        open_btn = f.locator("button:has-text('keyboard_double_arrow_right')")
        if open_btn.count() > 0:
            print("Clicking open sidebar button...")
            open_btn.first.click()
            m_page.wait_for_timeout(1500)
            
        mri_nav = f.locator("label:has-text('MRI Analysis')")
        if mri_nav.count() > 0:
            mri_nav.first.click()
            m_page.wait_for_timeout(5000)
            m_page.screenshot(path="reports/verify_mobile_mri_analysis.png")
            print("Mobile MRI Analysis screenshot saved to reports/verify_mobile_mri_analysis.png")
            mri_text = f.inner_text("body")
            print("MRI page loaded text check:", "Input Source" in mri_text or "Curated" in mri_text or "Upload" in mri_text)
            
    m_ctx.close()
    browser.close()
    print("\nVerification script finished.")
