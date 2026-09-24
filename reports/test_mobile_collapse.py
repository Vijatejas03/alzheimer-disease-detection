import time
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "https://alzheimer-xai-vijay.streamlit.app"
iphone_ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(
        viewport={"width": 390, "height": 844},
        device_scale_factor=2,
        is_mobile=True,
        has_touch=True,
        user_agent=iphone_ua
    )
    page = context.new_page()
    page.goto(URL, timeout=45000)
    page.wait_for_timeout(6000)
    
    # 1. Screenshot initial mobile load
    page.screenshot(path="reports/mobile_initial.png")
    print("Saved reports/mobile_initial.png")
    
    # 2. Find and click the collapse sidebar button in the app frame
    app_frame = [f for f in page.frames if "/~/+/" in f.url][0]
    # Button with keyboard_double_arrow_left
    collapse_btn = app_frame.locator("button:has-text('keyboard_double_arrow_left')")
    print("Collapse btn count:", collapse_btn.count())
    if collapse_btn.count() > 0:
        collapse_btn.first.click()
        page.wait_for_timeout(2000)
        page.screenshot(path="reports/mobile_sidebar_collapsed.png")
        print("Saved reports/mobile_sidebar_collapsed.png")
        
    browser.close()
