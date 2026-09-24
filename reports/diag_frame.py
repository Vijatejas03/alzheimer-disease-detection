import sys
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    print("Navigating to https://alzheimer-xai-vijay.streamlit.app...")
    page.goto("https://alzheimer-xai-vijay.streamlit.app", timeout=60000)
    page.wait_for_timeout(8000)
    
    print(f"Total frames: {len(page.frames)}")
    for i, f in enumerate(page.frames):
        print(f"Frame {i}: url={f.url}")
        
    app_frames = [f for f in page.frames if "/~/+/" in f.url or "streamlit" in f.url and f != page.main_frame]
    print(f"App frames matched: {len(app_frames)}")
    if app_frames:
        f = app_frames[0]
        labels = f.locator("label").all()
        print(f"Labels found in frame ({len(labels)}):")
        for l in labels:
            try:
                print(f"  - {l.inner_text()}")
            except Exception as e:
                print(f"  - err: {e}")
    browser.close()
