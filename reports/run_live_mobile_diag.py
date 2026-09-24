import sys, json, time
from pathlib import Path
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")

URL = "https://alzheimer-xai-vijay.streamlit.app"

def test_viewport(device_name, width, height, user_agent=None, is_mobile=False):
    print(f"\n==================================================")
    print(f"TESTING: {device_name} ({width}x{height}, mobile={is_mobile})")
    print(f"==================================================")
    
    console_logs = []
    failed_requests = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context_args = {
            "viewport": {"width": width, "height": height},
            "device_scale_factor": 2 if is_mobile else 1,
            "is_mobile": is_mobile,
            "has_touch": is_mobile,
        }
        if user_agent:
            context_args["user_agent"] = user_agent
            
        context = browser.new_context(**context_args)
        page = context.new_page()
        
        page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))
        page.on("requestfailed", lambda req: failed_requests.append(f"{req.method} {req.url} -> {req.failure}"))
        page.on("response", lambda resp: failed_requests.append(f"{resp.status} {resp.url}") if resp.status >= 400 else None)
        
        try:
            print(f"Navigating to {URL}...")
            resp = page.goto(URL, timeout=45000, wait_until="domcontentloaded")
            print(f"HTTP response status: {resp.status if resp else 'None'}")
            
            # Wait for content to stabilize
            time.sleep(10)
            
            title = page.title()
            print(f"Page title: {title}")
            
            # Check frames
            print(f"Frames count: {len(page.frames)}")
            for i, f in enumerate(page.frames):
                print(f"  Frame {i}: url={f.url[:80]} title={f.title() if hasattr(f, 'title') else ''}")
                
            # Screenshot main page
            out_img = Path(f"reports/diag_{device_name.lower().replace(' ', '_')}.png")
            page.screenshot(path=str(out_img), full_page=False)
            print(f"Screenshot saved to: {out_img} ({out_img.stat().st_size} bytes)")
            
            # Inspect body text or inner HTML
            body_text = page.inner_text("body")
            print(f"Outer page body text length: {len(body_text)}")
            print(f"Outer page body text snippet:\n{body_text[:400]}")
            
            # If there's an iframe, inspect it
            for i, f in enumerate(page.frames):
                if f != page.main_frame:
                    try:
                        f_text = f.inner_text("body")
                        print(f"Frame {i} body text length: {len(f_text)}")
                        print(f"Frame {i} text snippet:\n{f_text[:300]}")
                    except Exception as fe:
                        print(f"Frame {i} error getting text: {fe}")
                        
        except Exception as e:
            print(f"Navigation/Testing error: {e}")
            out_img = Path(f"reports/diag_{device_name.lower().replace(' ', '_')}_error.png")
            try:
                page.screenshot(path=str(out_img))
                print(f"Error screenshot saved to {out_img}")
            except:
                pass
                
        browser.close()
        
    print(f"\nConsole logs count: {len(console_logs)}")
    for log in console_logs[-15:]:
        print(f"  {log}")
        
    print(f"\nFailed/Error requests count: {len(failed_requests)}")
    for req in failed_requests[:15]:
        print(f"  {req}")

if __name__ == "__main__":
    # Test 1: Desktop
    test_viewport("Desktop", 1920, 1080)
    
    # Test 2: iPhone 14/15/16 (390x844)
    iphone_ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1"
    test_viewport("iPhone 14", 390, 844, user_agent=iphone_ua, is_mobile=True)
    
    # Test 3: Android Pixel (412x915)
    pixel_ua = "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36"
    test_viewport("Android Pixel", 412, 915, user_agent=pixel_ua, is_mobile=True)
