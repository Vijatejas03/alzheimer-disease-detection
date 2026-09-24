"""
create_real_tutorial_video.py
Full automated live screen recording of https://alzheimer-xai-vijay.streamlit.app
Generates reports/Alzheimer_Live_App_User_Tutorial.mp4
"""

import sys, os, time, math, io
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent.parent
SAMPLE_MRI = BASE / "data" / "test_samples" / "mild_33.jpg"
OUT_MP4    = BASE / "reports" / "Alzheimer_Live_App_User_Tutorial.mp4"
TMP_DIR    = BASE / "reports" / "tutorial_frames"

TMP_DIR.mkdir(parents=True, exist_ok=True)
OUT_MP4.parent.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# VIDEO DIMENSIONS & SETTINGS
# ──────────────────────────────────────────────
VIDEO_W = 1920
VIDEO_H = 1080
BROWSER_HEADER_H = 82
VIEWPORT_W = 1920
VIEWPORT_H = VIDEO_H - BROWSER_HEADER_H  # 998 px
FPS = 25

LIVE_URL = "https://alzheimer-xai-vijay.streamlit.app"

# ──────────────────────────────────────────────
# FONTS
# ──────────────────────────────────────────────
def get_font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()

FONT_TITLE = get_font(20, bold=True)
FONT_URL   = get_font(18, bold=False)
FONT_STEP  = get_font(22, bold=True)
FONT_CAP   = get_font(28, bold=True)
FONT_SUB   = get_font(20, bold=False)
FONT_RECAP = get_font(34, bold=True)
FONT_RECAP_SUB = get_font(22, bold=False)

# ──────────────────────────────────────────────
# BROWSER HEADER CHROME RENDERER
# ──────────────────────────────────────────────
def draw_browser_chrome(url_text=LIVE_URL, active_tab="🧠 Alzheimer's XAI Research"):
    """Render a modern desktop browser window header (tabs + address bar)."""
    header = Image.new("RGB", (VIDEO_W, BROWSER_HEADER_H), (241, 243, 244))
    d = ImageDraw.Draw(header)

    # Window control buttons (close, minimize, maximize) on top right
    cx = VIDEO_W - 135
    d.rectangle([cx, 0, cx + 45, 34], fill=(241, 243, 244))  # min
    d.line([(cx + 17, 18), (cx + 28, 18)], fill=(90, 90, 90), width=1)
    d.rectangle([cx + 45, 0, cx + 90, 34], fill=(241, 243, 244))  # max
    d.rectangle([cx + 62, 13, cx + 73, 23], outline=(90, 90, 90), width=1)
    d.rectangle([cx + 90, 0, cx + 135, 34], fill=(241, 243, 244))  # close
    d.line([(cx + 107, 13), (cx + 117, 23)], fill=(90, 90, 90), width=1)
    d.line([(cx + 107, 23), (cx + 117, 13)], fill=(90, 90, 90), width=1)

    # Active Tab
    tab_w = 340
    tab_h = 34
    d.rounded_rectangle([80, 6, 80 + tab_w, 6 + tab_h], radius=8, fill=(255, 255, 255))
    d.text((96, 12), active_tab, font=FONT_TITLE, fill=(30, 41, 59))
    d.line([(80 + tab_w - 22, 17), (80 + tab_w - 14, 25)], fill=(120, 120, 120), width=1)
    d.line([(80 + tab_w - 22, 25), (80 + tab_w - 14, 17)], fill=(120, 120, 120), width=1)

    # Navigation buttons (Back, Forward, Refresh)
    by = 44
    d.text((22, by + 4), "◀", font=FONT_TITLE, fill=(130, 140, 150))
    d.text((54, by + 4), "▶", font=FONT_TITLE, fill=(180, 190, 200))
    d.text((86, by + 4), "↻", font=FONT_TITLE, fill=(100, 110, 120))

    # Address bar
    url_box_x = 120
    url_box_w = VIDEO_W - 240
    d.rounded_rectangle([url_box_x, by, url_box_x + url_box_w, by + 32], radius=16, fill=(255, 255, 255), outline=(218, 220, 224))
    # Security lock icon
    d.text((url_box_x + 14, by + 6), "🔒", font=FONT_URL, fill=(22, 163, 74))
    # URL
    d.text((url_box_x + 40, by + 6), url_text, font=FONT_URL, fill=(30, 41, 59))

    # Bottom border line
    d.line([(0, BROWSER_HEADER_H - 1), (VIDEO_W, BROWSER_HEADER_H - 1)], fill=(226, 232, 240), width=1)
    return header

# ──────────────────────────────────────────────
# MOUSE CURSOR RENDERER
# ──────────────────────────────────────────────
def draw_cursor(img: Image.Image, x: int, y: int, clicking=False, click_progress=0.0):
    """Draw a realistic OS mouse pointer and optional click ripple."""
    d = ImageDraw.Draw(img)

    # Click ripple effect
    if clicking or click_progress > 0.0:
        radius = int(8 + click_progress * 26)
        alpha = int(255 * (1.0 - click_progress))
        if radius > 0:
            d.ellipse([x - radius, y - radius, x + radius, y + radius], outline=(0, 212, 255), width=3)

    # Standard pointer polygon (arrow)
    points = [
        (x, y),
        (x, y + 22),
        (x + 5, y + 17),
        (x + 10, y + 28),
        (x + 14, y + 26),
        (x + 9, y + 15),
        (x + 16, y + 15),
    ]
    # Shadow
    shadow_points = [(px + 2, py + 2) for px, py in points]
    d.polygon(shadow_points, fill=(0, 0, 0))
    # White pointer with dark outline
    d.polygon(points, fill=(255, 255, 255), outline=(15, 23, 42))

# ──────────────────────────────────────────────
# LOWER-THIRD TUTORIAL BANNER
# ──────────────────────────────────────────────
def draw_tutorial_banner(img: Image.Image, step_num: str, headline: str, subtext: str = ""):
    """Render a crisp, modern tutorial banner on the lower third of the screen."""
    overlay = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    banner_h = 88
    by = VIDEO_H - banner_h

    # Semi-transparent dark slate backdrop
    d.rectangle([0, by, VIDEO_W, VIDEO_H], fill=(10, 18, 38, 235))
    # Cyan accent bar on top of the banner
    d.rectangle([0, by, VIDEO_W, by + 4], fill=(0, 212, 255, 255))

    # Step Badge
    if step_num:
        badge_box = d.textbbox((0, 0), step_num, font=FONT_STEP)
        bw = badge_box[2] - badge_box[0] + 28
        d.rounded_rectangle([32, by + 18, 32 + bw, by + 58], radius=8, fill=(0, 136, 204, 255))
        d.text((46, by + 24), step_num, font=FONT_STEP, fill=(255, 255, 255, 255))
        text_x = 32 + bw + 20
    else:
        text_x = 36

    # Headline
    d.text((text_x, by + 16), headline, font=FONT_CAP, fill=(0, 212, 255, 255))

    # Subtext / description
    if subtext:
        d.text((text_x, by + 50), subtext, font=FONT_SUB, fill=(226, 232, 240, 240))

    # Live app link watermark on bottom right
    d.text((VIDEO_W - 350, by + 48), "alzheimer-xai-vijay.streamlit.app", font=FONT_URL, fill=(100, 220, 150, 220))

    img_rgba = img.convert("RGBA")
    composite = Image.alpha_composite(img_rgba, overlay)
    return composite.convert("RGB")

# ──────────────────────────────────────────────
# FRAME COMPOSITOR
# ──────────────────────────────────────────────
def compose_frame(page_screenshot: Image.Image,
                  header_img: Image.Image,
                  cursor_x: int, cursor_y: int,
                  clicking=False, click_prog=0.0,
                  step_num="", headline="", subtext="",
                  scroll_y=0):
    """
    Combines:
    1. Browser Chrome Header
    2. Real page viewport screenshot (optionally scrolled)
    3. Animated Mouse Cursor with click effects
    4. Lower-third tutorial caption banner
    """
    frame = Image.new("RGB", (VIDEO_W, VIDEO_H), (255, 255, 255))

    # Paste browser header
    frame.paste(header_img, (0, 0))

    # Crop/paste page screenshot into viewport area
    pw, ph = page_screenshot.size
    # Visible viewport height is VIEWPORT_H
    crop_y1 = max(0, min(scroll_y, max(0, ph - VIEWPORT_H)))
    crop_y2 = min(ph, crop_y1 + VIEWPORT_H)
    cropped_page = page_screenshot.crop((0, crop_y1, min(pw, VIEWPORT_W), crop_y2))

    frame.paste(cropped_page, (0, BROWSER_HEADER_H))

    # Draw mouse cursor
    screen_cursor_y = cursor_y + BROWSER_HEADER_H - crop_y1
    draw_cursor(frame, cursor_x, screen_cursor_y, clicking, click_prog)

    # Draw tutorial banner
    if headline:
        frame = draw_tutorial_banner(frame, step_num, headline, subtext)

    return frame

# ──────────────────────────────────────────────
# RECAP SLIDE COMPOSITOR (FINAL SCENE)
# ──────────────────────────────────────────────
def create_recap_frame(step_progress=1.0):
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), (10, 14, 39))
    d = ImageDraw.Draw(img)

    # Top & bottom cyan bars
    d.rectangle([0, 0, VIDEO_W, 8], fill=(0, 212, 255))
    d.rectangle([0, VIDEO_H - 8, VIDEO_W, VIDEO_H], fill=(0, 212, 255))

    # Header
    title = "Complete User Journey Walkthrough"
    d.text((80, 50), title, font=FONT_RECAP, fill=(0, 212, 255))
    d.text((80, 105), "Explainable Deep Learning-Based Multi-Stage Alzheimer's Disease Detection", font=FONT_SUB, fill=(226, 232, 240))

    # Journey Sequence Blocks
    steps = [
        ("01", "Open Link", "https://alzheimer-xai-vijay.streamlit.app"),
        ("02", "Explore Home", "8-Page Navigation, System Architecture"),
        ("03", "MRI Analysis", "Open Analysis workspace in sidebar"),
        ("04", "Upload MRI", "Upload axial T1 brain MRI scan"),
        ("05", "Validation", "Automated 8-stage input quality check"),
        ("06", "Select Model", "ResNet18, EfficientNet-B0, MobileNetV2"),
        ("07", "Run Analysis", "Automated neural network inference"),
        ("08", "Prediction", "4-class stage classification & confidence"),
        ("09", "Grad-CAM", "Visual model attribution heatmaps"),
        ("10", "Comparison", "Held-out test benchmark metrics & ROC"),
    ]

    block_w = 340
    block_h = 100
    for i, (num, name, desc) in enumerate(steps):
        col = i % 2
        row = i // 2
        bx = 80 + col * (VIDEO_W // 2 - 20)
        by = 175 + row * 115

        d.rounded_rectangle([bx, by, bx + 780, by + block_h], radius=10, fill=(18, 30, 66), outline=(0, 180, 240))
        d.rounded_rectangle([bx, by, bx + 80, by + block_h], radius=10, fill=(0, 100, 170))
        d.text((bx + 20, by + 32), num, font=FONT_STEP, fill=(255, 255, 255))
        d.text((bx + 105, by + 18), name, font=FONT_TITLE, fill=(0, 212, 255))
        d.text((bx + 105, by + 52), desc, font=FONT_SUB, fill=(203, 213, 225))

    # Team & Project Footer Card
    card_y = VIDEO_H - 195
    d.rounded_rectangle([80, card_y, VIDEO_W - 80, card_y + 160], radius=12, fill=(13, 27, 62), outline=(34, 197, 94))
    d.text((110, card_y + 20), "Project Access: https://alzheimer-xai-vijay.streamlit.app", font=FONT_CAP, fill=(34, 197, 94))
    d.text((110, card_y + 65), "Team Leader: Vijaytejas A C (1VK23CS074)  •  Parvati Revannavar  •  Moulya S  •  Priyadarshini K", font=FONT_SUB, fill=(255, 255, 255))
    d.text((110, card_y + 100), "Project Guide: Dr. Vidya A, HOD-CSE  •  Vivekananda Institute of Technology", font=FONT_SUB, fill=(180, 190, 205))

    return img

# ──────────────────────────────────────────────
# MAIN RECORDING ORCHESTRATOR
# ──────────────────────────────────────────────
def run_live_screen_recording():
    print("==================================================")
    print("STARTING REAL LIVE STREAMLIT TUTORIAL RECORDING")
    print("Target: https://alzheimer-xai-vijay.streamlit.app")
    print("==================================================")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(OUT_MP4), fourcc, FPS, (VIDEO_W, VIDEO_H))

    header_img = draw_browser_chrome()

    def write_frames(frame_pil: Image.Image, count: int):
        bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        for _ in range(count):
            video_writer.write(bgr)

    def animate_cursor_move(screenshot: Image.Image,
                            start_pos, end_pos,
                            duration_sec: float,
                            step_num="", headline="", subtext="",
                            scroll_y=0,
                            click_at_end=False):
        steps = int(duration_sec * FPS)
        for s in range(steps):
            t = s / max(1, steps - 1)
            # Smooth ease-in-out curve
            ease = 0.5 - 0.5 * math.cos(math.pi * t)
            cx = int(start_pos[0] + (end_pos[0] - start_pos[0]) * ease)
            cy = int(start_pos[1] + (end_pos[1] - start_pos[1]) * ease)
            f = compose_frame(screenshot, header_img, cx, cy, False, 0.0, step_num, headline, subtext, scroll_y)
            write_frames(f, 1)

        if click_at_end:
            # Click ripple animation
            ripple_steps = int(0.35 * FPS)
            for r in range(ripple_steps):
                prog = r / ripple_steps
                f = compose_frame(screenshot, header_img, end_pos[0], end_pos[1], True, prog, step_num, headline, subtext, scroll_y)
                write_frames(f, 1)

    def animate_scroll(screenshot: Image.Image,
                       cursor_pos,
                       start_scroll: int, end_scroll: int,
                       duration_sec: float,
                       step_num="", headline="", subtext=""):
        steps = int(duration_sec * FPS)
        for s in range(steps):
            t = s / max(1, steps - 1)
            ease = 0.5 - 0.5 * math.cos(math.pi * t)
            sy = int(start_scroll + (end_scroll - start_scroll) * ease)
            f = compose_frame(screenshot, header_img, cursor_pos[0], cursor_pos[1], False, 0.0, step_num, headline, subtext, sy)
            write_frames(f, 1)

    with sync_playwright() as p:
        print("[Playwright] Launching browser...")
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--force-device-scale-factor=1",
            ]
        )

        context = browser.new_context(
            viewport={"width": VIEWPORT_W, "height": 2200},  # generous height to capture full scrolling views
            device_scale_factor=1,
        )
        page = context.new_page()

        # =====================================================================
        # SCENE 1 — OPEN THE LINK
        # =====================================================================
        print("\n--- SCENE 1: Open the Project ---")
        # Empty loading canvas initially
        blank_screen = Image.new("RGB", (VIEWPORT_W, 2200), (248, 250, 252))
        loading_draw = ImageDraw.Draw(blank_screen)
        loading_draw.text((VIEWPORT_W // 2 - 220, 300), "Connecting to Streamlit Cloud...", font=FONT_CAP, fill=(100, 116, 139))

        # Cursor moves to address bar and enters link
        animate_cursor_move(blank_screen, (400, 400), (500, -35), 1.2,
                            "Step 1", "Open the Project", "Navigating to https://alzheimer-xai-vijay.streamlit.app")

        print("Loading live application URL...")
        page.goto("https://alzheimer-xai-vijay.streamlit.app/~/+/", timeout=70000)
        page.wait_for_timeout(6000)

        # Grab real rendered home page screenshot
        home_shot_path = TMP_DIR / "real_home.png"
        page.screenshot(path=str(home_shot_path), full_page=True)
        home_img = Image.open(home_shot_path).convert("RGB")
        print("Captured real home page.")

        # Show Home page loaded
        frame = compose_frame(home_img, header_img, 500, 100, False, 0.0,
                              "Step 1", "Open the Project", "Application loaded successfully on Streamlit Community Cloud")
        write_frames(frame, int(2.0 * FPS))

        # =====================================================================
        # SCENE 2 — EXPLORE HOME PAGE
        # =====================================================================
        print("\n--- SCENE 2: Explore Home Page ---")
        # Move cursor over main overview elements
        animate_cursor_move(home_img, (500, 100), (600, 260), 1.5,
                            "Step 2", "Explore the Project", "AI-based Multi-Stage Brain MRI Research Platform")
        animate_cursor_move(home_img, (600, 260), (1050, 420), 1.8,
                            "Step 2", "Explore the Project", "Multi-model agreement engine • Grad-CAM • Calibration")

        # Smooth scroll down home page to show architecture & metrics
        animate_scroll(home_img, (1050, 420), 0, 450, 2.0,
                       "Step 2", "Explore the Project", "Reviewing project pipeline, dataset distribution, and benchmark models")
        # Scroll back up to navigation
        animate_scroll(home_img, (1050, 420), 450, 0, 1.5,
                       "Step 2", "Explore the Project", "Returning to navigation sidebar")

        # =====================================================================
        # SCENE 3 — GO TO MRI ANALYSIS
        # =====================================================================
        print("\n--- SCENE 3: Go to MRI Analysis ---")
        # Cursor moves to sidebar radio button for MRI Analysis
        animate_cursor_move(home_img, (1050, 200), (140, 210), 1.6,
                            "Step 3", "Open MRI Analysis", "Clicking 'MRI Analysis' in the left navigation sidebar",
                            click_at_end=True)

        # Click the actual page
        page.locator("label:has-text('MRI Analysis')").click()
        page.wait_for_timeout(4000)

        mri_empty_shot_path = TMP_DIR / "real_mri_empty.png"
        page.screenshot(path=str(mri_empty_shot_path), full_page=True)
        mri_empty_img = Image.open(mri_empty_shot_path).convert("RGB")
        print("Captured MRI Analysis page (empty state).")

        frame = compose_frame(mri_empty_img, header_img, 140, 210, False, 0.0,
                              "Step 3", "Open MRI Analysis", "MRI Analysis workspace ready for user neuroimaging input")
        write_frames(frame, int(2.0 * FPS))

        # =====================================================================
        # SCENE 4 — SELECT / UPLOAD BRAIN MRI
        # =====================================================================
        print("\n--- SCENE 4: Select / Upload MRI ---")
        # Move cursor to file uploader
        animate_cursor_move(mri_empty_img, (140, 210), (700, 230), 1.5,
                            "Step 4", "Upload a Brain MRI", "Selecting an axial T1-weighted brain MRI scan",
                            click_at_end=True)

        # Perform actual real file upload!
        print(f"Uploading real project MRI: {SAMPLE_MRI}")
        file_input = page.locator("input[type='file']")
        file_input.set_input_files(str(SAMPLE_MRI))
        page.wait_for_timeout(7000)  # wait for inference & Grad-CAM pipeline to complete

        mri_loaded_shot_path = TMP_DIR / "real_mri_loaded.png"
        page.screenshot(path=str(mri_loaded_shot_path), full_page=True)
        mri_loaded_img = Image.open(mri_loaded_shot_path).convert("RGB")
        print("Captured MRI Analysis page with real prediction output.")

        # Show uploaded image preview
        frame = compose_frame(mri_loaded_img, header_img, 700, 230, False, 0.0,
                              "Step 4", "Upload a Brain MRI", f"Scan loaded: {SAMPLE_MRI.name} (Axial Grayscale Neuroimaging)")
        write_frames(frame, int(2.5 * FPS))

        # =====================================================================
        # SCENE 5 — INPUT VALIDATION
        # =====================================================================
        print("\n--- SCENE 5: Input Validation ---")
        animate_scroll(mri_loaded_img, (700, 230), 0, 220, 1.5,
                       "Validation", "Input Quality Validation Passed", "Integrity, aspect ratio, background darkness, and contrast verified")
        animate_cursor_move(mri_loaded_img, (700, 230), (550, 480), 1.5,
                            "Validation", "Input Domain Safeguard Active", "Confirmed valid brain MRI • Out-of-domain images are blocked",
                            scroll_y=220)

        # =====================================================================
        # SCENE 6 & 7 — SELECT MODEL & RUN INFERENCE
        # =====================================================================
        print("\n--- SCENE 6 & 7: Model Selection & Inference ---")
        animate_scroll(mri_loaded_img, (550, 480), 220, 480, 1.5,
                       "Step 5", "Select Model Architecture", "Supported: ResNet18, EfficientNet-B0, MobileNetV2")
        animate_cursor_move(mri_loaded_img, (550, 480), (520, 680), 1.4,
                            "Step 5", "Select Model Architecture", "Primary architecture selected for feature inspection & Grad-CAM",
                            scroll_y=480, click_at_end=True)

        animate_cursor_move(mri_loaded_img, (520, 680), (850, 720), 1.2,
                            "Step 6", "Run Deep Learning Inference", "Automated neural network forward pass & softmax probability computation",
                            scroll_y=480)

        # =====================================================================
        # SCENE 8 — VIEW PREDICTION
        # =====================================================================
        print("\n--- SCENE 8: View Prediction ---")
        animate_scroll(mri_loaded_img, (850, 720), 480, 780, 1.8,
                       "Step 7", "View Model Prediction", "Predicted Dementia Stage & Consensus Agreement")
        animate_cursor_move(mri_loaded_img, (850, 720), (620, 960), 1.5,
                            "Step 7", "View Model Prediction", "Class prediction displayed with calibrated confidence & low uncertainty",
                            scroll_y=780)
        # Highlight prediction card
        frame = compose_frame(mri_loaded_img, header_img, 620, 960, False, 0.0,
                              "Step 7", "View Model Prediction", "High-confidence clinical research stage identification",
                              scroll_y=780)
        write_frames(frame, int(2.5 * FPS))

        # =====================================================================
        # SCENE 9 — VIEW CLASS PROBABILITIES
        # =====================================================================
        print("\n--- SCENE 9: Class Probabilities ---")
        animate_scroll(mri_loaded_img, (620, 960), 780, 1080, 1.8,
                       "Step 7", "Class Probability Distribution", "Probabilities across Non-Demented, Very Mild, Mild, and Moderate stages")
        animate_cursor_move(mri_loaded_img, (620, 960), (750, 1260), 1.6,
                            "Step 7", "Class Probability Distribution", "Full 4-class multi-stage softmax distribution",
                            scroll_y=1080)

        # =====================================================================
        # SCENE 10 — GRAD-CAM EXPLANATION
        # =====================================================================
        print("\n--- SCENE 10: Grad-CAM Explainability ---")
        animate_scroll(mri_loaded_img, (750, 1260), 1080, 1480, 2.0,
                       "Step 8", "Grad-CAM Saliency Explanation", "Original MRI → Grad-CAM Heatmap → Attributed Overlay")
        animate_cursor_move(mri_loaded_img, (750, 1260), (950, 1720), 1.8,
                            "Step 8", "Grad-CAM Saliency Explanation", "Disclaimer: Model attribution visualization, NOT a clinical diagnosis",
                            scroll_y=1480)

        # Hold on Grad-CAM section to let user observe
        frame = compose_frame(mri_loaded_img, header_img, 950, 1720, False, 0.0,
                              "Step 8", "Grad-CAM Saliency Explanation", "Warm colors (Red/Yellow) highlight regions driving neural network prediction",
                              scroll_y=1480)
        write_frames(frame, int(3.5 * FPS))

        # =====================================================================
        # SCENE 11 — MODEL COMPARISON
        # =====================================================================
        print("\n--- SCENE 11: Model Comparison ---")
        animate_scroll(mri_loaded_img, (950, 1720), 1480, 0, 1.5,
                       "Step 9", "Compare Models", "Navigating to Model Comparison")
        animate_cursor_move(mri_loaded_img, (950, 200), (140, 245), 1.5,
                            "Step 9", "Compare Models", "Opening 'Model Comparison' page in sidebar",
                            click_at_end=True)

        page.locator("label:has-text('Model Comparison')").click()
        page.wait_for_timeout(4000)

        comp_shot_path = TMP_DIR / "real_comp.png"
        page.screenshot(path=str(comp_shot_path), full_page=True)
        comp_img = Image.open(comp_shot_path).convert("RGB")
        print("Captured Model Comparison page.")

        frame = compose_frame(comp_img, header_img, 140, 245, False, 0.0,
                              "Step 9", "Compare Models", "Held-Out Test Benchmarks: MobileNetV2 vs EfficientNet-B0 vs ResNet18")
        write_frames(frame, int(2.0 * FPS))

        animate_scroll(comp_img, (700, 350), 0, 350, 2.0,
                       "Step 9", "Compare Models", "Accuracy, Macro F1, Balanced Accuracy, MCC, ROC-AUC, and Latency")
        write_frames(compose_frame(comp_img, header_img, 700, 350, False, 0.0,
                                   "Step 9", "Compare Models", "ResNet18: 98.44% Acc • EfficientNet-B0: 98.23% Acc • MobileNetV2: 93.13% Acc",
                                   scroll_y=350), int(2.5 * FPS))

        # =====================================================================
        # SCENE 12 — EVALUATION RESULTS
        # =====================================================================
        print("\n--- SCENE 12: Evaluation Results ---")
        animate_scroll(comp_img, (700, 350), 350, 0, 1.2,
                       "Step 10", "Explore Evaluation Results", "Opening Evaluation page")
        animate_cursor_move(comp_img, (700, 150), (140, 280), 1.4,
                            "Step 10", "Explore Evaluation Results", "Clicking 'Evaluation' in sidebar",
                            click_at_end=True)

        page.locator("label:has-text('Evaluation')").click()
        page.wait_for_timeout(4000)

        eval_shot_path = TMP_DIR / "real_eval.png"
        page.screenshot(path=str(eval_shot_path), full_page=True)
        eval_img = Image.open(eval_shot_path).convert("RGB")
        print("Captured Evaluation Results page.")

        frame = compose_frame(eval_img, header_img, 140, 280, False, 0.0,
                              "Step 10", "Explore Evaluation Results", "Confusion Matrices, Per-Class ROC Curves, and Calibration")
        write_frames(frame, int(2.0 * FPS))

        animate_scroll(eval_img, (700, 350), 0, 420, 2.0,
                       "Step 10", "Explore Evaluation Results", "Reviewing confusion matrices and per-stage diagnostic sensitivity",
                       )
        write_frames(compose_frame(eval_img, header_img, 700, 350, False, 0.0,
                                   "Step 10", "Explore Evaluation Results", "Verified test metrics on 960 held-out clinical images",
                                   scroll_y=420), int(2.5 * FPS))

        # =====================================================================
        # SCENE 13 — EXPLORE OTHER RESEARCH PAGES
        # =====================================================================
        print("\n--- SCENE 13: Other Research Pages ---")
        animate_scroll(eval_img, (700, 350), 420, 0, 1.2,
                       "Research", "Explore Additional Research Pages", "Error & Robustness, Methodology, About")

        # Click Error & Robustness
        animate_cursor_move(eval_img, (700, 150), (140, 350), 1.3,
                            "Research", "Error & Robustness", "Testing model stability under blur, noise, and lighting shifts",
                            click_at_end=True)
        page.locator("label:has-text('Error & Robustness')").click()
        page.wait_for_timeout(3500)
        robust_shot = Image.open(io.BytesIO(page.screenshot(full_page=True))).convert("RGB")
        write_frames(compose_frame(robust_shot, header_img, 140, 350, False, 0.0,
                                   "Research", "Error & Robustness", "5 perturbation stress-testing categories evaluated empirically"), int(2.0 * FPS))

        # Click Methodology
        page.locator("label:has-text('Methodology')").click()
        page.wait_for_timeout(3000)
        method_shot = Image.open(io.BytesIO(page.screenshot(full_page=True))).convert("RGB")
        write_frames(compose_frame(method_shot, header_img, 140, 385, False, 0.0,
                                   "Research", "Research Methodology", "Pre-processing specifications, loss functions, and training configuration"), int(2.0 * FPS))

        # Click About
        page.locator("label:has-text('About')").click()
        page.wait_for_timeout(3000)
        about_shot = Image.open(io.BytesIO(page.screenshot(full_page=True))).convert("RGB")
        write_frames(compose_frame(about_shot, header_img, 140, 420, False, 0.0,
                                   "Research", "About & Scientific Disclaimer", "Academic research prototype • Not an FDA-cleared diagnostic tool"), int(2.0 * FPS))

        browser.close()

    # =====================================================================
    # SCENE 14 — COMPLETE USER JOURNEY RECAP SLIDE
    # =====================================================================
    print("\n--- SCENE 14: Recap & Live Project Access ---")
    recap_frame = create_recap_frame()
    write_frames(recap_frame, int(5.5 * FPS))

    # Fade to black
    black_img = Image.new("RGB", (VIDEO_W, VIDEO_H), (0, 0, 0))
    fade_steps = int(1.0 * FPS)
    recap_arr = np.array(recap_frame, dtype=np.float32)
    for i in range(fade_steps):
        t = i / fade_steps
        f = (recap_arr * (1.0 - t)).astype(np.uint8)
        write_frames(Image.fromarray(f), 1)

    video_writer.release()
    print("==================================================")
    print("RECORDING COMPLETE!")
    print(f"Saved: {OUT_MP4}")
    duration = OUT_MP4.stat().st_size
    print(f"File size: {duration / (1024*1024):.2f} MB")
    print("==================================================")

if __name__ == "__main__":
    run_live_screen_recording()
