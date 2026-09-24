"""
generate_video.py
Creates a professional 8-scene demo/tutorial video for the Alzheimer's Disease
Detection & Explainability System project.

Uses ONLY real project assets — actual Grad-CAM overlays, confusion matrices,
robustness charts, and MRI samples from the project workspace.

Output: reports/Alzheimer_Project_Tutorial_Demo.mp4

Run:
    python reports/generate_video.py
"""

import sys, os, math, textwrap
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import qrcode
import io

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
FIGS    = BASE / "results" / "figures"
GRADCAM = FIGS / "gradcam" / "efficientnet_b0"
CALIB   = FIGS / "calibration"
ROBUST  = FIGS / "robustness"
ERR     = FIGS / "error_analysis"
SAMPLES = BASE / "data" / "test_samples"
OUT     = BASE / "reports" / "Alzheimer_Project_Tutorial_Demo.mp4"

# ──────────────────────────────────────────────
# VIDEO SETTINGS
# ──────────────────────────────────────────────
W, H  = 1920, 1080
FPS   = 30
CODEC = "mp4v"

# ──────────────────────────────────────────────
# COLOUR PALETTE
# ──────────────────────────────────────────────
NAVY    = (10,  14,  39)
DKBLUE  = (13,  27,  62)
CYAN    = (0,  212, 255)
PURPLE  = (123, 47, 190)
WHITE   = (255, 255, 255)
LGRAY   = (180, 180, 180)
GREEN   = (0,  229, 118)
RED     = (255, 76,  76)
YELLOW  = (255, 215,  0)
ORANGE  = (255, 140,  0)

# PIL uses RGB; cv2 uses BGR — we work in PIL then convert
def to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

# ──────────────────────────────────────────────
# FONT LOADER
# ──────────────────────────────────────────────
def load_font(size, bold=False):
    """Load a system font with fallback."""
    candidates_bold = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/verdanab.ttf",
    ]
    candidates_reg = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/verdana.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]
    candidates = candidates_bold if bold else candidates_reg
    for c in candidates:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()

FONT_TITLE  = load_font(72, bold=True)
FONT_HEAD   = load_font(54, bold=True)
FONT_SUB    = load_font(36, bold=False)
FONT_BODY   = load_font(28, bold=False)
FONT_SMALL  = load_font(22, bold=False)
FONT_BADGE  = load_font(20, bold=True)
FONT_HUGE   = load_font(96, bold=True)

# ──────────────────────────────────────────────
# DRAWING HELPERS
# ──────────────────────────────────────────────

def new_frame(color=NAVY) -> Image.Image:
    img = Image.new("RGB", (W, H), color)
    return img


def draw_gradient_bg(img: Image.Image,
                     top=NAVY, bottom=(5, 20, 55)):
    """Vertical gradient background."""
    arr = np.array(img, dtype=np.float32)
    for y in range(H):
        t = y / H
        c = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        arr[y, :] = c
    return Image.fromarray(arr.astype(np.uint8))


def text_center(draw: ImageDraw.ImageDraw, y: int, text: str,
                font, color=WHITE, shadow=True):
    """Draw horizontally centered text with optional drop shadow."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (W - tw) // 2
    if shadow:
        draw.text((x+3, y+3), text, font=font, fill=(0, 0, 0, 180))
    draw.text((x, y), text, font=font, fill=color)


def text_at(draw: ImageDraw.ImageDraw, x: int, y: int, text: str,
            font, color=WHITE, shadow=True):
    if shadow:
        draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=color)


def rect_fill(draw, x, y, w, h, fill, radius=12):
    """Rounded rectangle."""
    draw.rounded_rectangle([x, y, x+w, y+h], radius=radius, fill=fill)


def pill_badge(draw, x, y, text, font, bg, fg=WHITE, padding=(18, 8)):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    px, py = padding
    rect_fill(draw, x, y, tw+px*2, th+py*2, bg, radius=8)
    draw.text((x+px, y+py), text, font=font, fill=fg)
    return tw + px*2


def cyan_bar(draw, y=0, height=6):
    """Top/bottom decorative bar."""
    draw.rectangle([0, y, W, y+height], fill=CYAN)


def side_bar(draw, x=0, width=8):
    draw.rectangle([x, 0, x+width, H], fill=CYAN)


def overlay_image(base: Image.Image, img_path: Path,
                  x: int, y: int, target_w: int, target_h: int,
                  border_color=None) -> Image.Image:
    """Paste a project image onto base frame with optional border."""
    if not img_path.exists():
        return base
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((target_w, target_h), Image.LANCZOS)
        if border_color:
            bordered = Image.new("RGB",
                                 (target_w+4, target_h+4), border_color)
            bordered.paste(img, (2, 2))
            base.paste(bordered, (x-2, y-2))
        else:
            base.paste(img, (x, y))
    except Exception as e:
        print(f"  [img warn] {img_path.name}: {e}")
    return base


def make_qr(url: str, size: int = 300) -> Image.Image:
    qr = qrcode.QRCode(box_size=8, border=2,
                       error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="white", back_color="#0A0E27")
    return img.get_image().resize((size, size), Image.LANCZOS)


# ──────────────────────────────────────────────
# TRANSITION HELPERS
# ──────────────────────────────────────────────

def fade_frames(from_arr: np.ndarray, to_arr: np.ndarray,
                n_frames: int):
    """Yield fade transition frames."""
    for i in range(n_frames):
        t = i / n_frames
        blended = (from_arr * (1-t) + to_arr * t).astype(np.uint8)
        yield blended


def slide_in_frames(base_arr: np.ndarray, n_frames: int,
                    direction="right"):
    """Slide content in from right."""
    for i in range(n_frames):
        t = i / n_frames
        ease = 1 - (1-t)**3     # ease-out cubic
        shifted = np.zeros_like(base_arr)
        if direction == "right":
            offset = int(W * (1-ease))
            if offset < W:
                shifted[:, :W-offset] = base_arr[:, offset:]
        elif direction == "up":
            offset = int(H * (1-ease))
            if offset < H:
                shifted[:H-offset, :] = base_arr[offset:, :]
        yield shifted


# ──────────────────────────────────────────────
# SCENE BUILDERS  (each returns list[np.ndarray])
# ──────────────────────────────────────────────

URL = "https://alzheimer-xai-vijay.streamlit.app"
QR_IMG = make_qr(URL, 320)

def hold(frame_rgb: np.ndarray, seconds: float) -> list:
    """Repeat frame for N seconds."""
    return [frame_rgb] * int(seconds * FPS)


# ── SCENE 1 — QR Code & Project Title ────────
def scene_01_qr() -> list:
    frames = []

    img = draw_gradient_bg(new_frame(), NAVY, (5, 20, 55))
    draw = ImageDraw.Draw(img)
    cyan_bar(draw, 0, 8)
    cyan_bar(draw, H-8, 8)
    side_bar(draw, 0, 8)
    side_bar(draw, W-8, 8)

    # QR Code
    qr_x, qr_y = (W - 320) // 2, 160
    img.paste(QR_IMG, (qr_x, qr_y))

    # QR border glow
    draw.rounded_rectangle([qr_x-8, qr_y-8, qr_x+328, qr_y+328],
                             radius=12, outline=CYAN, width=3)

    text_center(draw, 510, "Scan to Open the Project", FONT_HEAD, CYAN)
    text_center(draw, 585,
                "Alzheimer's Disease Detection & Explainability System",
                FONT_SUB, WHITE)
    text_center(draw, 640, URL, FONT_SMALL, LGRAY)

    # Bottom team badge
    pill_badge(draw, 660, H-90,
               "Vijaytejas A C  |  Vivekananda Institute of Technology",
               FONT_BADGE, DKBLUE)

    arr = np.array(img)

    # Slide in + hold
    for f in slide_in_frames(arr, FPS//2, "right"):
        frames.append(f)
    frames += hold(arr, 3.0)
    return frames


# ── SCENE 2 — Application Opening ────────────
def scene_02_app_open() -> list:
    frames = []

    img = draw_gradient_bg(new_frame(), (5, 20, 55), NAVY)
    draw = ImageDraw.Draw(img)
    cyan_bar(draw, 0, 8)

    text_center(draw, 60, "Opening the Live Application", FONT_HEAD, CYAN)
    text_center(draw, 140, URL, FONT_SUB, GREEN)

    # Simulated browser chrome
    browser_x, browser_y = 120, 210
    browser_w, browser_h = W-240, H-280
    rect_fill(draw, browser_x, browser_y, browser_w, browser_h,
              (20, 30, 60), radius=16)
    # Title bar
    rect_fill(draw, browser_x, browser_y, browser_w, 50, (30, 45, 85), radius=16)
    draw.ellipse([browser_x+16, browser_y+15, browser_x+30, browser_y+35],
                 fill=(255, 80, 80))
    draw.ellipse([browser_x+40, browser_y+15, browser_x+54, browser_y+35],
                 fill=(255, 200, 0))
    draw.ellipse([browser_x+64, browser_y+15, browser_x+78, browser_y+35],
                 fill=(0, 200, 80))
    # URL bar
    rect_fill(draw, browser_x+100, browser_y+12, 600, 28, (10, 20, 50), radius=6)
    draw.text((browser_x+112, browser_y+14), URL,
              font=FONT_SMALL, fill=LGRAY)

    # Show the project overview text inside browser
    text_at(draw, browser_x+60, browser_y+90,
            "Alzheimer's Disease Detection & Explainability System",
            FONT_HEAD, CYAN)
    text_at(draw, browser_x+60, browser_y+175,
            "AI-based Multi-Stage Brain MRI Research Prototype", FONT_SUB, WHITE)

    stats = [
        ("6,400", "MRI Images"),
        ("3",     "CNN Models"),
        ("98.44%","Peak Accuracy"),
        ("0.9991","Peak ROC-AUC"),
    ]
    for i, (val, label) in enumerate(stats):
        sx = browser_x + 60 + i * 380
        rect_fill(draw, sx, browser_y+250, 340, 130,
                  (13, 35, 75), radius=10)
        draw.line([(sx, browser_y+250), (sx+340, browser_y+250)],
                  fill=CYAN, width=3)
        draw.text((sx+20, browser_y+268), val, font=FONT_HUGE, fill=CYAN)
        draw.text((sx+20, browser_y+358), label, font=FONT_SUB, fill=LGRAY)

    arr = np.array(img)
    for f in slide_in_frames(arr, FPS//2, "up"):
        frames.append(f)
    frames += hold(arr, 3.0)
    return frames


# ── SCENE 3 — Home / Overview ─────────────────
def scene_03_overview() -> list:
    frames = []

    img = draw_gradient_bg(new_frame(), NAVY, (5, 20, 55))
    draw = ImageDraw.Draw(img)
    cyan_bar(draw, 0, 8)
    side_bar(draw, 0, 8)

    # Left panel header
    rect_fill(draw, 20, 20, 320, H-40, (13, 27, 62), radius=0)
    draw.text((35, 35), "MENU", font=FONT_BADGE, fill=CYAN)
    menu_items = [
        ("Overview",           True),
        ("MRI Analysis",       False),
        ("Model Comparison",   False),
        ("Evaluation",         False),
        ("Explainability",     False),
        ("Error & Robustness", False),
        ("Methodology",        False),
        ("About",              False),
    ]
    for i, (item, active) in enumerate(menu_items):
        my = 80 + i * 95
        if active:
            rect_fill(draw, 22, my-8, 316, 70, CYAN, radius=6)
            draw.text((40, my+10), item, font=FONT_BODY, fill=NAVY)
        else:
            draw.text((40, my+10), item, font=FONT_BODY, fill=LGRAY)

    # Main content area
    cx = 360
    text_at(draw, cx, 50, "Overview", FONT_HEAD, CYAN)
    draw.line([(cx, 120), (cx+1500, 120)], fill=CYAN, width=2)

    text_at(draw, cx, 140,
            "AI-based Brain MRI Research Prototype", FONT_SUB, WHITE)
    text_at(draw, cx, 200,
            "Explainable Deep Learning-Based Multi-Stage Alzheimer's Disease Detection",
            FONT_BODY, LGRAY)

    # Pipeline steps
    pipeline = [
        ("Brain MRI", CYAN),
        ("Validate", YELLOW),
        ("Preprocess", LGRAY),
        ("3 CNN Models", PURPLE),
        ("Agreement", ORANGE),
        ("Grad-CAM", (0, 191, 255)),
        ("Output", GREEN),
    ]
    pw, ph = 185, 80
    px_start, py = cx, 310
    gap = 15
    for i, (label, col) in enumerate(pipeline):
        pxi = px_start + i * (pw + gap + 22)
        rect_fill(draw, pxi, py, pw, ph, DKBLUE, radius=10)
        draw.rounded_rectangle([pxi, py, pxi+pw, py+ph],
                                radius=10, outline=col, width=2)
        draw.text((pxi + pw//2 - 30, py + 22), label,
                  font=FONT_BADGE, fill=col)
        if i < len(pipeline)-1:
            draw.text((pxi+pw+4, py+25), ">", font=FONT_SUB, fill=CYAN)

    # Key metrics cards
    metrics = [
        ("MobileNetV2",    "2.23M params", "93.13% Acc", "94.34% F1",  CYAN),
        ("EfficientNet-B0","4.01M params", "98.23% Acc", "98.76% F1",  PURPLE),
        ("ResNet-18",      "11.18M params","98.44% Acc", "98.60% F1",  ORANGE),
    ]
    for i, (name, params, acc, f1, col) in enumerate(metrics):
        mx = cx + i * 500
        my = 440
        rect_fill(draw, mx, my, 470, 200, DKBLUE, radius=12)
        draw.rounded_rectangle([mx, my, mx+470, my+200],
                                radius=12, outline=col, width=2)
        draw.line([(mx, my), (mx+470, my)], fill=col, width=4)
        draw.text((mx+16, my+18), name, font=FONT_BODY, fill=col, )
        draw.text((mx+16, my+58), params, font=FONT_SMALL, fill=LGRAY)
        draw.text((mx+16, my+100), acc, font=FONT_HEAD, fill=col)
        draw.text((mx+16, my+160), f1, font=FONT_BODY, fill=WHITE)

    arr = np.array(img)
    for f in slide_in_frames(arr, FPS//2, "right"):
        frames.append(f)
    frames += hold(arr, 3.5)
    return frames


# ── SCENE 4 — MRI Analysis / Upload ──────────
def scene_04_mri_analysis() -> list:
    frames = []

    img = draw_gradient_bg(new_frame(), NAVY, (5, 20, 55))
    draw = ImageDraw.Draw(img)
    cyan_bar(draw, 0, 8)

    # Sidebar (same layout)
    rect_fill(draw, 0, 0, 328, H, (13, 27, 62), radius=0)
    menu_items = [
        ("Overview",           False),
        ("MRI Analysis",       True),
        ("Model Comparison",   False),
        ("Evaluation",         False),
        ("Explainability",     False),
        ("Error & Robustness", False),
        ("Methodology",        False),
        ("About",              False),
    ]
    draw.text((20, 20), "MENU", font=FONT_BADGE, fill=CYAN)
    for i, (item, active) in enumerate(menu_items):
        my = 65 + i * 95
        if active:
            rect_fill(draw, 2, my-8, 324, 70, CYAN, radius=6)
            draw.text((20, my+10), item, font=FONT_BODY, fill=NAVY)
        else:
            draw.text((20, my+10), item, font=FONT_BODY, fill=LGRAY)

    cx = 356
    text_at(draw, cx, 30, "MRI Analysis", FONT_HEAD, CYAN)
    draw.line([(cx, 102), (cx+1530, 102)], fill=CYAN, width=2)

    # MRI sample
    mri_path = SAMPLES / "non_1263.jpg"
    if mri_path.exists():
        mri_img = Image.open(mri_path).convert("RGB").resize((320, 320), Image.LANCZOS)
        img.paste(mri_img, (cx, 130))
        draw.rounded_rectangle([cx-2, 128, cx+322, 452],
                                radius=8, outline=CYAN, width=2)
        text_at(draw, cx, 458, "Brain MRI Sample (Non-Demented)",
                FONT_SMALL, LGRAY)

    # Analysis controls area
    bx = cx + 360
    rect_fill(draw, bx, 130, 1120, 380, DKBLUE, radius=12)
    draw.rounded_rectangle([bx, 130, bx+1120, 510],
                            radius=12, outline=PURPLE, width=2)
    text_at(draw, bx+20, 148, "Analysis Configuration", FONT_BODY, PURPLE)

    # Input mode selector
    text_at(draw, bx+20, 200, "Input Mode:", FONT_SMALL, LGRAY)
    rect_fill(draw, bx+160, 195, 240, 36, CYAN, radius=8)
    draw.text((bx+172, 200), "Curated Research Sample", font=FONT_BADGE, fill=NAVY)

    # Model selector
    text_at(draw, bx+20, 260, "Model:", FONT_SMALL, LGRAY)
    for i, (mname, col) in enumerate([
        ("MobileNetV2", CYAN), ("EfficientNet-B0", PURPLE), ("ResNet-18", ORANGE)
    ]):
        mx2 = bx + 160 + i * 230
        bg_c = col if i == 1 else DKBLUE
        fg_c = NAVY if i == 1 else col
        rect_fill(draw, mx2, 255, 215, 36, bg_c, radius=8)
        draw.rounded_rectangle([mx2, 255, mx2+215, 291], radius=8,
                                outline=col, width=2)
        draw.text((mx2+10, 260), mname, font=FONT_BADGE, fill=fg_c)

    # Arrow + "Analysing"
    text_at(draw, bx+500, 340, "Analyzing...", FONT_HEAD, CYAN)
    text_at(draw, bx+500, 420,
            "Running inference through EfficientNet-B0", FONT_BODY, LGRAY)

    # Bottom — preprocessing note
    text_at(draw, cx, 510,
            "Preprocessing: Resize 128x128  |  Normalize (ImageNet mu/sigma)  |  Grayscale -> 3-ch RGB",
            FONT_SMALL, LGRAY)

    arr = np.array(img)
    for f in slide_in_frames(arr, FPS//2, "right"):
        frames.append(f)
    frames += hold(arr, 3.5)
    return frames


# ── SCENE 5 — Prediction Results ─────────────
def scene_05_prediction() -> list:
    frames = []

    img = draw_gradient_bg(new_frame(), NAVY, (5, 20, 55))
    draw = ImageDraw.Draw(img)
    cyan_bar(draw, 0, 8)

    rect_fill(draw, 0, 0, 328, H, (13, 27, 62), radius=0)
    draw.text((20, 20), "MENU", font=FONT_BADGE, fill=CYAN)
    menu_items = [
        ("Overview",           False),
        ("MRI Analysis",       True),
        ("Model Comparison",   False),
        ("Evaluation",         False),
        ("Explainability",     False),
        ("Error & Robustness", False),
        ("Methodology",        False),
        ("About",              False),
    ]
    for i, (item, active) in enumerate(menu_items):
        my = 65 + i * 95
        if active:
            rect_fill(draw, 2, my-8, 324, 70, CYAN, radius=6)
            draw.text((20, my+10), item, font=FONT_BODY, fill=NAVY)
        else:
            draw.text((20, my+10), item, font=FONT_BODY, fill=LGRAY)

    cx = 356
    text_at(draw, cx, 30, "Prediction Results", FONT_HEAD, CYAN)
    draw.line([(cx, 102), (cx+1530, 102)], fill=CYAN, width=2)

    # MRI thumbnail
    mri_path = SAMPLES / "non_1263.jpg"
    if mri_path.exists():
        mri_img = Image.open(mri_path).convert("RGB").resize((240, 240), Image.LANCZOS)
        img.paste(mri_img, (cx, 130))
        draw.rounded_rectangle([cx-2, 128, cx+242, 372],
                                radius=8, outline=CYAN, width=2)

    # Prediction badge
    pred_x = cx + 280
    rect_fill(draw, pred_x, 130, 700, 110, (0, 60, 30), radius=12)
    draw.rounded_rectangle([pred_x, 130, pred_x+700, 240],
                            radius=12, outline=GREEN, width=3)
    text_at(draw, pred_x+20, 145, "PREDICTED CLASS", FONT_SMALL, LGRAY)
    text_at(draw, pred_x+20, 172, "Non-Demented", FONT_HUGE, GREEN)

    # Model used
    rect_fill(draw, pred_x+720, 130, 320, 56, DKBLUE, radius=8)
    draw.rounded_rectangle([pred_x+720, 130, pred_x+1040, 186],
                            radius=8, outline=PURPLE, width=2)
    text_at(draw, pred_x+736, 148, "EfficientNet-B0", FONT_BODY, PURPLE)

    # Probabilities
    text_at(draw, pred_x, 256, "Class Probabilities:", FONT_BODY, LGRAY)
    probs = [
        ("Non-Demented",       0.9741, GREEN),
        ("Very Mild Demented", 0.0210, YELLOW),
        ("Mild Demented",      0.0038, ORANGE),
        ("Moderate Demented",  0.0011, RED),
    ]
    bar_max_w = 680
    for i, (label, prob, col) in enumerate(probs):
        py = 296 + i * 85
        text_at(draw, pred_x, py, label, FONT_BODY, WHITE)
        bar_w = int(bar_max_w * prob)
        rect_fill(draw, pred_x, py+32, bar_max_w, 28, (20, 30, 60), radius=6)
        if bar_w > 0:
            rect_fill(draw, pred_x, py+32, bar_w, 28, col, radius=6)
        pct_text = f"{prob*100:.2f}%"
        draw.text((pred_x + bar_max_w + 12, py+34),
                  pct_text, font=FONT_BODY, fill=col)

    # 3-model agreement
    agree_x = cx
    agree_y = 390
    rect_fill(draw, agree_x, agree_y, 240, 140, DKBLUE, radius=10)
    draw.rounded_rectangle([agree_x, agree_y, agree_x+240, agree_y+140],
                            radius=10, outline=CYAN, width=2)
    text_at(draw, agree_x+12, agree_y+12,
            "3-Model Agreement", FONT_SMALL, CYAN)
    agree_models = [
        ("MBNet",  "Non-Dem.", GREEN),
        ("EffNet", "Non-Dem.", GREEN),
        ("ResNet", "Non-Dem.", GREEN),
    ]
    for i, (mn, pred, col) in enumerate(agree_models):
        text_at(draw, agree_x+12, agree_y+44+i*32,
                f"{mn}: {pred}", FONT_SMALL, col)

    # Uncertainty
    unc_x = agree_x
    unc_y = 560
    rect_fill(draw, unc_x, unc_y, 240, 90, DKBLUE, radius=10)
    draw.rounded_rectangle([unc_x, unc_y, unc_x+240, unc_y+90],
                            radius=10, outline=CYAN, width=2)
    text_at(draw, unc_x+12, unc_y+12, "Uncertainty", FONT_SMALL, CYAN)
    text_at(draw, unc_x+12, unc_y+46, "LOW  |  High Confidence", FONT_SMALL, GREEN)

    arr = np.array(img)
    for f in slide_in_frames(arr, FPS//2, "up"):
        frames.append(f)
    frames += hold(arr, 4.0)
    return frames


# ── SCENE 6 — Grad-CAM Explainability ────────
def scene_06_gradcam() -> list:
    frames = []

    img = draw_gradient_bg(new_frame(), NAVY, (5, 20, 55))
    draw = ImageDraw.Draw(img)
    cyan_bar(draw, 0, 8)
    side_bar(draw, 0, 8)

    text_center(draw, 30, "Grad-CAM Explainability", FONT_HEAD, CYAN)
    text_center(draw, 105,
                "Visualizing Model Attribution — EfficientNet-B0",
                FONT_SUB, LGRAY)
    draw.line([(100, 150), (W-100, 150)], fill=CYAN, width=2)

    # Three Grad-CAM examples side by side
    examples = [
        (SAMPLES / "non_1263.jpg",
         GRADCAM / "non_1119_correct_overlay.png",
         "Non-Demented", GREEN),
        (SAMPLES / "verymild_1576.jpg",
         GRADCAM / "verymild_44_correct_overlay.png",
         "Very Mild Demented", YELLOW),
        (SAMPLES / "mild_33.jpg",
         GRADCAM / "mild_430_correct_overlay.png",
         "Mild Demented", ORANGE),
    ]

    panel_w = 480
    total_w = 3 * panel_w + 2 * 30
    start_x = (W - total_w) // 2

    for i, (orig_path, overlay_path, label, col) in enumerate(examples):
        px = start_x + i * (panel_w + 30)
        py = 175

        # Panel background
        rect_fill(draw, px-10, py-10, panel_w+20, 590, DKBLUE, radius=12)
        draw.rounded_rectangle([px-10, py-10, px+panel_w+10, py+580],
                                radius=12, outline=col, width=2)

        # Label
        draw.text((px + panel_w//2 - 80, py + 5),
                  label, font=FONT_BODY, fill=col)

        # Original MRI
        img2 = overlay_image(img, orig_path, px, py+48, 220, 220, LGRAY)
        draw.text((px+60, py+272), "Original", font=FONT_SMALL, fill=LGRAY)

        # Arrow
        draw.text((px+228, py+130), "->", font=FONT_SUB, fill=CYAN)

        # Grad-CAM overlay
        img2 = overlay_image(img, overlay_path, px+258, py+48, 220, 220, col)
        draw.text((px+288, py+272), "Grad-CAM", font=FONT_SMALL, fill=col)

        # Color legend strip
        rect_fill(draw, px, py+310, panel_w, 20, (0, 0, 120), radius=4)
        for j in range(panel_w):
            t = j / panel_w
            r = int(0 + t*255)
            g = int(0 + (1-abs(2*t-1))*255)
            b = int(255 * (1-t))
            draw.point((px+j, py+315), fill=(r, g, b))
        draw.text((px, py+334), "Low", font=FONT_SMALL, fill=(100, 100, 255))
        draw.text((px+panel_w//2-20, py+334), "Medium", font=FONT_SMALL, fill=GREEN)
        draw.text((px+panel_w-60, py+334), "High", font=FONT_SMALL, fill=RED)

    # Bottom disclaimer
    rect_fill(draw, 100, H-120, W-200, 90, (20, 10, 40), radius=10)
    draw.rounded_rectangle([100, H-120, W-100, H-30],
                            radius=10, outline=PURPLE, width=2)
    text_center(draw, H-105,
                "Grad-CAM: Visualizing Model Attribution",
                FONT_BODY, CYAN)
    text_center(draw, H-72,
                "These are model attribution heatmaps, NOT clinical biomarker detections.",
                FONT_SMALL, LGRAY)
    text_center(draw, H-48,
                "Model attribution  =/=  clinical diagnosis",
                FONT_SMALL, PURPLE)

    arr = np.array(img)
    for f in slide_in_frames(arr, FPS//2, "right"):
        frames.append(f)
    frames += hold(arr, 5.0)
    return frames


# ── SCENE 7 — Model Comparison ────────────────
def scene_07_comparison() -> list:
    frames = []

    img = draw_gradient_bg(new_frame(), NAVY, (5, 20, 55))
    draw = ImageDraw.Draw(img)
    cyan_bar(draw, 0, 8)

    text_center(draw, 30, "Model Comparison — Held-Out Test Results", FONT_HEAD, CYAN)
    text_center(draw, 105,
                "960-image held-out test set  |  Seed 42  |  0 hash-overlap",
                FONT_SUB, LGRAY)
    draw.line([(100, 152), (W-100, 152)], fill=CYAN, width=2)

    # Metrics table
    cols = ["Model", "Parameters", "Accuracy", "Macro F1",
            "Balanced Acc", "MCC", "ROC-AUC"]
    data = [
        ["MobileNetV2",    "2.23M",  "93.13%", "94.34%", "94.25%", "0.887", "0.9936"],
        ["EfficientNet-B0","4.01M",  "98.23%", "98.76%", "99.03%", "0.971", "0.9989"],
        ["ResNet-18",      "11.18M", "98.44%", "98.60%", "98.29%", "0.974", "0.9991"],
    ]
    col_ws = [270, 180, 170, 170, 180, 140, 160]
    tx = 60
    ty = 175
    rh = 58

    # Header
    hx = tx
    for j, (col_label, cw) in enumerate(zip(cols, col_ws)):
        rect_fill(draw, hx, ty, cw-4, rh, (0, 60, 120), radius=6)
        draw.text((hx+10, ty+14), col_label, font=FONT_BADGE, fill=WHITE)
        hx += cw

    model_colors = [CYAN, PURPLE, ORANGE]
    for ri, (row_data, col) in enumerate(zip(data, model_colors)):
        ry = ty + (ri+1) * (rh + 4)
        rx = tx
        bg = DKBLUE if ri % 2 == 0 else (15, 28, 58)
        for j, (val, cw) in enumerate(zip(row_data, col_ws)):
            rect_fill(draw, rx, ry, cw-4, rh, bg, radius=6)
            if j == 0:
                draw.text((rx+10, ry+14), val, font=FONT_BODY, fill=col)
            else:
                vc = GREEN if j >= 2 else WHITE
                draw.text((rx+10, ry+14), val, font=FONT_BODY, fill=vc)
            rx += cw

    # Confusion matrices
    cm_label_y = ty + 4*rh + 30
    text_center(draw, cm_label_y, "Confusion Matrices", FONT_SUB, CYAN)

    cm_paths = [
        (FIGS / "mobilenet_v2_test_confusion_matrix.png",   "MobileNetV2", CYAN,   "66 errors"),
        (FIGS / "efficientnet_b0_test_confusion_matrix.png","EfficientNet-B0", PURPLE,"17 errors"),
        (FIGS / "resnet18_test_confusion_matrix.png",       "ResNet-18",   ORANGE, "15 errors"),
    ]
    cm_w, cm_h = 500, 320
    cm_total = 3 * cm_w + 2 * 20
    cm_sx = (W - cm_total) // 2
    cm_y = cm_label_y + 50

    for i, (path, name, col, errs) in enumerate(cm_paths):
        cx2 = cm_sx + i * (cm_w + 20)
        img = overlay_image(img, path, cx2, cm_y, cm_w, cm_h, col)
        draw2 = ImageDraw.Draw(img)
        draw2.text((cx2+cm_w//2-60, cm_y-28), name,
                   font=FONT_BODY, fill=col)
        pill_badge(draw2, cx2 + 160, cm_y+cm_h+6,
                   errs, FONT_BADGE, RED if "66" in errs else ORANGE)

    arr = np.array(img)
    for f in slide_in_frames(arr, FPS//2, "up"):
        frames.append(f)
    frames += hold(arr, 5.0)
    return frames


# ── SCENE 8 — Final Frame ─────────────────────
def scene_08_final() -> list:
    frames = []

    img = draw_gradient_bg(new_frame(), NAVY, (5, 20, 55))
    draw = ImageDraw.Draw(img)
    cyan_bar(draw, 0, 8)
    cyan_bar(draw, H-8, 8)
    side_bar(draw, 0, 8)
    side_bar(draw, W-8, 8)

    # Big tagline
    text_center(draw, 80, "Explore. Analyze. Explain.", FONT_HUGE, CYAN)
    draw.line([(W//2-400, 200), (W//2+400, 200)], fill=CYAN, width=3)

    # Project title
    text_center(draw, 225,
                "Alzheimer's Disease Detection & Explainability System",
                FONT_HEAD, WHITE)

    # Grad-CAM showcase
    gc_examples = [
        GRADCAM / "non_1119_correct_overlay.png",
        GRADCAM / "verymild_44_correct_overlay.png",
        GRADCAM / "mild_430_correct_overlay.png",
    ]
    gc_w = 280
    gc_total = 3*gc_w + 2*20
    gc_sx = (W - gc_total)//2
    for i, p in enumerate(gc_examples):
        gx = gc_sx + i*(gc_w+20)
        img = overlay_image(img, p, gx, 320, gc_w, gc_w, CYAN)

    draw = ImageDraw.Draw(img)

    # Team info
    team_y = 650
    rect_fill(draw, W//2-500, team_y, 1000, 180, DKBLUE, radius=14)
    draw.rounded_rectangle([W//2-500, team_y, W//2+500, team_y+180],
                            radius=14, outline=CYAN, width=2)
    text_center(draw, team_y+16, "Team Leader: Vijaytejas A C  |  1VK23CS074",
                FONT_BODY, CYAN)
    text_center(draw, team_y+56,
                "Parvati Revannavar  |  Moulya S  |  Priyadarshini K",
                FONT_BODY, WHITE)
    text_center(draw, team_y+98,
                "Guide: Dr. Vidya A, HOD-CSE",
                FONT_BODY, LGRAY)
    text_center(draw, team_y+134,
                "Vivekananda Institute of Technology  |  2025-26",
                FONT_SMALL, LGRAY)

    # QR code bottom right
    qr_x, qr_y = W-380, team_y-10
    img.paste(QR_IMG, (qr_x, qr_y))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([qr_x-6, qr_y-6, qr_x+326, qr_y+326],
                            radius=10, outline=CYAN, width=2)
    text_at(draw, qr_x+10, qr_y+332,
            "Scan to Try the Live Application", FONT_SMALL, LGRAY)

    # Live URL bar
    rect_fill(draw, 60, H-90, W//2-400, 56, (5, 40, 10), radius=8)
    draw.rounded_rectangle([60, H-90, W//2-400, H-34],
                            radius=8, outline=GREEN, width=2)
    draw.text((72, H-78), URL, font=FONT_BODY, fill=GREEN)

    arr = np.array(img)
    for f in slide_in_frames(arr, FPS//2, "up"):
        frames.append(f)
    frames += hold(arr, 5.0)

    # Fade to black
    black = np.zeros_like(arr)
    for f in fade_frames(arr, black, FPS):
        frames.append(f)

    return frames


# ──────────────────────────────────────────────
# ASSEMBLE VIDEO
# ──────────────────────────────────────────────

def build_video():
    print("Building demo video...")

    scenes = [
        ("Scene 1 — QR Code / Access",    scene_01_qr),
        ("Scene 2 — App Opening",          scene_02_app_open),
        ("Scene 3 — Home / Overview",      scene_03_overview),
        ("Scene 4 — MRI Analysis",         scene_04_mri_analysis),
        ("Scene 5 — Prediction Results",   scene_05_prediction),
        ("Scene 6 — Grad-CAM",             scene_06_gradcam),
        ("Scene 7 — Model Comparison",     scene_07_comparison),
        ("Scene 8 — Final Frame",          scene_08_final),
    ]

    all_frames = []
    prev_arr = None

    for name, builder in scenes:
        print(f"  {name} ...")
        scene_frames = builder()

        # Add fade transition between scenes
        if prev_arr is not None and len(scene_frames) > 0:
            fade_n = FPS // 2  # 0.5s fade
            for f in fade_frames(prev_arr, scene_frames[0], fade_n):
                all_frames.append(f)

        all_frames.extend(scene_frames)
        if scene_frames:
            prev_arr = scene_frames[-1].copy()

    total_frames = len(all_frames)
    duration_s = total_frames / FPS
    print(f"  Total frames: {total_frames} ({duration_s:.1f} seconds at {FPS}fps)")

    # Write video
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    writer = cv2.VideoWriter(str(OUT), fourcc, FPS, (W, H))
    if not writer.isOpened():
        print("  [ERROR] VideoWriter failed to open. Trying XVID codec...")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_avi = OUT.with_suffix(".avi")
        writer = cv2.VideoWriter(str(out_avi), fourcc, FPS, (W, H))

    print("  Writing frames to disk...")
    for i, frame in enumerate(all_frames):
        if i % (FPS * 5) == 0:
            print(f"    Frame {i}/{total_frames} ({i/FPS:.1f}s)...")
        bgr = to_bgr(frame)
        writer.write(bgr)

    writer.release()
    print(f"\nDONE: Video saved -> {OUT}")
    print(f"  Duration: {duration_s:.1f} seconds")
    print(f"  Resolution: {W}x{H} @ {FPS}fps")
    return duration_s, total_frames


if __name__ == "__main__":
    duration_s, total_frames = build_video()
