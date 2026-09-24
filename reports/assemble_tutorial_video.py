"""
assemble_tutorial_video.py
Takes the real app screenshots captured by capture_screenshots.py and
assembles them into a professional tutorial MP4 with:
  - Step captions
  - Smooth transitions (fade/slide)
  - Cursor-highlight boxes
  - Step number overlays
  - Progress bar

Run AFTER capture_screenshots.py:
    python reports/assemble_tutorial_video.py
Output:
    reports/Alzheimer_Live_App_User_Tutorial.mp4
"""

import os, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
SHOT_DIR = BASE / "reports" / "screenshots"
OUT      = BASE / "reports" / "Alzheimer_Live_App_User_Tutorial.mp4"

W, H   = 1440, 900
FPS    = 25

# ──────────────────────────────────────────────
# COLOURS
# ──────────────────────────────────────────────
NAVY   = (10,  14,  39)
DKBLUE = (13,  27,  62)
CYAN   = (0,  212, 255)
WHITE  = (255, 255, 255)
LGRAY  = (180, 180, 180)
GREEN  = (0,  229, 118)
RED    = (255,  76,  76)
YELLOW = (255, 215,   0)
BLACK  = (0,    0,   0)
SEMI   = (0,    0,   0)   # used as RGBA via numpy blending

# ──────────────────────────────────────────────
# FONTS
# ──────────────────────────────────────────────
def load_font(size, bold=False):
    opts_b = ["C:/Windows/Fonts/arialbd.ttf",
               "C:/Windows/Fonts/calibrib.ttf"]
    opts_r = ["C:/Windows/Fonts/arial.ttf",
               "C:/Windows/Fonts/calibri.ttf",
               "C:/Windows/Fonts/verdana.ttf"]
    for p in (opts_b if bold else opts_r):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

F_STEP   = load_font(22, bold=True)
F_CAP    = load_font(28, bold=True)
F_SUB    = load_font(18, bold=False)
F_URL    = load_font(16, bold=False)
F_BADGE  = load_font(15, bold=True)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def to_bgr(arr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def load_shot(name: str) -> np.ndarray | None:
    """Load a screenshot as RGB numpy array, resized to W×H."""
    path = SHOT_DIR / f"{name}.png"
    if not path.exists():
        print(f"  [missing] {name}.png")
        return None
    img = Image.open(path).convert("RGB")
    img = img.resize((W, H), Image.LANCZOS)
    return np.array(img)


def add_caption_bar(arr: np.ndarray,
                    step_num: str,
                    caption: str,
                    sub: str = "",
                    bar_h: int = 72) -> np.ndarray:
    """
    Overlay a dark caption bar at the bottom of the frame.
    step_num : e.g. "Step 1"
    caption  : main caption text
    sub      : smaller sub-text
    """
    img = Image.fromarray(arr)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # Dark bar
    d.rectangle([0, H - bar_h, W, H], fill=(10, 14, 39, 210))
    # Cyan left accent
    d.rectangle([0, H - bar_h, 6, H], fill=(0, 212, 255, 255))

    # Step badge
    badge_w = d.textbbox((0,0), step_num, font=F_STEP)[2] + 28
    d.rounded_rectangle([12, H - bar_h + 10,
                          12 + badge_w, H - bar_h + 10 + 34],
                         radius=6, fill=(0, 100, 160, 230))
    d.text((24, H - bar_h + 14), step_num, font=F_STEP, fill=(255,255,255,255))

    # Caption
    cap_x = 12 + badge_w + 18
    d.text((cap_x, H - bar_h + 10), caption, font=F_CAP, fill=(0, 212, 255, 255))

    # Sub text
    if sub:
        d.text((cap_x, H - bar_h + 44), sub, font=F_SUB, fill=(180,180,180,220))

    # Project URL bottom right
    url = "alzheimer-xai-vijay.streamlit.app"
    url_w = d.textbbox((0,0), url, font=F_URL)[2]
    d.text((W - url_w - 16, H - 22), url, font=F_URL, fill=(100,180,100,200))

    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return np.array(img.convert("RGB"))


def add_highlight_box(arr: np.ndarray,
                      x1: int, y1: int, x2: int, y2: int,
                      color=(0, 212, 255), width=3) -> np.ndarray:
    """Draw a coloured rectangle highlight on the frame."""
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    d.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return np.array(img)


def add_top_bar(arr: np.ndarray, title: str = "") -> np.ndarray:
    """Add a thin project title bar at the very top."""
    img = Image.fromarray(arr)
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle([0, 0, W, 32], fill=(13, 27, 62, 220))
    d.rectangle([0, 0, W, 3], fill=(0, 212, 255, 255))
    label = "Alzheimer's Disease Detection & Explainability System — Live Tutorial"
    d.text((16, 6), label, font=F_URL, fill=(200, 200, 200, 200))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return np.array(img.convert("RGB"))


def hold(arr: np.ndarray, seconds: float) -> list:
    return [arr.copy()] * int(seconds * FPS)


def fade(a: np.ndarray, b: np.ndarray, n: int) -> list:
    frames = []
    for i in range(n):
        t = i / n
        frames.append((a * (1-t) + b * t).astype(np.uint8))
    return frames


def slide_in(arr: np.ndarray, n: int = 8) -> list:
    """Quick slide-in from right."""
    frames = []
    for i in range(n):
        t = 1 - (1 - i/n)**2   # ease-out quad
        offset = int(W * (1 - t))
        shifted = np.zeros_like(arr)
        if offset < W:
            shifted[:, :W-offset] = arr[:, offset:]
        frames.append(shifted)
    return frames


def zoom_in(arr: np.ndarray, cx: int, cy: int,
            zoom_start: float = 1.0, zoom_end: float = 1.35,
            n_frames: int = 20) -> list:
    """Smooth zoom into a region of interest."""
    frames = []
    h, w = arr.shape[:2]
    for i in range(n_frames):
        t = i / (n_frames - 1)
        ease = t * t * (3 - 2*t)    # smoothstep
        z = zoom_start + (zoom_end - zoom_start) * ease
        new_w = int(w / z)
        new_h = int(h / z)
        x1 = max(0, min(cx - new_w//2, w - new_w))
        y1 = max(0, min(cy - new_h//2, h - new_h))
        cropped = arr[y1:y1+new_h, x1:x1+new_w]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
        frames.append(resized)
    return frames


def make_title_card(title: str, sub: str = "", color=CYAN) -> np.ndarray:
    """Create a dark title card (used as transition)."""
    img = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 4], fill=color)
    d.rectangle([0, H-4, W, H], fill=color)
    # Title
    bbox = d.textbbox((0, 0), title, font=F_CAP)
    tw = bbox[2] - bbox[0]
    d.text(((W-tw)//2, H//2 - 40), title, font=F_CAP, fill=color)
    if sub:
        sbbox = d.textbbox((0, 0), sub, font=F_SUB)
        sw = sbbox[2] - sbbox[0]
        d.text(((W-sw)//2, H//2 + 20), sub, font=F_SUB, fill=LGRAY)
    return np.array(img)


def make_fallback(caption: str, sub: str = "") -> np.ndarray:
    """Fallback frame when screenshot is missing."""
    arr = make_title_card(caption, sub, color=YELLOW)
    return add_caption_bar(arr, "", caption, sub)

# ──────────────────────────────────────────────
# SCENE DEFINITIONS
# ──────────────────────────────────────────────
# Each scene: (screenshot_name, step_label, caption, sub_text,
#              hold_s, zoom_cx, zoom_cy, zoom_factor)

SCENES = [
    # (shot_name,             step,      caption,                           sub,                                  hold, zx,  zy,  zf)
    ("01b_home_loaded",       "Step 1",  "Open the Project",               "https://alzheimer-xai-vijay.streamlit.app", 3.0, 720, 200, 1.15),
    ("02a_home_overview",     "Step 2",  "Explore the Home Page",          "Project overview & navigation menu",  2.5, 720, 450, 1.1),
    ("02b_home_scrolled",     "Step 2",  "Home Page — Key Features",       "Multi-model AI • Grad-CAM • Calibration", 2.0, 720, 450, 1.0),
    ("03a_mri_page",          "Step 3",  "Open MRI Analysis",              "Navigate to MRI Analysis in the sidebar", 2.5, 720, 300, 1.1),
    ("04c_mri_with_sample",   "Step 4",  "Upload a Brain MRI",             "Select or upload a valid brain MRI scan", 2.5, 720, 450, 1.1),
    ("04d_mri_scrolled",      "Step 4",  "MRI Preview",                    "Image loaded and ready for analysis", 2.0, 720, 450, 1.1),
    ("05a_model_selected",    "Step 5",  "Select a Model",                 "Choose from MobileNetV2, EfficientNet-B0, or ResNet-18", 2.5, 720, 300, 1.15),
    ("06a_after_analyze",     "Step 6",  "Run Analysis",                   "Inference in progress...",            2.5, 720, 450, 1.1),
    ("06b_results_area",      "Step 6",  "Analysis Complete",              "Results are ready",                  2.0, 720, 500, 1.1),
    ("07a_prediction",        "Step 7",  "View the Prediction",            "Predicted class & confidence score",  3.0, 720, 400, 1.25),
    ("07b_probabilities",     "Step 7",  "Class Probabilities",            "Non-Demented | Very Mild | Mild | Moderate", 3.0, 720, 450, 1.2),
    ("08a_gradcam_area",      "Step 8",  "Grad-CAM Explanation",           "Model attribution heatmap",          2.5, 720, 450, 1.1),
    ("08c_gradcam_display",   "Step 8",  "Grad-CAM Heatmap",               "Model attribution — NOT a clinical diagnosis", 3.0, 720, 500, 1.2),
    ("09a_model_comparison",  "Step 9",  "Compare Models",                 "MobileNetV2 vs EfficientNet-B0 vs ResNet-18", 2.5, 720, 400, 1.1),
    ("09b_comparison_charts", "Step 9",  "Model Comparison Charts",        "Real held-out test metrics",         2.5, 720, 500, 1.15),
    ("10a_evaluation",        "Step 10", "Evaluation Results",             "Accuracy, F1, ROC-AUC, Confusion Matrix", 2.5, 720, 400, 1.1),
    ("10b_evaluation_charts", "Step 10", "Evaluation Charts",              "Full benchmark results from held-out test set", 2.5, 720, 500, 1.15),
    ("11a_robustness",        "Step 11", "Error & Robustness",             "Model behavior under image perturbations", 2.0, 720, 450, 1.1),
    ("12a_methodology",       "Step 11", "Research Methodology",           "Training pipeline & research design", 2.0, 720, 450, 1.1),
    ("13a_about",             "Step 11", "About & Disclaimer",             "Academic research prototype — not a medical device", 2.0, 720, 450, 1.1),
    ("14b_final_top",         "Done",    "Complete User Journey Done",     "Explore • Analyze • Explain",         3.5, 720, 450, 1.0),
]

# ──────────────────────────────────────────────
# MAIN ASSEMBLY
# ──────────────────────────────────────────────

def assemble():
    print("Assembling tutorial video from real screenshots...")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT), fourcc, FPS, (W, H))
    if not writer.isOpened():
        print("  [ERROR] VideoWriter could not open output file.")
        return

    all_frame_count = 0
    prev_arr = None

    # --- Intro title card ---
    intro = make_title_card(
        "Alzheimer's Disease Detection & Explainability System",
        "Step-by-Step User Tutorial  |  alzheimer-xai-vijay.streamlit.app"
    )
    for f in slide_in(intro, 12):
        writer.write(to_bgr(f))
    for f in hold(intro, 2.0):
        writer.write(to_bgr(f))
    all_frame_count += 12 + int(2.0*FPS)
    prev_arr = intro

    for i, (shot_name, step, caption, sub, hold_s, zx, zy, zf) in enumerate(SCENES):
        arr = load_shot(shot_name)

        # Use fallback if screenshot missing
        if arr is None:
            arr = make_fallback(caption, sub)
            print(f"  [fallback] scene {i+1}: {shot_name}")

        # Add top bar + caption
        arr = add_top_bar(arr)
        arr_captioned = add_caption_bar(arr, step, caption, sub)

        # Fade in from previous
        if prev_arr is not None:
            prev_cap = add_top_bar(prev_arr)
            prev_cap = add_caption_bar(prev_cap, "", "", "")
            fade_n = int(FPS * 0.4)
            for f in fade(prev_cap, arr_captioned, fade_n):
                writer.write(to_bgr(f))
            all_frame_count += fade_n

        # Slide in
        for f in slide_in(arr_captioned, 8):
            writer.write(to_bgr(f))
        all_frame_count += 8

        # Hold at normal zoom briefly
        hold_frames_1 = hold(arr_captioned, hold_s * 0.4)
        for f in hold_frames_1:
            writer.write(to_bgr(f))
        all_frame_count += len(hold_frames_1)

        # Zoom in to point of interest
        if zf > 1.01:
            zoom_n = int(FPS * 0.8)
            for f in zoom_in(arr_captioned, zx, zy, 1.0, zf, zoom_n):
                capt_f = add_caption_bar(
                    Image.fromarray(f).__array__() if False else f,
                    step, caption, sub
                )
                writer.write(to_bgr(capt_f))
            all_frame_count += zoom_n

            # Hold at zoomed level
            zoomed_final = zoom_in(arr_captioned, zx, zy, zf, zf, 1)[0]
            zoomed_final = add_caption_bar(zoomed_final, step, caption, sub)
            hold_frames_2 = hold(zoomed_final, hold_s * 0.6)
            for f in hold_frames_2:
                writer.write(to_bgr(f))
            all_frame_count += len(hold_frames_2)
            prev_arr_raw = zoomed_final
        else:
            hold_frames_2 = hold(arr_captioned, hold_s * 0.6)
            for f in hold_frames_2:
                writer.write(to_bgr(f))
            all_frame_count += len(hold_frames_2)
            prev_arr_raw = arr_captioned

        prev_arr = arr  # use un-captioned for next fade

        print(f"  Scene {i+1:02d}/{len(SCENES)} done: {shot_name}")

    # --- Outro title card ---
    outro_lines = [
        "Alzheimer's Disease Detection & Explainability System",
    ]
    outro = make_title_card(
        "Alzheimer's Disease Detection & Explainability System",
        "Live at: https://alzheimer-xai-vijay.streamlit.app"
    )
    # Add team info
    outro_img = Image.fromarray(outro)
    d = ImageDraw.Draw(outro_img)
    team_lines = [
        "Team Leader: Vijaytejas A C | 1VK23CS074",
        "Parvati Revannavar  |  Moulya S  |  Priyadarshini K",
        "Guide: Dr. Vidya A, HOD-CSE  |  Vivekananda Institute of Technology",
    ]
    for j, line in enumerate(team_lines):
        bbox = d.textbbox((0,0), line, font=F_SUB)
        tw = bbox[2]-bbox[0]
        d.text(((W-tw)//2, H//2 + 70 + j*28), line, font=F_SUB, fill=LGRAY)
    outro = np.array(outro_img)

    fade_n = int(FPS * 0.5)
    for f in fade(prev_arr_raw, outro, fade_n):
        writer.write(to_bgr(f))
    all_frame_count += fade_n

    for f in hold(outro, 4.0):
        writer.write(to_bgr(f))
    all_frame_count += int(4.0*FPS)

    # Final fade to black
    black = np.zeros_like(outro)
    for f in fade(outro, black, int(FPS*1.0)):
        writer.write(to_bgr(f))
    all_frame_count += int(FPS*1.0)

    writer.release()

    duration = all_frame_count / FPS
    size_mb = OUT.stat().st_size / (1024*1024)
    print(f"\nDONE")
    print(f"  Output: {OUT}")
    print(f"  Frames: {all_frame_count}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Resolution: {W}x{H} @ {FPS}fps")
    print(f"  File size: {size_mb:.1f} MB")
    return duration


if __name__ == "__main__":
    assemble()
