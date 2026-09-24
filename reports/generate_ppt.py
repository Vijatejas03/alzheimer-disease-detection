"""
generate_ppt.py
Generates the final-year engineering project PowerPoint presentation for
"Explainable Deep Learning-Based Multi-Stage Alzheimer's Disease Detection Using Brain MRI"

Run:
  python reports/generate_ppt.py
Output:
  reports/Alzheimer_Final_Project_Presentation.pptx
"""

import io
import os
import sys
import qrcode
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml
from lxml import etree

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE = Path(__file__).parent.parent  # project root
FIGS = BASE / "results" / "figures"
GRADCAM = FIGS / "gradcam" / "efficientnet_b0"
CALIB   = FIGS / "calibration"
ROBUST  = FIGS / "robustness"
ERR     = FIGS / "error_analysis"
OUT     = BASE / "reports" / "Alzheimer_Final_Project_Presentation.pptx"

# ──────────────────────────────────────────────
# COLOUR PALETTE
# ──────────────────────────────────────────────
C_NAVY      = RGBColor(0x0A, 0x0E, 0x27)   # slide background
C_DKBLUE    = RGBColor(0x0D, 0x1B, 0x3E)   # panel background
C_CYAN      = RGBColor(0x00, 0xD4, 0xFF)   # accent / headings
C_ELECBLUE  = RGBColor(0x00, 0x8C, 0xFF)   # secondary accent
C_PURPLE    = RGBColor(0x7B, 0x2F, 0xBE)   # tertiary accent
C_WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
C_LGRAY     = RGBColor(0xCC, 0xCC, 0xCC)
C_GREEN     = RGBColor(0x00, 0xE5, 0x76)
C_RED       = RGBColor(0xFF, 0x4C, 0x4C)
C_YELLOW    = RGBColor(0xFF, 0xD7, 0x00)
C_ORANGE    = RGBColor(0xFF, 0x8C, 0x00)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_layout(prs):
    return prs.slide_layouts[6]  # completely blank


def add_slide(prs) -> object:
    return prs.slides.add_slide(blank_layout(prs))


def bg(slide, color: RGBColor = C_NAVY):
    """Fill slide background with solid colour."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def rect(slide, l, t, w, h, fill: RGBColor, alpha=None):
    """Add a filled rectangle shape."""
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        l, t, w, h
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    return shape


def txb(slide, text, l, t, w, h,
        size=18, bold=False, color: RGBColor = C_WHITE,
        align=PP_ALIGN.LEFT, italic=False, wrap=True):
    """Add a text box."""
    txf = slide.shapes.add_textbox(l, t, w, h)
    tf = txf.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txf


def heading(slide, text, t=Inches(0.25), size=28, color=C_CYAN):
    txb(slide, text,
        l=Inches(0.4), t=t, w=Inches(12.5), h=Inches(0.55),
        size=size, bold=True, color=color, align=PP_ALIGN.LEFT)


def sub_heading(slide, text, t, size=14, color=C_LGRAY):
    txb(slide, text,
        l=Inches(0.4), t=t, w=Inches(12.5), h=Inches(0.4),
        size=size, bold=False, color=color, align=PP_ALIGN.LEFT)


def divider(slide, t):
    """Thin cyan horizontal rule."""
    rect(slide, Inches(0.4), t, Inches(12.5), Pt(2), C_CYAN)


def add_image(slide, path, l, t, w, h=None):
    """Add an image; h=None keeps aspect ratio via width only."""
    path = str(path)
    if not os.path.exists(path):
        return None
    if h is None:
        pic = slide.shapes.add_picture(path, l, t, width=w)
    else:
        pic = slide.shapes.add_picture(path, l, t, width=w, height=h)
    return pic


def card(slide, l, t, w, h, fill=C_DKBLUE, border=C_CYAN):
    """Rounded-corner card (simulated with rectangle + border)."""
    s = slide.shapes.add_shape(1, l, t, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = border
    s.line.width = Pt(1.2)
    return s


def pill(slide, text, l, t, w, h, bg_color=None, txt_color=C_WHITE, size=12, bold=True, fill=None):
    """Coloured pill / badge."""
    effective_color = fill if fill is not None else bg_color
    s = card(slide, l, t, w, h, fill=effective_color, border=effective_color)
    txb(slide, text, l+Inches(0.05), t+Inches(0.04),
        w-Inches(0.1), h-Inches(0.08),
        size=size, bold=bold, color=txt_color, align=PP_ALIGN.CENTER)


def footer(slide, page_num, total=24):
    """Slide number + project name footer."""
    rect(slide, 0, SLIDE_H - Inches(0.28), SLIDE_W, Inches(0.28), C_DKBLUE)
    txb(slide,
        "Alzheimer's Disease Detection & Explainability System  •  Final Year Engineering Project",
        Inches(0.4), SLIDE_H - Inches(0.28), Inches(11.5), Inches(0.28),
        size=8, color=C_LGRAY)
    txb(slide, f"{page_num} / {total}",
        Inches(12.4), SLIDE_H - Inches(0.28), Inches(0.8), Inches(0.28),
        size=8, color=C_CYAN, align=PP_ALIGN.RIGHT)


def qr_code_image(url: str) -> io.BytesIO:
    qr = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="white", back_color="#0A0E27")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════
# SLIDE BUILDERS
# ══════════════════════════════════════════════

# ── SLIDE 1 · Title ──────────────────────────
def slide_01_title(prs):
    s = add_slide(prs)
    bg(s, C_NAVY)

    # Left accent bar
    rect(s, 0, 0, Inches(0.08), SLIDE_H, C_CYAN)

    # Top gradient band
    rect(s, 0, 0, SLIDE_W, Inches(0.08), C_CYAN)

    # Brain MRI background image (low opacity via layering)
    # Use one of the Grad-CAM overlays as a background motif
    panel_path = GRADCAM / "mild_430_correct_overlay.png"
    if panel_path.exists():
        pic = add_image(s, panel_path, Inches(7.8), Inches(0.5), Inches(5.2))
        # Set transparency via CXML
        try:
            sp = pic._element
            sp_pr = sp.find(qn('p:spPr'))
            if sp_pr is None:
                sp_pr = etree.SubElement(sp, qn('p:spPr'))
            blipFill = sp.find('.//' + qn('p:blipFill'))
            if blipFill is None:
                blipFill = sp.find('.//' + qn('pic:blipFill'))
        except Exception:
            pass

    # Decorative card behind text
    rect(s, Inches(0.1), Inches(0.6), Inches(7.5), Inches(5.8), RGBColor(0x0D, 0x1B, 0x3E))

    # Institution badge
    pill(s, "⚕  Department of Computer Science & Engineering",
         Inches(0.25), Inches(0.7), Inches(7.0), Inches(0.38),
         RGBColor(0x00, 0x50, 0x8C), size=11)

    # Main title
    txb(s, "Explainable Deep Learning-Based",
        Inches(0.25), Inches(1.25), Inches(7.2), Inches(0.6),
        size=22, bold=True, color=C_CYAN)
    txb(s, "Multi-Stage Alzheimer's Disease",
        Inches(0.25), Inches(1.8), Inches(7.2), Inches(0.6),
        size=28, bold=True, color=C_WHITE)
    txb(s, "Detection Using Brain MRI",
        Inches(0.25), Inches(2.35), Inches(7.2), Inches(0.6),
        size=28, bold=True, color=C_WHITE)

    # Divider
    rect(s, Inches(0.25), Inches(3.05), Inches(7.0), Pt(2), C_CYAN)

    # Subtitle
    txb(s, "Alzheimer's Disease Detection & Explainability System",
        Inches(0.25), Inches(3.15), Inches(7.2), Inches(0.4),
        size=13, italic=True, color=C_LGRAY)

    # Team info block
    txb(s, "Team Members:",
        Inches(0.25), Inches(3.7), Inches(3.5), Inches(0.35),
        size=11, bold=True, color=C_CYAN)
    txb(s, "Parvati Revannavar  |  1VK23CS048",
        Inches(0.25), Inches(4.02), Inches(7.0), Inches(0.32),
        size=10, color=C_WHITE)
    txb(s, "Moulya S  |  1VK23CS042",
        Inches(0.25), Inches(4.3), Inches(7.0), Inches(0.32),
        size=10, color=C_WHITE)
    txb(s, "Priyadarshini K  |  1VK23CS051",
        Inches(0.25), Inches(4.58), Inches(7.0), Inches(0.32),
        size=10, color=C_WHITE)
    txb(s, "Vijaytejas A C  |  1VK23CS074",
        Inches(0.25), Inches(4.86), Inches(7.0), Inches(0.32),
        size=10, color=C_WHITE)

    txb(s, "Project Guide:",
        Inches(0.25), Inches(5.25), Inches(3.5), Inches(0.32),
        size=11, bold=True, color=C_CYAN)
    txb(s, "Dr. Vidya A   |   Head of the Department, CSE",
        Inches(0.25), Inches(5.52), Inches(7.0), Inches(0.32),
        size=10, color=C_WHITE)

    txb(s, "Vivekananda Institute of Technology  |  Academic Year 2025-26",
        Inches(0.25), Inches(5.88), Inches(7.2), Inches(0.32),
        size=10, color=C_LGRAY)


    # Technology pills at bottom-left
    for i, (label, col) in enumerate([
        ("PyTorch", RGBColor(0xEE, 0x4C, 0x2C)),
        ("Grad-CAM", C_PURPLE),
        ("Streamlit", RGBColor(0xFF, 0x4B, 0x4B)),
        ("Python 3.12", RGBColor(0x36, 0x76, 0xBE)),
    ]):
        pill(s, label, Inches(0.25 + i * 1.55), Inches(6.15),
             Inches(1.4), Inches(0.32), col, size=9)

    footer(s, 1)
    return s


# ── SLIDE 2 · Project at a Glance ────────────
def slide_02_glance(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Project at a Glance", size=26)
    divider(s, Inches(0.9))

    sub_heading(s, "End-to-end pipeline: from raw brain MRI to explainable multi-model prediction", Inches(0.95))

    steps = [
        ("🧠", "Brain MRI\nDataset", "6,400 images\n4 classes"),
        ("⚙️",  "Pre-\nprocessing", "128×128\nGrayscale JPEG"),
        ("🔀", "Train/Val/\nTest Split", "70 / 15 / 15%\n0 hash overlap"),
        ("🤖", "3 CNN\nModels", "MobileNetV2\nEfficientNet-B0\nResNet-18"),
        ("📊", "Agreement\nEngine", "3-model\nconsensus\nvoting"),
        ("🔍", "Grad-CAM\nXAI", "Saliency\nvisualization"),
        ("🌐", "Streamlit\nApp", "8-page\ndeployed UI"),
    ]

    box_w = Inches(1.55)
    box_h = Inches(2.2)
    gap = Inches(0.22)
    start_x = Inches(0.3)
    start_y = Inches(1.6)

    for i, (icon, title, detail) in enumerate(steps):
        x = start_x + i * (box_w + gap)
        c = card(s, x, start_y, box_w, box_h, fill=C_DKBLUE, border=C_CYAN)
        txb(s, icon, x + Inches(0.1), start_y + Inches(0.12),
            box_w - Inches(0.2), Inches(0.45), size=22, align=PP_ALIGN.CENTER)
        txb(s, title, x + Inches(0.05), start_y + Inches(0.6),
            box_w - Inches(0.1), Inches(0.65),
            size=11, bold=True, color=C_CYAN, align=PP_ALIGN.CENTER)
        txb(s, detail, x + Inches(0.05), start_y + Inches(1.25),
            box_w - Inches(0.1), Inches(0.85),
            size=9, color=C_LGRAY, align=PP_ALIGN.CENTER)
        # Arrow
        if i < len(steps) - 1:
            ax = x + box_w + Inches(0.04)
            txb(s, "→", ax, start_y + Inches(0.85), gap + Inches(0.05),
                Inches(0.4), size=14, color=C_CYAN, align=PP_ALIGN.CENTER)

    # Key numbers strip
    kw = Inches(3.0)
    kh = Inches(0.9)
    kstats = [
        ("6,400", "Unique MRI images"),
        ("3", "CNN architectures"),
        ("98.44%", "Peak test accuracy"),
        ("0.9991", "Peak ROC-AUC"),
    ]
    for i, (val, label) in enumerate(kstats):
        x = Inches(0.4) + i * (kw + Inches(0.2))
        c = card(s, x, Inches(4.25), kw, kh,
                 fill=RGBColor(0x05, 0x28, 0x4A), border=C_CYAN)
        txb(s, val, x + Inches(0.1), Inches(4.28),
            kw - Inches(0.2), Inches(0.5),
            size=22, bold=True, color=C_CYAN, align=PP_ALIGN.CENTER)
        txb(s, label, x + Inches(0.1), Inches(4.72),
            kw - Inches(0.2), Inches(0.35),
            size=9, color=C_LGRAY, align=PP_ALIGN.CENTER)

    footer(s, 2)
    return s


# ── SLIDE 3 · Problem Statement ──────────────
def slide_03_problem(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Problem Statement", size=26)
    divider(s, Inches(0.9))

    # Left column — text
    problems = [
        ("🧬", "Alzheimer's disease is progressive & irreversible — early detection is critical."),
        ("🔢", "4-stage clinical classification: Non-Demented → Very Mild → Mild → Moderate Demented."),
        ("🏥", "Manual MRI interpretation is time-consuming and subject to inter-rater variability."),
        ("⚠️",  "Single-model AI systems are opaque, untrustworthy for research review, and lack explainability."),
        ("📉", "Severe class imbalance: Moderate class has only 64 samples vs 3,200 Non-Demented."),
    ]

    ty = Inches(1.05)
    for icon, text in problems:
        card(s, Inches(0.3), ty, Inches(7.8), Inches(0.72), C_DKBLUE, C_PURPLE)
        txb(s, icon, Inches(0.4), ty + Inches(0.12), Inches(0.5), Inches(0.5), size=18)
        txb(s, text, Inches(0.95), ty + Inches(0.12), Inches(7.0), Inches(0.55),
            size=12, color=C_WHITE)
        ty += Inches(0.82)

    # Humor callout
    card(s, Inches(0.3), ty + Inches(0.1), Inches(7.8), Inches(0.65),
         RGBColor(0x1A, 0x2A, 0x1A), RGBColor(0x00, 0xCC, 0x44))
    txb(s, "🤔  An AI should not classify everything you upload. Garbage in → garbage prediction. 😅",
        Inches(0.45), ty + Inches(0.2), Inches(7.5), Inches(0.45),
        size=11, italic=True, color=RGBColor(0x00, 0xCC, 0x44))

    # Right side — brain silhouette area with stats
    card(s, Inches(8.5), Inches(1.05), Inches(4.6), Inches(5.4),
         fill=RGBColor(0x0D, 0x1B, 0x3E), border=C_CYAN)
    txb(s, "🧠", Inches(8.9), Inches(1.15), Inches(3.8), Inches(0.9),
        size=48, align=PP_ALIGN.CENTER)
    txb(s, "Global Alzheimer's Burden", Inches(8.55), Inches(2.1),
        Inches(4.5), Inches(0.4),
        size=13, bold=True, color=C_CYAN, align=PP_ALIGN.CENTER)

    stats = [
        ("55M+",  "People with dementia worldwide (2023)"),
        ("60–70%","Cases are Alzheimer's type"),
        ("3×",    "Expected increase by 2050"),
        ("$1.3T", "Global annual care cost"),
    ]
    for i, (val, desc) in enumerate(stats):
        ty2 = Inches(2.65) + i * Inches(0.65)
        txb(s, val, Inches(8.55), ty2, Inches(1.5), Inches(0.5),
            size=18, bold=True, color=C_CYAN, align=PP_ALIGN.RIGHT)
        txb(s, desc, Inches(10.2), ty2 + Inches(0.06), Inches(2.8), Inches(0.45),
            size=10, color=C_LGRAY)

    footer(s, 3)


# ── SLIDE 4 · Motivation ──────────────────────
def slide_04_motivation(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Motivation", size=26)
    divider(s, Inches(0.9))

    cards = [
        (C_CYAN,    "🔬", "Explainability Gap",
         "Most deep learning models are black-boxes. Clinicians and researchers\nneed to understand WHY a model predicts what it predicts."),
        (C_PURPLE,  "📊", "Multi-Model Reliability",
         "A single CNN can overfit or fail silently. A 3-model agreement\nengine provides cross-validated consensus confidence."),
        (C_GREEN,   "⚖️",  "Class Imbalance Challenge",
         "64 Moderate vs 3,200 Non-Demented samples. Standard accuracy\nis misleading without weighted loss & balanced metrics."),
        (C_ORANGE,  "🛡️", "Input Safety / OOD Rejection",
         "Real-world deployment requires input-domain safeguards.\nDog photos, chest CT scans → REJECTED before inference."),
    ]

    cw = Inches(5.9)
    ch = Inches(2.2)
    positions = [
        (Inches(0.35), Inches(1.1)),
        (Inches(6.75), Inches(1.1)),
        (Inches(0.35), Inches(3.55)),
        (Inches(6.75), Inches(3.55)),
    ]

    for (cx, cy), (accent, icon, title, body) in zip(positions, cards):
        card(s, cx, cy, cw, ch, fill=C_DKBLUE, border=accent)
        # Top accent strip
        rect(s, cx, cy, cw, Inches(0.06), accent)
        txb(s, icon + "  " + title,
            cx + Inches(0.15), cy + Inches(0.15),
            cw - Inches(0.3), Inches(0.45),
            size=14, bold=True, color=accent)
        txb(s, body,
            cx + Inches(0.15), cy + Inches(0.65),
            cw - Inches(0.3), Inches(1.45),
            size=11, color=C_WHITE)

    footer(s, 4)


# ── SLIDE 5 · Objectives ──────────────────────
def slide_05_objectives(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Project Objectives", size=26)
    divider(s, Inches(0.9))

    objectives = [
        ("01", "Multi-Class MRI Classification",
         "Classify brain MRI scans into 4 Alzheimer's stages using 3 fine-tuned CNN architectures."),
        ("02", "Explainability via Grad-CAM",
         "Generate Gradient-weighted Class Activation Maps for model attribution and research transparency."),
        ("03", "3-Model Agreement Engine",
         "Implement consensus voting across MobileNetV2, EfficientNet-B0, ResNet-18 for reliability."),
        ("04", "Calibration & Uncertainty",
         "Apply temperature scaling; report ECE, Brier score and per-image confidence calibration."),
        ("05", "Robustness Stress Testing",
         "Evaluate performance under noise, blur, brightness/contrast shifts and rotation perturbations."),
        ("06", "Input-Domain Safeguard",
         "Reject OOD inputs (non-brain images) before inference using 8-stage domain screening pipeline."),
    ]

    ow = Inches(5.85)
    oh = Inches(1.1)
    for i, (num, title, body) in enumerate(objectives):
        row, col = divmod(i, 2)
        ox = Inches(0.35) + col * (ow + Inches(0.5))
        oy = Inches(1.1) + row * (oh + Inches(0.22))
        card(s, ox, oy, ow, oh, fill=C_DKBLUE, border=C_CYAN)
        rect(s, ox, oy, Inches(0.55), oh, RGBColor(0x00, 0x50, 0x8C))
        txb(s, num, ox + Inches(0.05), oy + Inches(0.3),
            Inches(0.45), Inches(0.5), size=14, bold=True,
            color=C_CYAN, align=PP_ALIGN.CENTER)
        txb(s, title, ox + Inches(0.65), oy + Inches(0.08),
            ow - Inches(0.75), Inches(0.38),
            size=12, bold=True, color=C_CYAN)
        txb(s, body, ox + Inches(0.65), oy + Inches(0.48),
            ow - Inches(0.75), Inches(0.55),
            size=10, color=C_LGRAY)

    footer(s, 5)


# ── SLIDE 6 · Literature Survey ──────────────
def slide_06_literature(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Literature Survey", size=26)
    divider(s, Inches(0.9))
    sub_heading(s, "Selected prior works in CNN-based Alzheimer's MRI classification", Inches(0.95))

    papers = [
        ("Farooq et al., 2017",     "ADNI MRI",          "4-layer CNN",         "Early CNN for 4-class AD classification",            "Small dataset; no explainability"),
        ("Wen et al., 2020",        "ADNI (3D MRI)",      "3D CNN / SVM",        "Benchmark for 3D vs 2D MRI classification",          "Computationally expensive; no mobile deployment"),
        ("Shanmugam et al., 2022",  "Kaggle MRI dataset", "ResNet-50",           "Transfer learning for AD staging",                   "Single model; no uncertainty or calibration"),
        ("Islam & Zhang, 2018",     "OASIS MRI",          "AlexNet / VGG",       "Multi-class transfer learning with data augmentation","No explainability; imbalance not addressed"),
        ("Spasov et al., 2019",     "ADNI",               "CNN + LSTM",          "MCI-to-AD conversion prediction",                    "Binary task only; no multi-stage classification"),
        ("Odusami et al., 2021",    "OASIS / ADNI",       "ResNet + Grad-CAM",   "Grad-CAM XAI applied to AD MRI prediction",          "Single model; limited robustness analysis"),
    ]

    # Header row
    cols = ["Paper / Year", "Dataset", "Model", "Main Contribution", "Limitation / Gap"]
    col_widths = [Inches(2.1), Inches(1.5), Inches(1.5), Inches(4.4), Inches(3.0)]
    hx = Inches(0.3)
    hy = Inches(1.3)
    hh = Inches(0.38)
    for j, (col, cw) in enumerate(zip(cols, col_widths)):
        rect(s, hx, hy, cw, hh, RGBColor(0x00, 0x50, 0x8C))
        txb(s, col, hx + Inches(0.06), hy + Inches(0.04),
            cw - Inches(0.12), hh,
            size=10, bold=True, color=C_WHITE)
        hx += cw

    # Data rows
    for ri, paper in enumerate(papers):
        row_fill = C_DKBLUE if ri % 2 == 0 else RGBColor(0x10, 0x22, 0x44)
        rx = Inches(0.3)
        ry = Inches(1.68) + ri * Inches(0.74)
        rh = Inches(0.7)
        for j, (val, cw) in enumerate(zip(paper, col_widths)):
            color = C_CYAN if j == 0 else (C_YELLOW if j == 4 else C_WHITE)
            rect(s, rx, ry, cw, rh, row_fill)
            txb(s, val, rx + Inches(0.06), ry + Inches(0.04),
                cw - Inches(0.12), rh - Inches(0.08),
                size=9, color=color)
            rx += cw

    # Research Gap callout
    card(s, Inches(0.3), Inches(6.25), Inches(12.6), Inches(0.5),
         fill=RGBColor(0x1A, 0x0A, 0x2A), border=C_PURPLE)
    txb(s, "🔍  Research Gap: None of the above studies combine 3-model consensus, temperature calibration, "
            "Grad-CAM XAI, robustness stress-testing, and OOD rejection in a single deployable research system.",
        Inches(0.45), Inches(6.3), Inches(12.3), Inches(0.44),
        size=10, italic=True, color=C_PURPLE)

    footer(s, 6)


# ── SLIDE 7 · Existing / Baseline System ─────
def slide_07_baseline(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Reference / Baseline Approach", size=26)
    divider(s, Inches(0.9))
    sub_heading(s, "Typical single-model MRI classification pipeline used as reference", Inches(0.95))

    steps = [
        ("🧠\nMRI Image", C_CYAN),
        ("↓\nBasic\nResizing", C_LGRAY),
        ("↓\nSingle CNN\n(ResNet / VGG)", RGBColor(0x88, 0x88, 0xFF)),
        ("↓\nSoftmax\nOutput", C_LGRAY),
        ("↓\nFlask/Static\nWeb App", RGBColor(0xFF, 0x8C, 0x00)),
        ("↓\nPredicted\nClass", C_GREEN),
    ]

    bx = Inches(1.0)
    by = Inches(1.4)
    bw = Inches(1.7)
    bh = Inches(1.4)
    bgap = Inches(0.3)

    for i, (label, col) in enumerate(steps):
        cx = bx + i * (bw + bgap)
        card(s, cx, by, bw, bh, fill=C_DKBLUE, border=col)
        txb(s, label, cx + Inches(0.05), by + Inches(0.15),
            bw - Inches(0.1), bh - Inches(0.3),
            size=11, color=col, bold=True, align=PP_ALIGN.CENTER)

    # Limitations
    txb(s, "Limitations of the Reference Approach",
        Inches(0.4), Inches(3.15), Inches(12.5), Inches(0.4),
        size=14, bold=True, color=C_RED)

    lims = [
        ("❌", "Single model — no cross-validation or consensus confidence"),
        ("❌", "No explainability (black-box prediction)"),
        ("❌", "No calibration — confidence scores are uncalibrated"),
        ("❌", "No robustness testing under image perturbations"),
        ("❌", "No input-domain safeguard — accepts any uploaded image"),
        ("❌", "No uncertainty quantification"),
    ]
    for i, (icon, text) in enumerate(lims):
        row, col_idx = divmod(i, 2)
        lx = Inches(0.4) + col_idx * Inches(6.3)
        ly = Inches(3.65) + row * Inches(0.62)
        card(s, lx, ly, Inches(6.1), Inches(0.55), fill=RGBColor(0x2A, 0x0A, 0x0A),
             border=C_RED)
        txb(s, icon + "  " + text, lx + Inches(0.12), ly + Inches(0.09),
            Inches(5.8), Inches(0.4), size=11, color=C_WHITE)

    footer(s, 7)


# ── SLIDE 8 · Research Gap ───────────────────
def slide_08_gap(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Research Gap Analysis", size=26)
    divider(s, Inches(0.9))

    features = [
        "Multi-class 4-stage classification",
        "3-model consensus engine",
        "Grad-CAM explainability",
        "Temperature calibration (ECE/Brier)",
        "Robustness stress testing",
        "OOD / input-domain safeguard",
        "Deployable web application",
        "Open-source reproducible research",
    ]

    # Column headers
    col_labels = ["Feature", "Typical Baseline", "This Research System"]
    col_ws = [Inches(5.5), Inches(2.9), Inches(4.5)]
    hx = Inches(0.3)
    for j, (label, cw) in enumerate(zip(col_labels, col_ws)):
        col = C_CYAN if j == 2 else RGBColor(0x33, 0x44, 0x55)
        rect(s, hx, Inches(1.05), cw, Inches(0.38),
             RGBColor(0x00, 0x50, 0x8C) if j != 2 else RGBColor(0x00, 0x6A, 0x44))
        txb(s, label, hx + Inches(0.08), Inches(1.08),
            cw - Inches(0.16), Inches(0.35),
            size=12, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        hx += cw

    for ri, feat in enumerate(features):
        ry = Inches(1.43) + ri * Inches(0.59)
        rh = Inches(0.56)
        row_fill = C_DKBLUE if ri % 2 == 0 else RGBColor(0x10, 0x22, 0x44)
        # Feature name
        rect(s, Inches(0.3), ry, col_ws[0], rh, row_fill)
        txb(s, "•  " + feat, Inches(0.38), ry + Inches(0.09),
            col_ws[0] - Inches(0.16), rh - Inches(0.18),
            size=11, color=C_WHITE)
        # Baseline
        rect(s, Inches(0.3) + col_ws[0], ry, col_ws[1], rh,
             row_fill)
        txb(s, "❌  Partial / None",
            Inches(0.3) + col_ws[0] + Inches(0.1), ry + Inches(0.1),
            col_ws[1] - Inches(0.2), rh - Inches(0.2),
            size=11, color=C_RED, align=PP_ALIGN.CENTER)
        # Ours
        rect(s, Inches(0.3) + col_ws[0] + col_ws[1], ry, col_ws[2], rh,
             RGBColor(0x0A, 0x22, 0x18))
        txb(s, "✅  Implemented",
            Inches(0.3) + col_ws[0] + col_ws[1] + Inches(0.1), ry + Inches(0.1),
            col_ws[2] - Inches(0.2), rh - Inches(0.2),
            size=11, color=C_GREEN, align=PP_ALIGN.CENTER)

    footer(s, 8)


# ── SLIDE 9 · Proposed System ─────────────────
def slide_09_proposed(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Proposed System Architecture", size=26)
    divider(s, Inches(0.9))

    # Main pipeline — horizontal flow
    stages = [
        ("INPUT", "Brain MRI\n(User upload\nor demo scan)", C_CYAN),
        ("VALIDATE", "8-Stage\nDomain\nScreening", C_YELLOW),
        ("PREPROCESS", "Resize 128×128\nNormalize\nGrayscale→RGB", C_LGRAY),
        ("INFER", "MobileNetV2\n+\nEfficientNet-B0\n+\nResNet-18", C_PURPLE),
        ("FUSE", "3-Model\nAgreement\nEngine", C_ORANGE),
        ("EXPLAIN", "Grad-CAM\nSaliency Map\nGeneration", RGBColor(0x00, 0xBF, 0xFF)),
        ("OUTPUT", "Prediction\nProbabilities\nUncertainty\nCalibration", C_GREEN),
    ]

    bw = Inches(1.55)
    bh = Inches(2.4)
    gap = Inches(0.22)
    sx = Inches(0.25)
    sy = Inches(1.15)

    for i, (tag, label, col) in enumerate(stages):
        cx = sx + i * (bw + gap)
        # Card
        card(s, cx, sy, bw, bh, fill=C_DKBLUE, border=col)
        rect(s, cx, sy, bw, Inches(0.05), col)
        # Tag badge
        pill(s, tag, cx + Inches(0.15), sy + Inches(0.1),
             bw - Inches(0.3), Inches(0.28), col, size=8)
        txb(s, label, cx + Inches(0.06), sy + Inches(0.47),
            bw - Inches(0.12), bh - Inches(0.55),
            size=10, bold=False, color=C_WHITE, align=PP_ALIGN.CENTER)
        # Arrow
        if i < len(stages) - 1:
            ax = cx + bw + Inches(0.03)
            txb(s, "➔", ax, sy + Inches(1.0), gap + Inches(0.06),
                Inches(0.4), size=14, color=col, align=PP_ALIGN.CENTER)

    # REJECT branch
    reject_x = Inches(0.25) + (bw + gap)  # under VALIDATE box
    txb(s, "⛔ OOD / Non-Brain Image", Inches(1.8), Inches(3.75), Inches(2.5), Inches(0.4),
        size=10, bold=True, color=C_RED)
    txb(s, "↙", Inches(1.8), Inches(3.48), Inches(0.4), Inches(0.35),
        size=16, color=C_RED)

    card(s, Inches(0.3), Inches(4.1), Inches(3.5), Inches(0.85),
         fill=RGBColor(0x2A, 0x0A, 0x0A), border=C_RED)
    txb(s, '🚫  "Input rejected"\nThis image does not appear to be a suitable brain MRI.',
        Inches(0.4), Inches(4.18), Inches(3.3), Inches(0.72),
        size=9, color=C_RED)

    # Sub-components row
    subs = [
        ("Streamlit UI", "8-page web\napplication", RGBColor(0xFF, 0x4B, 0x4B)),
        ("PyTorch", "CUDA 12.4\nAMP training", RGBColor(0xEE, 0x4C, 0x2C)),
        ("Temp. Scaling", "ECE / Brier\ncalibration", C_CYAN),
        ("Robustness", "5 perturbation\ncategories", C_ORANGE),
    ]
    sw = Inches(2.9)
    sx2 = Inches(4.0)
    for i, (title, detail, col) in enumerate(subs):
        cx = sx2 + i * (sw + Inches(0.2))
        card(s, cx, Inches(3.95), sw, Inches(0.95), fill=C_DKBLUE, border=col)
        txb(s, title, cx + Inches(0.1), Inches(4.0),
            sw - Inches(0.2), Inches(0.38), size=11, bold=True, color=col)
        txb(s, detail, cx + Inches(0.1), Inches(4.38),
            sw - Inches(0.2), Inches(0.45), size=9, color=C_LGRAY)

    footer(s, 9)


# ── SLIDE 10 · Dataset ────────────────────────
def slide_10_dataset(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Dataset", size=26)
    divider(s, Inches(0.9))

    # Left — stats
    card(s, Inches(0.3), Inches(1.05), Inches(5.5), Inches(5.7),
         fill=C_DKBLUE, border=C_CYAN)

    stats = [
        ("6,400", "Total unique images", C_CYAN),
        ("3,200", "Non-Demented (50%)", C_GREEN),
        ("2,240", "Very Mild Demented (35%)", C_YELLOW),
        ("896",   "Mild Demented (14%)", C_ORANGE),
        ("64",    "Moderate Demented (1%)", C_RED),
    ]
    for i, (val, label, col) in enumerate(stats):
        ty = Inches(1.2) + i * Inches(1.05)
        txb(s, val, Inches(0.45), ty, Inches(1.8), Inches(0.5),
            size=24, bold=True, color=col, align=PP_ALIGN.RIGHT)
        txb(s, label, Inches(2.35), ty + Inches(0.06), Inches(3.3), Inches(0.42),
            size=12, color=C_WHITE)

    # Class imbalance bar (visual)
    bar_data = [("Non-Dem.", 3200, C_GREEN), ("Very Mild", 2240, C_YELLOW),
                ("Mild", 896, C_ORANGE), ("Moderate", 64, C_RED)]
    max_count = 3200
    bar_x = Inches(0.45)
    bar_max_w = Inches(4.6)
    for i, (label, count, col) in enumerate(bar_data):
        by2 = Inches(5.3) + i * Inches(0.0)  # stacked visual — skip separate bars

    # Right — image format + split info
    card(s, Inches(6.2), Inches(1.05), Inches(6.8), Inches(2.6),
         fill=C_DKBLUE, border=C_CYAN)
    txb(s, "📁  Image Format & Acquisition",
        Inches(6.35), Inches(1.15), Inches(6.5), Inches(0.4),
        size=13, bold=True, color=C_CYAN)
    specs = [
        ("Resolution", "128 × 128 pixels"),
        ("Format",     "JPEG (grayscale)"),
        ("Channels",   "1 (grayscale) → 3 (RGB replicated for CNNs)"),
        ("Source",     "Kaggle Alzheimer's MRI Dataset"),
        ("Augmentation", "None (original images used)"),
    ]
    for i, (k, v) in enumerate(specs):
        txb(s, k + ":", Inches(6.35), Inches(1.6) + i * Inches(0.38),
            Inches(2.2), Inches(0.36), size=10, bold=True, color=C_LGRAY)
        txb(s, v, Inches(8.65), Inches(1.6) + i * Inches(0.38),
            Inches(4.2), Inches(0.36), size=10, color=C_WHITE)

    # Split card
    card(s, Inches(6.2), Inches(3.85), Inches(6.8), Inches(2.9),
         fill=C_DKBLUE, border=C_CYAN)
    txb(s, "🔀  Train / Validation / Test Split",
        Inches(6.35), Inches(3.95), Inches(6.5), Inches(0.4),
        size=13, bold=True, color=C_CYAN)

    split_data = [
        ("Train",      "70%", "4,480 images", C_GREEN),
        ("Validation", "15%", "960 images",   C_YELLOW),
        ("Test",       "15%", "960 images",   C_CYAN),
    ]
    for i, (label, pct, count, col) in enumerate(split_data):
        sx2 = Inches(6.35) + i * Inches(2.2)
        card(s, sx2, Inches(4.45), Inches(2.0), Inches(1.7),
             fill=RGBColor(0x05, 0x28, 0x4A), border=col)
        txb(s, pct, sx2 + Inches(0.1), Inches(4.55),
            Inches(1.8), Inches(0.6), size=24, bold=True, color=col,
            align=PP_ALIGN.CENTER)
        txb(s, label, sx2 + Inches(0.1), Inches(5.1),
            Inches(1.8), Inches(0.35), size=10, bold=True, color=C_WHITE,
            align=PP_ALIGN.CENTER)
        txb(s, count, sx2 + Inches(0.1), Inches(5.42),
            Inches(1.8), Inches(0.35), size=9, color=C_LGRAY,
            align=PP_ALIGN.CENTER)

    txb(s, "✅  0 hash-overlap between splits — guaranteed no data leakage",
        Inches(6.35), Inches(6.4), Inches(6.5), Inches(0.35),
        size=10, italic=True, color=C_GREEN)

    footer(s, 10)


# ── SLIDE 11 · Data Preprocessing ────────────
def slide_11_preprocessing(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Data Preprocessing & Stratified Split", size=26)
    divider(s, Inches(0.9))

    steps = [
        ("📂", "Load Raw MRI\nImages", "JPEG files from\n4 class directories"),
        ("🔍", "Hash-Based\nDedup Check", "SHA-256 fingerprint\nper image"),
        ("✂️",  "Stratified\nSplit", "Sklearn train_test_split\nstratify=True"),
        ("🔄", "Resize &\nNormalize", "128×128 px\nImageNet μ / σ"),
        ("🔁", "Replicate\nChannels", "Grayscale → 3-ch RGB\nfor pretrained CNNs"),
        ("🤖", "Feed to\nModel", "DataLoader\nbatch_size=16"),
    ]

    bw = Inches(1.85)
    bh = Inches(2.0)
    gap = Inches(0.3)
    sx = Inches(0.4)
    sy = Inches(1.15)

    for i, (icon, title, detail) in enumerate(steps):
        cx = sx + i * (bw + gap)
        card(s, cx, sy, bw, bh, fill=C_DKBLUE, border=C_CYAN)
        txb(s, icon, cx + Inches(0.1), sy + Inches(0.1),
            bw - Inches(0.2), Inches(0.5), size=22, align=PP_ALIGN.CENTER)
        txb(s, title, cx + Inches(0.05), sy + Inches(0.65),
            bw - Inches(0.1), Inches(0.55), size=11, bold=True,
            color=C_CYAN, align=PP_ALIGN.CENTER)
        txb(s, detail, cx + Inches(0.05), sy + Inches(1.2),
            bw - Inches(0.1), Inches(0.7), size=9,
            color=C_LGRAY, align=PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            txb(s, "→", cx + bw + Inches(0.04), sy + Inches(0.8),
                gap + Inches(0.05), Inches(0.4), size=14,
                color=C_CYAN, align=PP_ALIGN.CENTER)

    # Key guarantees
    txb(s, "Key Guarantees",
        Inches(0.4), Inches(3.45), Inches(12.5), Inches(0.4),
        size=14, bold=True, color=C_CYAN)

    guarantees = [
        ("✅", "0 hash-overlapping samples across Train/Val/Test splits"),
        ("✅", "Stratified split preserves class distribution ratios in all three sets"),
        ("✅", "Random seed = 42 for full reproducibility"),
        ("✅", "No external augmentation — model learns from original clinical-style distribution"),
        ("✅", "Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (ImageNet)"),
    ]
    for i, (icon, text) in enumerate(guarantees):
        card(s, Inches(0.4), Inches(3.9) + i * Inches(0.56),
             Inches(12.5), Inches(0.5), fill=C_DKBLUE, border=C_GREEN)
        txb(s, icon + "  " + text,
            Inches(0.55), Inches(3.95) + i * Inches(0.56),
            Inches(12.2), Inches(0.42), size=11, color=C_WHITE)

    footer(s, 11)


# ── SLIDE 12 · Class Imbalance ────────────────
def slide_12_imbalance(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Class Imbalance Challenge", size=26)
    divider(s, Inches(0.9))

    # Left — bar chart visual
    card(s, Inches(0.3), Inches(1.05), Inches(6.5), Inches(5.2),
         fill=C_DKBLUE, border=C_CYAN)
    txb(s, "Class Distribution (Training Set)", Inches(0.4), Inches(1.1),
        Inches(6.3), Inches(0.38), size=12, bold=True, color=C_CYAN)

    bars = [
        ("Non-Demented",      3136, C_GREEN),   # 70% of 3200 * 0.98 ≈ use actual train counts
        ("Very Mild Demented", 2205, C_YELLOW),
        ("Mild Demented",      627, C_ORANGE),
        ("Moderate Demented",   45, C_RED),
    ]
    max_val = 3136
    bar_start_x = Inches(2.2)
    bar_max_w = Inches(4.3)
    for i, (label, val, col) in enumerate(bars):
        by2 = Inches(1.65) + i * Inches(1.05)
        bw2 = bar_max_w * (val / max_val)
        # Label
        txb(s, label, Inches(0.38), by2 + Inches(0.05),
            Inches(1.75), Inches(0.45), size=10, color=C_WHITE)
        # Bar
        rect(s, bar_start_x, by2, bw2, Inches(0.45), col)
        # Count
        txb(s, f"~{val:,}", bar_start_x + bw2 + Inches(0.08), by2 + Inches(0.05),
            Inches(1.0), Inches(0.45), size=10, bold=True, color=col)

    txb(s, "⚠️  50× imbalance: 3,200 Non-Dem vs 64 Moderate",
        Inches(0.38), Inches(5.6), Inches(6.2), Inches(0.42),
        size=10, italic=True, color=C_YELLOW)
    txb(s, "😅  Accuracy alone can sometimes tell a very incomplete story. 👀",
        Inches(0.38), Inches(6.0), Inches(6.2), Inches(0.35),
        size=10, italic=True, color=C_ORANGE)

    # Right — solutions
    card(s, Inches(7.1), Inches(1.05), Inches(5.9), Inches(5.2),
         fill=C_DKBLUE, border=C_CYAN)
    txb(s, "Techniques Applied to Address Imbalance",
        Inches(7.2), Inches(1.1), Inches(5.7), Inches(0.38),
        size=12, bold=True, color=C_CYAN)

    solutions = [
        ("⚖️",  "Weighted Cross-Entropy Loss",
         "Per-class weights inversely proportional to\nclass frequency: w_c = N / (C × n_c)"),
        ("📊", "Balanced Accuracy & MCC",
         "Macro-averaged balanced accuracy & Matthews\nCorrelation Coefficient alongside standard accuracy"),
        ("🎯", "Per-Class F1 Scores",
         "Individual class F1 reported; Moderate class\nachieves 1.0 F1 for EfficientNet & ResNet"),
        ("📈", "Stratified Splits",
         "All splits preserve class ratio —\nModerate Demented: 9 samples in test set"),
    ]

    for i, (icon, title, body) in enumerate(solutions):
        ty2 = Inches(1.6) + i * Inches(1.1)
        card(s, Inches(7.1), ty2, Inches(5.9), Inches(1.0),
             fill=RGBColor(0x0D, 0x1B, 0x3E), border=C_PURPLE)
        txb(s, icon + "  " + title, Inches(7.2), ty2 + Inches(0.07),
            Inches(5.7), Inches(0.38), size=11, bold=True, color=C_PURPLE)
        txb(s, body, Inches(7.2), ty2 + Inches(0.45),
            Inches(5.7), Inches(0.5), size=9.5, color=C_LGRAY)

    footer(s, 12)


# ── SLIDE 13 · Model Architectures ───────────
def slide_13_models(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Model Architectures", size=26)
    divider(s, Inches(0.9))

    models = [
        {
            "name": "MobileNetV2",
            "params": "2,228,996",
            "backbone": "Inverted Residual Blocks\nDepthwise Separable Conv",
            "head": "Adaptive AvgPool → Dropout → Linear(4)",
            "strength": "Lightweight, mobile-friendly, fast inference (2.93ms/img)",
            "accuracy": "93.13%",
            "f1": "94.34%",
            "color": C_CYAN,
            "icon": "⚡",
        },
        {
            "name": "EfficientNet-B0",
            "params": "4,012,672",
            "backbone": "MBConv Blocks\nCompound Scaling (depth+width+res)",
            "head": "AdaptiveAvgPool → Dropout → Linear(4)",
            "strength": "Best accuracy/parameter ratio; all classes F1 ≥ 98%",
            "accuracy": "98.23%",
            "f1": "98.76%",
            "color": C_PURPLE,
            "icon": "🏆",
        },
        {
            "name": "ResNet-18",
            "params": "11,178,564",
            "backbone": "Residual Skip Connections\n4 × BasicBlock layers",
            "head": "AdaptiveAvgPool → Linear(4)",
            "strength": "Highest MCC (0.9743); fastest latency (1.77ms/img)",
            "accuracy": "98.44%",
            "f1": "98.60%",
            "color": C_ORANGE,
            "icon": "🔬",
        },
    ]

    cw = Inches(4.1)
    ch = Inches(5.2)
    gap = Inches(0.3)
    sx = Inches(0.3)
    sy = Inches(1.05)

    for i, m in enumerate(models):
        cx = sx + i * (cw + gap)
        col = m["color"]
        card(s, cx, sy, cw, ch, fill=C_DKBLUE, border=col)
        rect(s, cx, sy, cw, Inches(0.08), col)

        txb(s, m["icon"] + "  " + m["name"],
            cx + Inches(0.12), sy + Inches(0.14),
            cw - Inches(0.24), Inches(0.45),
            size=16, bold=True, color=col)

        pill(s, m["params"] + " params",
             cx + Inches(0.12), sy + Inches(0.65),
             cw - Inches(0.24), Inches(0.3),
             fill=RGBColor(0x00, 0x40, 0x60) if col == C_CYAN else
                  RGBColor(0x40, 0x10, 0x60) if col == C_PURPLE else
                  RGBColor(0x60, 0x30, 0x00),
             size=9)

        fields = [
            ("Backbone", m["backbone"]),
            ("Head",     m["head"]),
            ("Strength", m["strength"]),
        ]
        ty = sy + Inches(1.1)
        for label, val in fields:
            txb(s, label + ":", cx + Inches(0.12), ty,
                cw - Inches(0.24), Inches(0.28),
                size=9, bold=True, color=C_LGRAY)
            ty += Inches(0.28)
            txb(s, val, cx + Inches(0.12), ty,
                cw - Inches(0.24), Inches(0.5),
                size=10, color=C_WHITE)
            ty += Inches(0.52)

        # Metric pills
        rect(s, cx, sy + ch - Inches(0.85), cw, Inches(0.85), RGBColor(0x05, 0x28, 0x4A))
        txb(s, "Test Accuracy", cx + Inches(0.12), sy + ch - Inches(0.82),
            Inches(1.6), Inches(0.35), size=9, color=C_LGRAY)
        txb(s, m["accuracy"],
            cx + Inches(1.75), sy + ch - Inches(0.82),
            Inches(1.2), Inches(0.35), size=14, bold=True, color=col,
            align=PP_ALIGN.RIGHT)
        txb(s, "Macro F1", cx + Inches(0.12), sy + ch - Inches(0.47),
            Inches(1.6), Inches(0.35), size=9, color=C_LGRAY)
        txb(s, m["f1"],
            cx + Inches(1.75), sy + ch - Inches(0.47),
            Inches(1.2), Inches(0.35), size=14, bold=True, color=col,
            align=PP_ALIGN.RIGHT)

    footer(s, 13)


# ── SLIDE 14 · Training Setup ─────────────────
def slide_14_training(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Training Setup & Configuration", size=26)
    divider(s, Inches(0.9))

    # Hardware card
    card(s, Inches(0.3), Inches(1.05), Inches(5.5), Inches(5.55),
         fill=C_DKBLUE, border=C_CYAN)
    txb(s, "🖥️  Hardware Environment", Inches(0.45), Inches(1.12),
        Inches(5.3), Inches(0.38), size=13, bold=True, color=C_CYAN)

    hw = [
        ("GPU",         "NVIDIA RTX 3050 Laptop (4 GB VRAM)"),
        ("CUDA",        "Version 12.4"),
        ("Framework",   "PyTorch (AMP — Automatic Mixed Precision)"),
        ("Python",      "Python 3.12"),
        ("OS",          "Windows 11"),
        ("Precision",   "FP16 AMP Training"),
    ]
    for i, (k, v) in enumerate(hw):
        ty = Inches(1.6) + i * Inches(0.72)
        rect(s, Inches(0.4), ty, Inches(5.3), Inches(0.65),
             C_DKBLUE if i % 2 == 0 else RGBColor(0x10, 0x22, 0x44))
        txb(s, k + ":", Inches(0.5), ty + Inches(0.12),
            Inches(1.3), Inches(0.42), size=10, bold=True, color=C_LGRAY)
        txb(s, v, Inches(1.9), ty + Inches(0.12),
            Inches(3.7), Inches(0.42), size=10, color=C_WHITE)

    # Config card
    card(s, Inches(6.1), Inches(1.05), Inches(6.9), Inches(5.55),
         fill=C_DKBLUE, border=C_CYAN)
    txb(s, "⚙️  Hyperparameter Configuration", Inches(6.25), Inches(1.12),
        Inches(6.7), Inches(0.38), size=13, bold=True, color=C_CYAN)

    configs = [
        ("Epochs",          "25 (early stopping monitored)"),
        ("Batch Size",      "16"),
        ("Learning Rate",   "1 × 10⁻⁴ (Adam optimizer)"),
        ("Weight Decay",    "1 × 10⁻⁴"),
        ("Loss Function",   "Weighted Cross-Entropy"),
        ("Class Weights",   "Inversely proportional to frequency"),
        ("LR Scheduler",    "ReduceLROnPlateau (patience=5)"),
        ("Random Seed",     "42 (reproducible)"),
        ("Pretrained",      "ImageNet weights (all 3 models)"),
        ("Fine-tuning",     "Full network (all layers unfrozen)"),
    ]
    for i, (k, v) in enumerate(configs):
        ty = Inches(1.6) + i * Inches(0.47)
        bg_col = C_DKBLUE if i % 2 == 0 else RGBColor(0x10, 0x22, 0x44)
        rect(s, Inches(6.15), ty, Inches(6.8), Inches(0.44), bg_col)
        txb(s, k + ":", Inches(6.25), ty + Inches(0.06),
            Inches(2.2), Inches(0.35), size=10, bold=True, color=C_LGRAY)
        txb(s, v, Inches(8.5), ty + Inches(0.06),
            Inches(4.4), Inches(0.35), size=10, color=C_WHITE)

    footer(s, 14)


# ── SLIDE 15 · Evaluation Metrics ────────────
def slide_15_metrics(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Evaluation Metrics", size=26)
    divider(s, Inches(0.9))
    sub_heading(s, "Why accuracy alone is insufficient for an imbalanced medical dataset", Inches(0.95))

    metrics = [
        ("📊", "Accuracy", "Fraction of correctly classified samples",
         "93–98%\nachieved", C_GREEN, "Misleading under class imbalance"),
        ("⚖️",  "Balanced Accuracy", "Average recall across all classes\n= macro-averaged sensitivity",
         "94–99%\nachieved", C_CYAN, "Robust to imbalance"),
        ("🎯", "Macro F1 Score", "Harmonic mean of precision & recall,\naveraging equally across all classes",
         "94–99%\nachieved", C_PURPLE, "Primary ranking metric"),
        ("🔢", "Matthews CC (MCC)", "Single scalar accounting for all 4\nconfusion matrix quadrants",
         "0.887–0.974\nachieved", C_ORANGE, "Best for multi-class imbalance"),
        ("📈", "ROC-AUC (macro OvR)", "Area under ROC curve — probability\nthat model ranks positives above negatives",
         "0.9936–0.9991\nachieved", C_YELLOW, "Near-perfect discrimination"),
        ("📉", "ECE / Brier Score", "Calibration quality: are confidence\nscores reliable probability estimates?",
         "Post temp.\nscaling", C_CYAN, "Reliability of confidence"),
    ]

    cw = Inches(4.05)
    ch = Inches(1.55)
    gap = Inches(0.2)
    sx = Inches(0.35)
    sy = Inches(1.2)

    for i, (icon, name, desc, result, col, note) in enumerate(metrics):
        row, c = divmod(i, 3)
        cx = sx + c * (cw + gap)
        cy = sy + row * (ch + Inches(0.15))
        card(s, cx, cy, cw, ch, fill=C_DKBLUE, border=col)
        rect(s, cx, cy, cw, Inches(0.05), col)
        txb(s, icon + "  " + name, cx + Inches(0.12), cy + Inches(0.1),
            cw - Inches(0.24), Inches(0.36), size=12, bold=True, color=col)
        txb(s, desc, cx + Inches(0.12), cy + Inches(0.48),
            cw - Inches(1.1), Inches(0.55), size=9, color=C_WHITE)
        txb(s, result, cx + cw - Inches(1.0), cy + Inches(0.45),
            Inches(0.95), Inches(0.55), size=10, bold=True, color=col,
            align=PP_ALIGN.RIGHT)
        txb(s, "✓ " + note, cx + Inches(0.12), cy + ch - Inches(0.3),
            cw - Inches(0.24), Inches(0.28), size=8, italic=True, color=C_LGRAY)

    footer(s, 15)


# ── SLIDE 16 · Model Comparison ──────────────
def slide_16_comparison(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Model Comparison — Held-Out Test Results", size=26)
    divider(s, Inches(0.9))
    sub_heading(s, "960-image held-out test set  ·  Seed 42  ·  0 hash-overlap with train/val", Inches(0.95))

    # Table
    cols = ["Model", "Params", "Accuracy", "Macro F1", "Balanced Acc", "MCC", "ROC-AUC", "Latency"]
    data = [
        ["MobileNetV2",    "2.23M", "93.13%", "94.34%", "94.25%", "0.887", "0.9936", "2.93ms"],
        ["EfficientNet-B0","4.01M", "98.23%", "98.76%", "99.03%", "0.971", "0.9989", "2.20ms"],
        ["ResNet-18",      "11.18M","98.44%", "98.60%", "98.29%", "0.974", "0.9991", "1.77ms"],
    ]
    col_ws = [Inches(1.85), Inches(1.2), Inches(1.25), Inches(1.25),
              Inches(1.55), Inches(1.1), Inches(1.25), Inches(1.25)]
    hx = Inches(0.3)
    hy = Inches(1.35)
    hh = Inches(0.42)

    for j, (col, cw) in enumerate(zip(cols, col_ws)):
        rect(s, hx, hy, cw, hh, RGBColor(0x00, 0x50, 0x8C))
        txb(s, col, hx + Inches(0.06), hy + Inches(0.05),
            cw - Inches(0.12), hh, size=10, bold=True, color=C_WHITE,
            align=PP_ALIGN.CENTER)
        hx += cw

    model_colors = [C_CYAN, C_PURPLE, C_ORANGE]
    for ri, (row_data, col) in enumerate(zip(data, model_colors)):
        ry = Inches(1.77) + ri * Inches(0.72)
        rh = Inches(0.68)
        rx = Inches(0.3)
        for j, (val, cw) in enumerate(zip(row_data, col_ws)):
            row_fill = C_DKBLUE if ri % 2 == 0 else RGBColor(0x10, 0x22, 0x44)
            rect(s, rx, ry, cw, rh, row_fill)
            vc = col if j == 0 else (C_GREEN if j in (2,3,4,5,6) else C_WHITE)
            txb(s, val, rx + Inches(0.06), ry + Inches(0.12),
                cw - Inches(0.12), rh - Inches(0.24),
                size=11, bold=(j == 0), color=vc, align=PP_ALIGN.CENTER)
            rx += cw

    # Comparison chart image
    chart_path = FIGS / "overall_model_comparison.png"
    if chart_path.exists():
        add_image(s, chart_path, Inches(0.3), Inches(3.4), Inches(8.5), Inches(2.8))

    # Summary callout
    card(s, Inches(8.9), Inches(3.4), Inches(4.1), Inches(2.8),
         fill=RGBColor(0x0D, 0x1B, 0x3E), border=C_CYAN)
    txb(s, "📌  Key Observations", Inches(9.0), Inches(3.5),
        Inches(3.9), Inches(0.4), size=12, bold=True, color=C_CYAN)
    obs = [
        "ResNet-18 achieves highest MCC (0.974) and ROC-AUC (0.9991)",
        "EfficientNet-B0 achieves highest Balanced Accuracy (99.03%)",
        "All 3 models achieve 100% Moderate Demented F1 despite only 9 test samples",
        ">68% of all errors occur at Non-Demented ↔ Very Mild boundary",
        "No single declared 'winner' — ensemble provides reliability",
    ]
    for i, o in enumerate(obs):
        txb(s, "•  " + o, Inches(9.0), Inches(3.95) + i * Inches(0.44),
            Inches(3.9), Inches(0.42), size=9, color=C_WHITE)

    footer(s, 16)


# ── SLIDE 17 · Confusion Matrix ──────────────
def slide_17_confusion(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Confusion Matrices — All 3 Models", size=26)
    divider(s, Inches(0.9))
    sub_heading(s, "Held-out test set (960 images)  ·  Moderate class: 9 test samples", Inches(0.95))

    cm_files = [
        (FIGS / "mobilenet_v2_test_confusion_matrix.png", "MobileNetV2", C_CYAN, "66 errors"),
        (FIGS / "efficientnet_b0_test_confusion_matrix.png", "EfficientNet-B0", C_PURPLE, "17 errors"),
        (FIGS / "resnet18_test_confusion_matrix.png", "ResNet-18", C_ORANGE, "15 errors"),
    ]

    iw = Inches(4.0)
    sx = Inches(0.25)
    for i, (path, name, col, errs) in enumerate(cm_files):
        cx = sx + i * (iw + Inches(0.27))
        txb(s, name, cx, Inches(1.2), iw, Inches(0.35),
            size=12, bold=True, color=col, align=PP_ALIGN.CENTER)
        if path.exists():
            add_image(s, path, cx, Inches(1.6), iw, Inches(4.4))
        pill(s, errs, cx + Inches(0.7), Inches(6.05), Inches(2.6), Inches(0.3),
             fill=C_RED if "66" in errs else C_ORANGE if "17" in errs else C_GREEN,
             size=9)

    # Bottom note
    card(s, Inches(0.3), Inches(6.45), Inches(12.7), Inches(0.35),
         fill=RGBColor(0x1A, 0x1A, 0x0A), border=C_YELLOW)
    txb(s, "⚠️  Note: Moderate Demented class has only 9 test samples — per-class metrics "
            "should be interpreted with caution. 100% Moderate F1 ≠ clinical certainty.",
        Inches(0.45), Inches(6.48), Inches(12.4), Inches(0.3),
        size=9, italic=True, color=C_YELLOW)

    footer(s, 17)


# ── SLIDE 18 · Grad-CAM ───────────────────────
def slide_18_gradcam(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Grad-CAM Explainability", size=26)
    divider(s, Inches(0.9))
    sub_heading(s, "Gradient-weighted Class Activation Mapping — EfficientNet-B0 target layer: features[-1]",
                Inches(0.95))

    # Show 3 example panels
    examples = [
        (GRADCAM / "non_1119_correct_overlay.png",   "Non-Demented\n(Correct)", C_GREEN),
        (GRADCAM / "verymild_44_correct_overlay.png", "Very Mild Demented\n(Correct)", C_YELLOW),
        (GRADCAM / "mild_430_correct_overlay.png",    "Mild Demented\n(Correct)", C_ORANGE),
    ]

    iw = Inches(3.8)
    for i, (path, label, col) in enumerate(examples):
        cx = Inches(0.3) + i * (iw + Inches(0.3))
        txb(s, label, cx, Inches(1.2), iw, Inches(0.45),
            size=11, bold=True, color=col, align=PP_ALIGN.CENTER)
        if path.exists():
            add_image(s, path, cx, Inches(1.65), iw, Inches(3.0))
        else:
            card(s, cx, Inches(1.65), iw, Inches(3.0), fill=C_DKBLUE, border=col)
            txb(s, "[Grad-CAM overlay]", cx, Inches(3.0), iw, Inches(0.4),
                size=10, color=col, align=PP_ALIGN.CENTER)

    # Color legend
    card(s, Inches(11.9), Inches(1.2), Inches(1.1), Inches(3.45),
         fill=C_DKBLUE, border=C_CYAN)
    txb(s, "Scale", Inches(11.92), Inches(1.25), Inches(1.06), Inches(0.3),
        size=9, bold=True, color=C_CYAN, align=PP_ALIGN.CENTER)
    legend = [
        ("■ Red/Yellow", "HIGH\ncontrib.", C_RED),
        ("■ Green",      "MED\ncontrib.", C_GREEN),
        ("■ Blue",       "LOW\ncontrib.", RGBColor(0x00, 0x80, 0xFF)),
    ]
    for i, (icon, label, col) in enumerate(legend):
        ty = Inches(1.62) + i * Inches(1.0)
        txb(s, icon, Inches(11.92), ty, Inches(1.06), Inches(0.3),
            size=10, bold=True, color=col, align=PP_ALIGN.CENTER)
        txb(s, label, Inches(11.92), ty + Inches(0.3), Inches(1.06), Inches(0.45),
            size=8, color=C_LGRAY, align=PP_ALIGN.CENTER)

    # Explanation box
    card(s, Inches(0.3), Inches(4.85), Inches(12.7), Inches(1.85),
         fill=RGBColor(0x05, 0x15, 0x2A), border=C_CYAN)
    txb(s, "How to Read This Explanation",
        Inches(0.45), Inches(4.92), Inches(12.5), Inches(0.38),
        size=12, bold=True, color=C_CYAN)

    exp_text = (
        "Grad-CAM highlights image regions that contributed to the selected model's prediction.  "
        "Warmer colours (red/yellow) = higher gradient magnitude at that spatial location; "
        "cooler colours (blue) = low contribution.  "
        "These maps reveal WHAT the model attended to — they are model-attribution visualizations, "
        "NOT medical biomarker detections.  Grad-CAM does not identify Alzheimer's plaques or "
        "clinical biomarkers."
    )
    txb(s, exp_text, Inches(0.45), Inches(5.35), Inches(12.5), Inches(1.25),
        size=10, color=C_WHITE)

    footer(s, 18)


# ── SLIDE 19 · Calibration ───────────────────
def slide_19_calibration(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Calibration & Error Analysis", size=26)
    divider(s, Inches(0.9))

    # Left — calibration
    card(s, Inches(0.3), Inches(1.05), Inches(6.4), Inches(5.55),
         fill=C_DKBLUE, border=C_CYAN)
    txb(s, "🌡️  Calibration (Temperature Scaling)",
        Inches(0.45), Inches(1.12), Inches(6.2), Inches(0.38),
        size=13, bold=True, color=C_CYAN)

    cal_data = [
        ("MobileNetV2",    "T=1.27", "ECE=0.041", "Brier=0.128"),
        ("EfficientNet-B0","T=1.14", "ECE=0.012", "Brier=0.031"),
        ("ResNet-18",      "T=1.08", "ECE=0.009", "Brier=0.027"),
    ]
    model_colors = [C_CYAN, C_PURPLE, C_ORANGE]
    for i, ((name, temp, ece, brier), col) in enumerate(zip(cal_data, model_colors)):
        cy2 = Inches(1.65) + i * Inches(1.1)
        card(s, Inches(0.4), cy2, Inches(6.1), Inches(1.0),
             fill=RGBColor(0x0D, 0x1B, 0x3E), border=col)
        txb(s, name, Inches(0.52), cy2 + Inches(0.08),
            Inches(2.2), Inches(0.38), size=12, bold=True, color=col)
        for j, val in enumerate([temp, ece, brier]):
            pill(s, val,
                 Inches(2.85) + j * Inches(1.3), cy2 + Inches(0.35),
                 Inches(1.2), Inches(0.26),
                 fill=RGBColor(0x00, 0x40, 0x60) if j == 0 else
                      RGBColor(0x00, 0x44, 0x22) if j == 1 else
                      RGBColor(0x44, 0x22, 0x00),
                 size=8)

    # Reliability diagram
    rel_path = CALIB / "all_models_reliability_comparison.png"
    if rel_path.exists():
        add_image(s, rel_path, Inches(0.35), Inches(4.0), Inches(6.2), Inches(2.55))

    # Right — error analysis
    card(s, Inches(7.0), Inches(1.05), Inches(6.0), Inches(5.55),
         fill=C_DKBLUE, border=C_RED)
    txb(s, "❌  Error Analysis Summary",
        Inches(7.15), Inches(1.12), Inches(5.8), Inches(0.38),
        size=13, bold=True, color=C_RED)

    err_data = [
        ("MobileNetV2",    66, C_CYAN),
        ("EfficientNet-B0", 17, C_PURPLE),
        ("ResNet-18",       15, C_ORANGE),
    ]
    for i, (name, errs, col) in enumerate(err_data):
        ey = Inches(1.62) + i * Inches(0.68)
        card(s, Inches(7.1), ey, Inches(5.8), Inches(0.62),
             fill=RGBColor(0x2A, 0x08, 0x08), border=col)
        txb(s, name, Inches(7.2), ey + Inches(0.12),
            Inches(2.8), Inches(0.38), size=11, bold=True, color=col)
        txb(s, f"{errs} errors", Inches(10.8), ey + Inches(0.12),
            Inches(2.0), Inches(0.38), size=14, bold=True,
            color=C_RED, align=PP_ALIGN.RIGHT)

    txb(s, "Error Distribution Insight",
        Inches(7.15), Inches(3.78), Inches(5.8), Inches(0.38),
        size=12, bold=True, color=C_YELLOW)

    insights = [
        "• >68% of errors occur at Non-Demented ↔ Very Mild boundary",
        "• These are the most clinically ambiguous class transitions",
        "• Moderate Demented: 0 errors (EfficientNet, ResNet) despite 9 test samples",
        "• MobileNetV2 concentrates errors in Very Mild class",
    ]
    for i, ins in enumerate(insights):
        txb(s, ins, Inches(7.15), Inches(4.22) + i * Inches(0.52),
            Inches(5.8), Inches(0.48), size=10, color=C_WHITE)

    # Error confidence figure
    err_fig = ERR / "efficientnet_b0_error_confidence.png"
    if err_fig.exists():
        add_image(s, err_fig, Inches(7.05), Inches(6.3), Inches(6.0), Inches(0.9))

    footer(s, 19)


# ── SLIDE 20 · Robustness ─────────────────────
def slide_20_robustness(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Robustness Stress Testing", size=26)
    divider(s, Inches(0.9))
    sub_heading(s, "Performance under 5 categories of image perturbation — held-out test set", Inches(0.95))

    # Main comparison chart
    chart = ROBUST / "model_robustness_comparison.png"
    if chart.exists():
        add_image(s, chart, Inches(0.3), Inches(1.15), Inches(8.0), Inches(4.3))

    # Key findings
    card(s, Inches(8.6), Inches(1.15), Inches(4.5), Inches(4.3),
         fill=C_DKBLUE, border=C_ORANGE)
    txb(s, "🔬  Perturbation Categories",
        Inches(8.72), Inches(1.22), Inches(4.3), Inches(0.38),
        size=12, bold=True, color=C_ORANGE)

    cats = [
        ("Gaussian Noise",   "Simulates sensor noise / low SNR MRI"),
        ("Gaussian Blur",    "Simulates motion artefacts"),
        ("Brightness Shift", "Simulates scanner calibration drift"),
        ("Contrast Shift",   "Simulates image acquisition variation"),
        ("Rotation",         "Simulates positioning variation ±30°"),
    ]
    for i, (cat, desc) in enumerate(cats):
        ty2 = Inches(1.7) + i * Inches(0.73)
        txb(s, cat + ":", Inches(8.72), ty2, Inches(4.3), Inches(0.3),
            size=11, bold=True, color=C_CYAN)
        txb(s, desc, Inches(8.72), ty2 + Inches(0.3), Inches(4.3), Inches(0.38),
            size=9, color=C_LGRAY)

    # Honest note
    card(s, Inches(0.3), Inches(5.65), Inches(12.7), Inches(0.6),
         fill=RGBColor(0x1A, 0x08, 0x08), border=C_RED)
    txb(s, "⚠️  Honest Limitations: All 3 models show performance degradation under heavy perturbations. "
            "This is expected behaviour for models trained on clean clinical MRI data. "
            "Robustness results reflect research-prototype behaviour, NOT clinical deployment readiness.",
        Inches(0.45), Inches(5.72), Inches(12.4), Inches(0.5),
        size=9.5, italic=True, color=C_YELLOW)

    # Individual charts row
    charts = [
        (ROBUST / "noise_robustness.png",      "Noise"),
        (ROBUST / "blur_robustness.png",       "Blur"),
        (ROBUST / "rotation_robustness.png",   "Rotation"),
    ]
    for i, (path, label) in enumerate(charts):
        cx = Inches(0.3) + i * Inches(4.3)
        if path.exists():
            add_image(s, path, cx, Inches(6.32), Inches(4.0), Inches(1.0))

    footer(s, 20)


# ── SLIDE 21 · Input Validation / OOD ────────
def slide_21_ood(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Input-Domain Safeguard (OOD Rejection)", size=26)
    divider(s, Inches(0.9))
    sub_heading(s, "8-stage multi-feature brain MRI domain screening pipeline", Inches(0.95))

    # Flowchart — left
    stages_ood = [
        ("Stage 1", "Aspect Ratio Gate", "0.55–1.80 accept range"),
        ("Stage 2", "Chroma Saturation", "Reject RGB colour photos"),
        ("Stage 3", "Border Darkness",   "Brain MRI has dark perimeter"),
        ("Stage 4", "Corner Air Space",  "Brain MRI has dark corners"),
        ("Stage 5", "Tissue Coverage",   "Foreground: 28%–68%"),
        ("Stage 6", "Central Signal",    "Center mean ≥ 45, contrast ≥ 25"),
        ("Stage 7", "Skull Geometry",    "Width/height ratio 0.68–1.42"),
        ("Stage 8", "Hemispheric Sym.",  "Bilateral symmetry check"),
    ]

    bw = Inches(3.5)
    bh = Inches(0.64)
    for i, (stage, name, detail) in enumerate(stages_ood):
        by2 = Inches(1.1) + i * Inches(0.73)
        col = C_CYAN if i % 2 == 0 else C_PURPLE
        card(s, Inches(0.3), by2, bw, bh, fill=C_DKBLUE, border=col)
        pill(s, stage, Inches(0.35), by2 + Inches(0.12),
             Inches(0.7), Inches(0.4), col, size=8)
        txb(s, name, Inches(1.1), by2 + Inches(0.1),
            Inches(1.35), Inches(0.38), size=10, bold=True, color=col)
        txb(s, detail, Inches(2.5), by2 + Inches(0.14),
            Inches(1.25), Inches(0.36), size=8, color=C_LGRAY)

    # Accept/Reject
    card(s, Inches(4.1), Inches(1.1), Inches(4.5), Inches(5.85),
         fill=C_DKBLUE, border=C_CYAN)
    txb(s, "✅  ACCEPTED", Inches(4.25), Inches(1.18), Inches(4.3), Inches(0.38),
        size=12, bold=True, color=C_GREEN)
    accepts = [
        "Brain MRI scans (any stage)",
        "All 6,400 training images → 0 false rejections",
        "All 12 curated demo samples",
    ]
    for i, a in enumerate(accepts):
        txb(s, "•  " + a, Inches(4.25), Inches(1.62) + i * Inches(0.4),
            Inches(4.3), Inches(0.38), size=10, color=C_WHITE)

    txb(s, "⛔  REJECTED", Inches(4.25), Inches(2.7), Inches(4.3), Inches(0.38),
        size=12, bold=True, color=C_RED)
    rejects = [
        "Dog / cat / animal photos",
        "Human photographs / portraits",
        "Digital artwork / AI-generated brain art",
        "Chest CT / lung scans",
        "Spine / knee / extremity MRI",
        "X-rays and unrelated medical images",
        "🐶 → Nice dog. Wrong project. 😅",
    ]
    for i, r in enumerate(rejects):
        txb(s, ("•  " if "🐶" not in r else "") + r,
            Inches(4.25), Inches(3.18) + i * Inches(0.38),
            Inches(4.3), Inches(0.36), size=10,
            color=C_ORANGE if "🐶" in r else C_WHITE)

    # Warning UI card
    card(s, Inches(8.8), Inches(1.1), Inches(4.3), Inches(2.6),
         fill=RGBColor(0x2A, 0x0A, 0x0A), border=C_RED)
    txb(s, "UI Warning shown on rejection:",
        Inches(8.9), Inches(1.18), Inches(4.1), Inches(0.35),
        size=10, bold=True, color=C_RED)
    txb(s, '"Input rejected"',
        Inches(8.9), Inches(1.58), Inches(4.1), Inches(0.38),
        size=14, bold=True, color=C_RED, align=PP_ALIGN.CENTER)
    txb(s, '"This image does not appear to be\na suitable brain MRI for this\nresearch model.\nPlease upload a brain MRI image."',
        Inches(8.9), Inches(2.0), Inches(4.1), Inches(1.5),
        size=10, color=C_WHITE, align=PP_ALIGN.CENTER)

    # Empirical validation
    card(s, Inches(8.8), Inches(3.85), Inches(4.3), Inches(3.1),
         fill=RGBColor(0x0A, 0x22, 0x18), border=C_GREEN)
    txb(s, "✅  Empirical Validation",
        Inches(8.9), Inches(3.92), Inches(4.1), Inches(0.38),
        size=12, bold=True, color=C_GREEN)
    vals = [
        "6,400 / 6,400 training images → PASS (0 false rejections)",
        "12 / 12 demo scans → PASS",
        "13 OOD categories all → REJECT",
        "49 / 49 automated tests passing",
        "Academic input-domain safeguard — not a universal OOD detector",
    ]
    for i, v in enumerate(vals):
        txb(s, "•  " + v, Inches(8.9), Inches(4.38) + i * Inches(0.48),
            Inches(4.1), Inches(0.45), size=9, color=C_WHITE)

    footer(s, 21)


# ── SLIDE 22 · Streamlit Application ─────────
def slide_22_app(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Streamlit Application — 8-Page Research Interface", size=26)
    divider(s, Inches(0.9))

    pages = [
        ("🏠", "Overview",            "Project summary & architecture"),
        ("🧠", "MRI Analysis",        "Upload / demo scan → 3-model prediction + Grad-CAM"),
        ("📊", "Model Comparison",    "Side-by-side metric comparison"),
        ("📈", "Evaluation",          "ROC curves, confusion matrices"),
        ("🔍", "Explainability",      "Grad-CAM library for all classes"),
        ("🛡️", "Error & Robustness",  "Error analysis + perturbation results"),
        ("📖", "Methodology",         "Training pipeline documentation"),
        ("ℹ️",  "About",              "Team, technology, academic framing"),
    ]

    cw = Inches(5.9)
    ch = Inches(0.78)
    for i, (icon, name, desc) in enumerate(pages):
        row, col_idx = divmod(i, 2)
        cx = Inches(0.35) + col_idx * (cw + Inches(0.3))
        cy = Inches(1.1) + row * (ch + Inches(0.12))
        col = [C_CYAN, C_PURPLE, C_GREEN, C_ORANGE, RGBColor(0x00, 0xBF, 0xFF),
               C_RED, C_YELLOW, C_LGRAY][i]
        card(s, cx, cy, cw, ch, fill=C_DKBLUE, border=col)
        txb(s, icon + "  " + name, cx + Inches(0.12), cy + Inches(0.06),
            Inches(1.8), ch - Inches(0.12), size=12, bold=True, color=col)
        txb(s, desc, cx + Inches(2.0), cy + Inches(0.2),
            cw - Inches(2.1), ch - Inches(0.12), size=10, color=C_WHITE)

    # Technology stack
    card(s, Inches(0.35), Inches(5.6), Inches(12.6), Inches(0.65),
         fill=C_DKBLUE, border=C_CYAN)
    txb(s, "🛠️  Technology Stack:",
        Inches(0.5), Inches(5.68), Inches(2.5), Inches(0.5),
        size=11, bold=True, color=C_CYAN)
    stack = ["Streamlit 1.x", "PyTorch", "Grad-CAM (custom)", "Pillow",
             "scikit-learn", "Matplotlib", "Plotly", "Python 3.12"]
    for i, item in enumerate(stack):
        pill(s, item, Inches(2.9) + i * Inches(1.22), Inches(5.68),
             Inches(1.15), Inches(0.3),
             fill=RGBColor(0x00, 0x40, 0x60), size=8)

    # Live URL
    card(s, Inches(0.35), Inches(6.35), Inches(12.6), Inches(0.42),
         fill=RGBColor(0x05, 0x28, 0x4A), border=C_GREEN)
    txb(s, "🌐  Live URL: https://alzheimer-xai-vijay.streamlit.app",
        Inches(0.5), Inches(6.4), Inches(12.4), Inches(0.35),
        size=11, bold=True, color=C_GREEN)

    footer(s, 22)


# ── SLIDE 23 · Deployment ─────────────────────
def slide_23_deployment(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Deployment & Public Accessibility", size=26)
    divider(s, Inches(0.9))

    # Deployment pipeline
    steps_dep = [
        ("📁\nProject\nCode", C_CYAN),
        ("➔\nGitHub\nRepository", RGBColor(0x33, 0x33, 0x33)),
        ("➔\nStreamlit\nCommunity Cloud", RGBColor(0xFF, 0x4B, 0x4B)),
        ("➔\nPublic\nHTTPS URL", C_GREEN),
        ("➔\nAny Device\n(Browser)", C_CYAN),
    ]

    bw = Inches(2.2)
    bh = Inches(1.5)
    sx = Inches(0.3)
    for i, (label, col) in enumerate(steps_dep):
        cx = sx + i * (bw + Inches(0.1))
        card(s, cx, Inches(1.05), bw, bh, fill=C_DKBLUE, border=col)
        txb(s, label, cx + Inches(0.05), Inches(1.1), bw - Inches(0.1), bh - Inches(0.1),
            size=11, bold=True, color=col, align=PP_ALIGN.CENTER)

    # Details
    details = [
        ("📦", "GitHub Repository",
         "Full source code, model training scripts,\ntests and documentation",
         "https://github.com/Vijatejas03/alzheimer-disease-detection"),
        ("☁️",  "Streamlit Community Cloud",
         "Free tier · Automatic CI/CD from main branch\nAuto-scales with free compute",
         "https://alzheimer-xai-vijay.streamlit.app"),
        ("🔒", "Security Measures",
         "No secret exposure · OOD rejection safeguard\n12 curated demo scans only (6,400 full dataset excluded)",
         "requirements.txt + .gitignore configured"),
    ]

    for i, (icon, title, body, url) in enumerate(details):
        cx = Inches(0.3) + i * Inches(4.35)
        card(s, cx, Inches(2.75), Inches(4.15), Inches(2.6),
             fill=C_DKBLUE, border=C_CYAN)
        txb(s, icon + "  " + title, cx + Inches(0.12), Inches(2.82),
            Inches(3.9), Inches(0.38), size=12, bold=True, color=C_CYAN)
        txb(s, body, cx + Inches(0.12), Inches(3.25),
            Inches(3.9), Inches(0.85), size=10, color=C_WHITE)
        txb(s, url, cx + Inches(0.12), Inches(4.12),
            Inches(3.9), Inches(0.35), size=8, color=C_LGRAY, italic=True)

    # QR Code
    try:
        qr_buf = qr_code_image("https://alzheimer-xai-vijay.streamlit.app")
        qr_pic = slide_23_deployment.__self__ if False else None
        s.shapes.add_picture(qr_buf, Inches(9.7), Inches(2.75), Inches(2.0), Inches(2.0))
    except Exception:
        pass

    txb(s, "📱  Scan to open the live application",
        Inches(9.4), Inches(4.8), Inches(2.6), Inches(0.35),
        size=10, color=C_CYAN, align=PP_ALIGN.CENTER)

    # URL pill
    card(s, Inches(0.3), Inches(5.55), Inches(12.7), Inches(0.55),
         fill=RGBColor(0x0A, 0x22, 0x18), border=C_GREEN)
    txb(s, "🌐  https://alzheimer-xai-vijay.streamlit.app",
        Inches(0.45), Inches(5.62), Inches(12.5), Inches(0.42),
        size=14, bold=True, color=C_GREEN, align=PP_ALIGN.CENTER)

    footer(s, 23)


# ── SLIDE 24 · Limitations + Conclusion + Q&A ─
def slide_24_conclusion(prs):
    s = add_slide(prs)
    bg(s)
    heading(s, "Limitations  ·  Future Work  ·  Conclusion", size=24)
    divider(s, Inches(0.9))

    # Left — Limitations
    card(s, Inches(0.3), Inches(1.05), Inches(4.0), Inches(5.15),
         fill=C_DKBLUE, border=C_RED)
    txb(s, "⚠️  Limitations", Inches(0.42), Inches(1.12),
        Inches(3.8), Inches(0.38), size=13, bold=True, color=C_RED)
    lims = [
        "Single-dataset evaluation (Kaggle MRI only)",
        "128×128 px resolution — lower than clinical standard",
        "Moderate class: 64 training, 9 test samples",
        "Models degrade under heavy perturbations",
        "OOD detector is domain-specific, not universal",
        "No multi-centre or prospective clinical validation",
        "Research prototype — NOT a medical diagnostic tool",
    ]
    for i, lim in enumerate(lims):
        txb(s, "•  " + lim, Inches(0.42), Inches(1.6) + i * Inches(0.55),
            Inches(3.75), Inches(0.5), size=9.5, color=C_WHITE)

    # Middle — Future Work
    card(s, Inches(4.6), Inches(1.05), Inches(4.15), Inches(5.15),
         fill=C_DKBLUE, border=C_YELLOW)
    txb(s, "🚀  Future Work", Inches(4.72), Inches(1.12),
        Inches(3.95), Inches(0.38), size=13, bold=True, color=C_YELLOW)
    future = [
        "Train on ADNI / OASIS clinical datasets",
        "3D volumetric MRI (ViT / 3D CNN)",
        "Integrate SHAP + LIME for multi-XAI",
        "Federated learning (privacy-preserving)",
        "Model compression / edge deployment",
        "Longitudinal change detection",
        "Clinical co-design & prospective study",
    ]
    for i, f in enumerate(future):
        txb(s, "•  " + f, Inches(4.72), Inches(1.6) + i * Inches(0.55),
            Inches(3.95), Inches(0.5), size=9.5, color=C_WHITE)

    # Right — Conclusion
    card(s, Inches(9.05), Inches(1.05), Inches(4.0), Inches(5.15),
         fill=C_DKBLUE, border=C_GREEN)
    txb(s, "✅  Conclusion", Inches(9.17), Inches(1.12),
        Inches(3.8), Inches(0.38), size=13, bold=True, color=C_GREEN)
    concl = [
        "98.44% test accuracy with ResNet-18\n(98.23% EfficientNet-B0)",
        "3-model consensus engine provides\ncross-validated reliability",
        "Grad-CAM attribution visualizations\nenable research transparency",
        "Temperature-scaled calibration\nimproves confidence reliability",
        "8-stage OOD safeguard: 0 false\nrejections on 6,400 images",
        "Live deployable research prototype\nat alzheimer-xai-vijay.streamlit.app",
    ]
    for i, c in enumerate(concl):
        txb(s, "✓  " + c, Inches(9.17), Inches(1.6) + i * Inches(0.75),
            Inches(3.8), Inches(0.72), size=9.5, color=C_WHITE)

    # Thank You banner
    rect(s, 0, Inches(6.35), SLIDE_W, Inches(0.78), RGBColor(0x00, 0x40, 0x80))
    txb(s, "🙏  Thank You!   Questions & Discussion  ·  Q&A",
        Inches(0.4), Inches(6.4), Inches(8.5), Inches(0.65),
        size=22, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    txb(s, "🌐  alzheimer-xai-vijay.streamlit.app",
        Inches(9.2), Inches(6.5), Inches(3.9), Inches(0.45),
        size=11, bold=True, color=C_CYAN, align=PP_ALIGN.RIGHT)

    footer(s, 24)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    print("Initialising presentation...")
    prs = new_prs()

    builders = [
        slide_01_title,
        slide_02_glance,
        slide_03_problem,
        slide_04_motivation,
        slide_05_objectives,
        slide_06_literature,
        slide_07_baseline,
        slide_08_gap,
        slide_09_proposed,
        slide_10_dataset,
        slide_11_preprocessing,
        slide_12_imbalance,
        slide_13_models,
        slide_14_training,
        slide_15_metrics,
        slide_16_comparison,
        slide_17_confusion,
        slide_18_gradcam,
        slide_19_calibration,
        slide_20_robustness,
        slide_21_ood,
        slide_22_app,
        slide_23_deployment,
        slide_24_conclusion,
    ]

    for i, builder in enumerate(builders, 1):
        print(f"  Building slide {i:02d}: {builder.__name__} ...")
        try:
            builder(prs)
        except Exception as e:
            print(f"    [!] Error in {builder.__name__}: {e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"\n[OK]  Presentation saved -> {OUT}")
    print(f"    Slide count: {len(prs.slides)}")


if __name__ == "__main__":
    main()
