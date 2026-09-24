"""
Application Header Component for Medical AI Research Theme.
Provides high-contrast 3D-styled header and contextual breadcrumbs.
"""

import streamlit as st
from app.utils.inference_engine import get_inference_device
from app.utils.ui_helpers import render_html
from app.utils.navigation import navigate_to


PAGE_SUBTITLES = {
    "Overview": "Interactive Research Platform & System Dashboard",
    "MRI Analysis": "Deep Learning Inference, Input Quality Validation & Model Attribution",
    "Model Comparison": "Comparative Architecture Benchmarks across 960 Held-Out Scans",
    "Evaluation": "Comprehensive Evaluation Metrics, Confusion Matrices & ROC Curves",
    "Explainability": "Grad-CAM Saliency Maps & Visual Feature Attribution Gallery",
    "Error & Robustness": "Failure Mode Profiling & Controlled Acquisition Perturbations",
    "Methodology": "11-Stage Scientific Experimental Protocol & Cryptographic Data Audit",
    "About": "System Architecture, Deep Learning Stack & Ethical Academic Governance"
}


def render_app_header(current_page: str = "Overview"):
    """
    Render authoritative medical-AI top banner.
    Adapts intelligently between Overview and inner module views.
    """
    device, dev_desc = get_inference_device()
    dev_str = "CUDA GPU (RTX 3050)" if device.type == "cuda" else "CPU Compute Mode"
    
    if current_page == "Overview":
        # Overview Top Utility Strip
        strip_html = (
            '<div class="overview-top-strip">'
            '<div class="top-strip-left">'
            '<span class="pulse-indicator"><span class="pulse-dot"></span> Live Prototype</span>'
            '<span class="top-strip-item"><strong>Platform:</strong> PyTorch 2.6.0</span>'
            f'<span class="top-strip-item"><strong>Hardware:</strong> {dev_str}</span>'
            '<span class="top-strip-item"><strong>Data Quarantine:</strong> SHA-256 Verified</span>'
            '</div>'
            '<div class="top-strip-right">'
            '<span class="badge-chip badge-chip-info">Academic Research Prototype</span>'
            '</div>'
            '</div>'
        )
        render_html(strip_html)
    else:
        # Inner Module 3D Header Banner
        sub = PAGE_SUBTITLES.get(current_page, "Academic Deep Learning Research Platform")
        banner_html = (
            '<div class="med-header-banner">'
            '<div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1.0rem;">'
            '<div style="max-width: 840px;">'
            '<div class="header-breadcrumb">'
            '<span>🏠 Overview</span>'
            '<span style="opacity: 0.6;">/</span>'
            f'<span style="color: #67E8F9; font-weight: 700;">{current_page}</span>'
            '</div>'
            f'<div class="med-header-title">{current_page}</div>'
            f'<div class="med-header-subtitle">{sub}</div>'
            '</div>'
            '<div style="text-align: right; white-space: nowrap;">'
            '<span class="med-header-badge">ACADEMIC RESEARCH PROTOTYPE</span>'
            f'<div style="color: #BAE6FD; font-size: 0.74rem; margin-top: 0.35rem; font-weight: 600;">{dev_str}</div>'
            '</div>'
            '</div>'
            '</div>'
        )
        render_html(banner_html)
