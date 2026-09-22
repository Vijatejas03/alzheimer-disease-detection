"""
Application Header Component for Medical AI Research Theme.
"""

from app.utils.inference_engine import get_inference_device
from app.utils.ui_helpers import render_html


def render_app_header(current_page: str = "Home"):
    """Render authoritative medical-AI top banner with guaranteed safe HTML."""
    device, dev_desc = get_inference_device()
    dev_str = "CUDA Enabled (RTX 3050)" if device.type == "cuda" else "CPU Inference Mode"
    
    header_html = (
        '<div class="med-header-banner">'
        '<div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1.0rem;">'
        '<div style="max-width: 820px;">'
        '<div class="med-header-title">'
        "Alzheimer's Disease Detection & Explainability System"
        '</div>'
        '<div class="med-header-subtitle">'
        'Explainable Deep Learning-Based Multi-Stage Alzheimer’s Disease Detection Using Brain MRI'
        '</div>'
        '</div>'
        '<div style="text-align: right; white-space: nowrap;">'
        '<span class="med-header-badge">ACADEMIC RESEARCH PROTOTYPE</span>'
        f'<div style="color: #E2E8F0; font-size: 0.74rem; margin-top: 0.35rem; letter-spacing: 0.03em;">{dev_str}</div>'
        '</div>'
        '</div>'
        '</div>'
    )
    render_html(header_html)

