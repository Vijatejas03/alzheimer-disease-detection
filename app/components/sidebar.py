"""
Sidebar Navigation and System Status Component.
Provides cohesive, high-contrast medical AI navigation.
"""

import streamlit as st
from app.utils.inference_engine import get_inference_device
from app.utils.ui_helpers import render_html


def render_sidebar() -> str:
    """
    Render professional sidebar navigation and return selected page name.
    """
    with st.sidebar:
        # Header Brand Lockup
        brand_html = (
            '<div style="padding: 0.35rem 0 0.95rem 0; border-bottom: 1.5px solid #E2E8F0; margin-bottom: 0.85rem;">'
            '<div style="font-size: 1.28rem; font-weight: 800; color: #0F4C81; letter-spacing: -0.015em; line-height: 1.2;">'
            "🧠 Alzheimer's XAI"
            '</div>'
            '<div style="font-size: 0.80rem; color: #64748B; font-weight: 650; margin-top: 0.25rem; letter-spacing: 0.03em; text-transform: uppercase;">'
            'Research Platform'
            '</div>'
            '</div>'
        )
        render_html(brand_html)
        
        st.markdown(
            '<div style="font-size: 0.72rem; font-weight: 750; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.45rem;">'
            'Navigation'
            '</div>',
            unsafe_allow_html=True
        )
        
        page_options = [
            "Overview",
            "MRI Analysis",
            "Model Comparison",
            "Evaluation",
            "Explainability",
            "Error & Robustness",
            "Methodology",
            "About"
        ]
        
        # Clean professional icons
        icon_map = {
            "Overview": "▤",
            "MRI Analysis": "⊕",
            "Model Comparison": "⊞",
            "Evaluation": "▲",
            "Explainability": "◎",
            "Error & Robustness": "⚡",
            "Methodology": "≡",
            "About": "ℹ"
        }
        
        selected_page = st.radio(
            "Select View:",
            options=page_options,
            format_func=lambda p: f"{icon_map.get(p, '•')}  {p}",
            label_visibility="collapsed"
        )
        
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        
        # SYSTEM STATUS Card - Real Technical Metrics Only
        device, dev_desc = get_inference_device()
        is_cuda = (device.type == "cuda")
        dev_label = "CUDA Enabled" if is_cuda else "CPU Mode"
        
        status_html = (
            '<div class="med-card" style="padding: 0.85rem 0.95rem; margin-bottom: 0.75rem; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px;">'
            '<div style="font-size: 0.72rem; font-weight: 800; color: #0F4C81; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem;">'
            'Research Platform Spec'
            '</div>'
            '<div style="font-size: 0.80rem; color: #1E293B; line-height: 1.75;">'
            f'• <strong>Device:</strong> {dev_label}<br>'
            '• <strong>Dataset:</strong> 6,400 Images<br>'
            '• <strong>Test Split:</strong> 960 Scans (15%)<br>'
            '• <strong>Architectures:</strong> 3 Models'
            '</div>'
            '</div>'
        )
        render_html(status_html)
        
        # Academic Research Prototype Footnote
        footer_html = (
            '<div style="font-size: 0.72rem; color: #64748B; text-align: center; margin-top: 0.75rem; line-height: 1.5; padding: 0.4rem 0.2rem; border-top: 1px solid #E2E8F0;">'
            '<strong>Academic Research Prototype</strong><br>'
            'Non-Clinical Engineering Study'
            '</div>'
        )
        render_html(footer_html)
        
    return selected_page
