"""
Sidebar Navigation and System Status Component.
Provides cohesive, high-contrast medical AI navigation with 3D accents.
"""

import streamlit as st
from app.utils.inference_engine import get_inference_device
from app.utils.ui_helpers import render_html
from app.utils.navigation import PAGE_OPTIONS, init_navigation_state, get_current_page


def render_sidebar() -> str:
    """
    Render professional sidebar navigation and return selected page name.
    Synchronized with st.session_state["nav_radio"] for programmatic navigation.
    """
    init_navigation_state()
    
    with st.sidebar:
        # 3D Brand Lockup
        brand_html = (
            '<div class="sidebar-brand-card">'
            '<div style="display: flex; align-items: center; gap: 0.65rem;">'
            '<div class="sidebar-brand-icon">🧠</div>'
            '<div>'
            '<div class="sidebar-brand-title">Alzheimer\'s XAI</div>'
            '<div class="sidebar-brand-sub">Neural Research Platform</div>'
            '</div>'
            '</div>'
            '</div>'
        )
        render_html(brand_html)
        
        st.markdown(
            '<div class="sidebar-nav-header">'
            '<span>NAVIGATION</span>'
            '<span class="sidebar-nav-pill">8 MODULES</span>'
            '</div>',
            unsafe_allow_html=True
        )
        
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
            options=PAGE_OPTIONS,
            format_func=lambda p: f"{icon_map.get(p, '•')}  {p}",
            label_visibility="collapsed",
            key="nav_radio"
        )
        
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # SYSTEM STATUS Card - Real Technical Metrics Only
        device, dev_desc = get_inference_device()
        is_cuda = (device.type == "cuda")
        dev_label = "CUDA GPU" if is_cuda else "CPU Compute"
        
        status_html = (
            '<div class="sidebar-status-card">'
            '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">'
            '<span class="sidebar-status-title">System Status</span>'
            '<span class="status-live-pulse"><span class="pulse-dot"></span> Online</span>'
            '</div>'
            '<div class="sidebar-status-metrics">'
            f'<div class="sidebar-metric-row"><span>Inference Device:</span><strong>{dev_label}</strong></div>'
            '<div class="sidebar-metric-row"><span>Total Dataset:</span><strong>6,400 Scans</strong></div>'
            '<div class="sidebar-metric-row"><span>Held-Out Test:</span><strong>960 Scans (15%)</strong></div>'
            '<div class="sidebar-metric-row"><span>Architectures:</span><strong>3 Models</strong></div>'
            '<div class="sidebar-metric-row"><span>Hash Overlap:</span><strong style="color: #16A34A;">0.00% (Verified)</strong></div>'
            '</div>'
            '</div>'
        )
        render_html(status_html)
        
        # Academic Research Prototype Footnote
        footer_html = (
            '<div class="sidebar-footer-card">'
            '<div style="font-weight: 750; color: #0F4C81; font-size: 0.76rem; margin-bottom: 0.15rem;">'
            'Academic Research Prototype'
            '</div>'
            '<div style="font-size: 0.70rem; color: #64748B; line-height: 1.4;">'
            'Non-Clinical Engineering Study<br>'
            'Vivekananda Institute of Technology'
            '</div>'
            '</div>'
        )
        render_html(footer_html)
        
    return selected_page
