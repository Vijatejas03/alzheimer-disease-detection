"""
Grad-CAM User Guide & Attribution Explanation Component.
Provides first-time users with clear, plain-language guidance on:
- Interpreting the 3 Grad-CAM panels (Original MRI, Heatmap, Overlay)
- Horizontal visual gradient color legend (Blue -> Green -> Yellow/Orange -> Red)
- Strict scientific distinction (Model Attribution vs Disease Severity)
- 5-step operational expander on how Grad-CAM works
- Important notice on explainability visualization
"""

import streamlit as st
from app.utils.ui_helpers import render_html


def render_gradcam_user_guide():
    """
    Render a polished, accessible 'How to Read This Explanation' research card
    with a horizontal visual gradient color legend, 3 panel descriptions,
    strict scientific guardrails, and a 5-step operational expander.
    """
    guide_html = (
        '<div class="med-card" style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.15rem 1.35rem; margin-top: 1.0rem; margin-bottom: 0.9rem; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);">'
        '<!-- Header -->'
        '<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.35rem;">'
        '<div style="display: flex; align-items: center; gap: 0.5rem;">'
        '<span style="font-size: 1.15rem;">💡</span>'
        '<div style="font-size: 1.05rem; font-weight: 750; color: #0F172A; letter-spacing: -0.01em;">'
        'How to Read This Explanation'
        '</div>'
        '</div>'
        '<span title="Model Attribution: Mathematical contribution score of receptive field activations toward the predicted class." style="font-size: 0.74rem; font-weight: 700; color: #2563EB; background: #EFF6FF; border: 1px solid #BFDBFE; padding: 2px 8px; border-radius: 4px; cursor: help;">'
        'Model Attribution ℹ'
        '</span>'
        '</div>'
        
        '<div style="font-size: 0.88rem; color: #334155; line-height: 1.55; margin-bottom: 0.95rem;">'
        "Grad-CAM highlights image regions that contributed to the selected model's prediction."
        '</div>'
        
        '<!-- 3 Panel Explanations Grid -->'
        '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.75rem; margin-bottom: 1.1rem;">'
        '<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.75rem 0.9rem;">'
        '<div style="font-size: 0.78rem; font-weight: 750; color: #0F4C81; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.25rem;">Original MRI</div>'
        '<div style="font-size: 0.82rem; color: #475569; line-height: 1.45;">The original brain MRI used for analysis.</div>'
        '</div>'
        '<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.75rem 0.9rem;">'
        '<div style="font-size: 0.78rem; font-weight: 750; color: #0F4C81; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.25rem;">Grad-CAM Heatmap</div>'
        '<div style="font-size: 0.82rem; color: #475569; line-height: 1.45;">A color map showing the relative strength of image-region contributions to the model prediction.</div>'
        '</div>'
        '<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.75rem 0.9rem;">'
        '<div style="font-size: 0.78rem; font-weight: 750; color: #0F4C81; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.25rem;">Grad-CAM Overlay</div>'
        '<div style="font-size: 0.82rem; color: #475569; line-height: 1.45;">The attribution map overlaid on the MRI to make contributing regions easier to locate.</div>'
        '</div>'
        '</div>'
        
        '<!-- Color Legend Heading -->'
        '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">'
        '<div style="font-size: 0.80rem; font-weight: 750; color: #0F172A; text-transform: uppercase; letter-spacing: 0.04em;">'
        'Attribution Color Legend'
        '</div>'
        '<div style="font-size: 0.74rem; color: #64748B;">Low → High Model Contribution</div>'
        '</div>'
        
        '<!-- Horizontal Visual Gradient Bar (Jet Colormap) -->'
        '<div style="height: 14px; border-radius: 7px; background: linear-gradient(to right, #000080, #0000FF, #00BFFF, #00FF80, #80FF00, #FFFF00, #FF7F00, #FF0000); border: 1px solid #CBD5E1; margin-bottom: 0.45rem;"></div>'
        
        '<!-- Color Key Labels -->'
        '<div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.5rem; font-size: 0.78rem; color: #334155; margin-bottom: 1.05rem;">'
        '<div><strong style="color: #0000FF;">Blue</strong> → Low contribution</div>'
        '<div><strong style="color: #059669;">Green</strong> → Moderate contribution</div>'
        '<div><strong style="color: #D97706;">Yellow/Orange</strong> → High contribution</div>'
        '<div><strong style="color: #DC2626;">Red</strong> → Strongest contribution</div>'
        '</div>'
        
        '<!-- Scientific Distinction Notice -->'
        '<div style="background-color: #F8FAFC; border-left: 4px solid #0F4C81; border-radius: 4px; padding: 0.75rem 0.95rem; margin-bottom: 0.75rem; font-size: 0.84rem; color: #1E293B; line-height: 1.55;">'
        '<strong>Scientific Attribution Notice:</strong> These colors represent <strong>model attribution</strong>, not disease severity or confirmed brain pathology.'
        '</div>'
        
        '<!-- Important Notice -->'
        '<div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #D97706; border-radius: 4px; padding: 0.75rem 0.95rem; font-size: 0.82rem; color: #92400E; line-height: 1.55;">'
        '<strong>Important:</strong> Grad-CAM is an explainability visualization. It provides a visual representation of which image regions contributed to the model\'s prediction. These highlighted regions are model attribution and are not proof of specific anatomical biomarkers or clinical findings.'
        '</div>'
        '</div>'
    )
    render_html(guide_html)
    
    # 5-Step Simple Mechanism Expander
    with st.expander("How does Grad-CAM work?", expanded=False):
        st.markdown(
            """
**Step 1 — Deep Learning Analysis:**  
The MRI is processed by the selected deep-learning model.

**Step 2 — Model Output:**  
The model produces a prediction for one of the four classes.

**Step 3 — Gradient Backpropagation:**  
Grad-CAM traces relevant activation information back toward the image.

**Step 4 — Regional Heatmap:**  
A heatmap is generated showing relative regional contribution.

**Step 5 — Visual Alignment:**  
The heatmap is overlaid on the MRI for easier visual interpretation.
            """
        )
