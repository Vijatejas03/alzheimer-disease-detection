"""
Standard Academic & Research Prototype Disclaimer Component.
"""

from app.utils.ui_helpers import render_html


def render_research_disclaimer():
    """Render restrained academic research disclaimer callout."""
    disclaimer_html = (
        '<div class="disclaimer-alert">'
        '<div class="disclaimer-alert-title">Academic & Research Prototype Notice</div>'
        'This system is intended for educational and algorithmic demonstration purposes only. '
        'It has not been clinically validated and must not be used for medical diagnosis, '
        'screening, or clinical decision-making.'
        '</div>'
    )
    render_html(disclaimer_html)


def render_attribution_disclaimer():
    """Render Grad-CAM attribution clarification disclaimer."""
    attribution_html = (
        '<div style="background-color: #F0F9FF; border: 1px solid #BAE6FD; border-left: 4px solid #0284C7; border-radius: 6px; padding: 0.8rem 1.1rem; margin: 0.9rem 0; font-size: 0.84rem; color: #334155; line-height: 1.55;">'
        '<strong style="color: #0369A1;">Scientific Attribution Clarification:</strong> '
        'Grad-CAM visualizations identify image regions that contributed to the model’s prediction. '
        'These visualizations are attribution maps and do not establish anatomical biomarkers, pathology, disease mechanism, or clinical validity.'
        '</div>'
    )
    render_html(attribution_html)

