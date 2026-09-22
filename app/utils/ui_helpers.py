"""
UI Rendering Helpers and Theme Standards for Alzheimer's Detection System.
Guarantees consistent, high-contrast, safe HTML rendering without Markdown escaping.
"""

import streamlit as st


def render_html(html_str: str):
    """Safely render unindented HTML markup with guaranteed unsafe_allow_html=True."""
    # Ensure no leading whitespace blocks that could trigger markdown code block parsing
    cleaned = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def render_badge(text: str, badge_type: str = "info") -> str:
    """Generate HTML string for a status badge chip."""
    return f'<span class="badge-chip badge-chip-{badge_type}">{text}</span>'
