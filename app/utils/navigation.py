"""
Central navigation state management for Streamlit application.
Allows programmatic page switching from any button, card, or component.
"""

import streamlit as st

PAGE_OPTIONS = [
    "Overview",
    "MRI Analysis",
    "Model Comparison",
    "Evaluation",
    "Explainability",
    "Error & Robustness",
    "Methodology",
    "About"
]


def init_navigation_state():
    """Ensure navigation state is properly initialized in session_state."""
    if "nav_radio" not in st.session_state:
        st.session_state["nav_radio"] = "Overview"
    elif st.session_state["nav_radio"] not in PAGE_OPTIONS:
        st.session_state["nav_radio"] = "Overview"


def navigate_to(page_name: str, rerun: bool = False):
    """
    Set target page in session state for navigation.
    Designed for seamless use as an on_click callback:
    st.button("Label", on_click=navigate_to, args=("MRI Analysis",))
    """
    init_navigation_state()
    if page_name in PAGE_OPTIONS:
        st.session_state["nav_radio"] = page_name
        if rerun:
            st.rerun()


def get_current_page() -> str:
    """Get the currently selected page name."""
    init_navigation_state()
    return st.session_state.get("nav_radio", "Overview")
