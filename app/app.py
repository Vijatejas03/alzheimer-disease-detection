"""
Explainable Deep Learning-Based Multi-Stage Alzheimer's Disease Detection Using Brain MRI
Main Streamlit Application Entry Point.
"""

from pathlib import Path
import sys

# Robust project root resolution (without fragile hard-coded paths)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = str(Path(__file__).resolve().parent)

# When Streamlit executes app/app.py, it automatically prepends the script directory (app/) to sys.path.
# If app/ is in sys.path, Python resolves 'app' as the app.py module instead of the app/ package.
# Remove APP_DIR and place PROJECT_ROOT at sys.path[0] so 'app' is recognized as a package.
while APP_DIR in sys.path:
    sys.path.remove(APP_DIR)

if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure 'app' in sys.modules is treated as the package (not a stray single-file module)
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import os
import streamlit as st

# Page Configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Alzheimer's Disease Detection & Explainability System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto"
)

# Load and inject custom CSS theme
css_path = PROJECT_ROOT / "app" / "styles" / "custom.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Import Components & Page Views
from app.components.header import render_app_header
from app.components.sidebar import render_sidebar
from app.components.home_page import render_home_page
from app.components.mri_analysis_page import render_mri_analysis_page
from app.components.model_comparison_page import render_model_comparison_page
from app.components.evaluation_results_page import render_evaluation_results_page
from app.components.gradcam_gallery_page import render_gradcam_gallery_page
from app.components.error_robustness_page import render_error_robustness_page
from app.components.methodology_page import render_methodology_page
from app.components.about_page import render_about_page


def main():
    # 1. Render Sidebar Navigation
    current_page = render_sidebar()
    
    # 2. Render Header Banner
    render_app_header(current_page=current_page)
    
    # 3. Route to Selected View (supporting both modern concise labels and legacy aliases)
    try:
        if current_page in ("Overview", "Home"):
            render_home_page()
        elif current_page == "MRI Analysis":
            render_mri_analysis_page()
        elif current_page == "Model Comparison":
            render_model_comparison_page()
        elif current_page in ("Evaluation", "Evaluation Results"):
            render_evaluation_results_page()
        elif current_page in ("Explainability", "Grad-CAM Explainability"):
            render_gradcam_gallery_page()
        elif current_page in ("Error & Robustness", "Robustness & Error Analysis"):
            render_error_robustness_page()
        elif current_page in ("Methodology", "Research Methodology"):
            render_methodology_page()
        elif current_page in ("About", "About & Disclaimer"):
            render_about_page()
        else:
            render_home_page()
    except Exception as page_err:
        st.error(f"An unexpected error occurred while rendering the '{current_page}' view: {page_err}")
        st.info("System remains operational. You may switch to another page via the sidebar.")


if __name__ == "__main__":
    main()
