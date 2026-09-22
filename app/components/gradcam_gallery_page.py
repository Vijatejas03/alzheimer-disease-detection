"""
Page 5: Grad-CAM Visual Explainability Gallery.
Displays layer-specific gradient attributions for representative test set cases.
"""

from pathlib import Path
import streamlit as st

from app.utils.data_loader import load_gradcam_catalog
from app.utils.inference_engine import CLASS_COLORS
from app.utils.ui_helpers import render_html
from app.components.disclaimer import render_attribution_disclaimer, render_research_disclaimer
from app.components.gradcam_user_guide import render_gradcam_user_guide

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def render_gradcam_gallery_page():
    st.markdown(
        '## Grad-CAM Explainability '
        '<span title="Grad-CAM (Gradient-weighted Class Activation Mapping): Visualizes which image regions contributed most strongly to the model\'s prediction." style="font-size: 0.85rem; font-weight: normal; color: #2563EB; background: #EFF6FF; border: 1px solid #BFDBFE; padding: 2px 8px; border-radius: 4px; cursor: help; vertical-align: middle;">ℹ What is Grad-CAM?</span>',
        unsafe_allow_html=True
    )
    st.markdown("Inspect visual attribution heatmaps highlighting image regions that influenced the model’s prediction.")
    
    catalog = load_gradcam_catalog()
    if not catalog or "models" not in catalog:
        st.warning("Grad-CAM explainability catalog is not available.")
        return
        
    models_dict = catalog["models"]
    
    # Horizontally Aligned Filter Controls
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        model_options = list(models_dict.keys())
        model_labels = {k: models_dict[k]["display_name"] for k in model_options}
        sel_model_slug = st.selectbox(
            "Model Architecture:",
            options=model_options,
            format_func=lambda x: model_labels[x]
        )
        
    with col_f2:
        class_filter = st.selectbox(
            "True Class:",
            options=["All Stages", "Non-Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]
        )
        
    with col_f3:
        status_filter = st.selectbox(
            "Prediction Outcome:",
            options=["All Examples", "Correct Only", "Misclassified Only"]
        )
        
    model_entry = models_dict[sel_model_slug]
    all_examples = model_entry.get("examples", [])
    
    # Apply filters
    filtered_examples = []
    for ex in all_examples:
        if class_filter != "All Stages" and ex["true_class_name"] != class_filter:
            continue
        if status_filter == "Correct Only" and not ex["is_correct"]:
            continue
        if status_filter == "Misclassified Only" and ex["is_correct"]:
            continue
        filtered_examples.append(ex)
        
    if not filtered_examples:
        st.info("No visualizations match the selected filter combination.")
        return
        
    # Case Selector
    case_labels = [
        f"{ex['sample_id']} | True: {ex['true_class_name']} → Pred: {ex['predicted_class_name']} ({'Correct' if ex['is_correct'] else 'Misclassified'}, {ex['confidence']:.1%})"
        for ex in filtered_examples
    ]
    selected_idx = st.selectbox("Select Test Image to Inspect:", options=range(len(filtered_examples)), format_func=lambda i: case_labels[i])
    current_case = filtered_examples[selected_idx]
    
    st.markdown("---")
    
    # Display 3-Panel Diagnostic Figure
    panel_rel_path = current_case.get("panel_rel_path", "")
    panel_abs_path = PROJECT_ROOT / panel_rel_path
    
    if panel_abs_path.exists():
        st.image(str(panel_abs_path), use_container_width=True, caption=f"Grad-CAM 3-Panel Attribution: {current_case['sample_id']} ({model_labels[sel_model_slug]})")
    else:
        c_p1, c_p2 = st.columns(2)
        ov_path = PROJECT_ROOT / current_case.get("overlay_rel_path", "")
        hm_path = PROJECT_ROOT / current_case.get("heatmap_rel_path", "")
        with c_p1:
            if hm_path.exists():
                st.image(str(hm_path), caption="Grad-CAM Heatmap", use_container_width=True)
        with c_p2:
            if ov_path.exists():
                st.image(str(ov_path), caption="Grad-CAM Overlay", use_container_width=True)

    # 4 Clean Metadata Cards Underneath
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    true_color = CLASS_COLORS.get(current_case["true_class_name"], "#2563EB")
    pred_color = CLASS_COLORS.get(current_case["predicted_class_name"], "#2563EB")
    is_corr = current_case["is_correct"]
    
    with col_m1:
        render_html(
            '<div class="med-metric-card" style="background: #FFFFFF;">'
            f'<div style="font-size: 1.05rem; font-weight: 700; color: {true_color};">{current_case["true_class_name"]}</div>'
            '<div class="med-metric-lbl">True Class</div>'
            '</div>'
        )
    with col_m2:
        status_tag = ' <span style="font-size: 0.72rem; color: #16A34A; font-weight: 700;">(Match)</span>' if is_corr else ' <span style="font-size: 0.72rem; color: #DC2626; font-weight: 700;">(Mismatch)</span>'
        render_html(
            '<div class="med-metric-card" style="background: #FFFFFF;">'
            f'<div style="font-size: 1.05rem; font-weight: 700; color: {pred_color};">{current_case["predicted_class_name"]}{status_tag}</div>'
            '<div class="med-metric-lbl">Predicted Class</div>'
            '</div>'
        )
    with col_m3:
        render_html(
            '<div class="med-metric-card" style="background: #FFFFFF;" title="Model Confidence: Softmax probability of the top predicted class, representing computational certainty rather than clinical diagnosis.">'
            f'<div class="med-metric-val">{current_case["confidence"]:.1%}</div>'
            '<div class="med-metric-lbl" style="cursor: help;">Model Confidence ℹ</div>'
            '</div>'
        )
    with col_m4:
        render_html(
            '<div class="med-metric-card" style="background: #FFFFFF;">'
            f'<div style="font-size: 1.05rem; font-weight: 700; color: #0F4C81;">{model_labels[sel_model_slug]}</div>'
            '<div class="med-metric-lbl">Architecture</div>'
            '</div>'
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    # Softmax Probability Breakdown
    st.markdown("### Softmax Probability Distribution")
    p_dict = current_case.get("probabilities", {})
    p_cols = st.columns(4)
    for col, (cls_k, p_val) in zip(p_cols, p_dict.items()):
        c_c = CLASS_COLORS.get(cls_k, "#2563EB")
        with col:
            render_html(
                '<div class="med-card" style="padding: 0.75rem 0.85rem; text-align: center; margin-bottom: 0.5rem; background: #FFFFFF;">'
                f'<div style="font-size: 1.25rem; font-weight: 750; color: {c_c};">{p_val*100:.1f}%</div>'
                f'<div style="font-size: 0.76rem; font-weight: 650; color: #334155; margin-top: 0.2rem;">{cls_k}</div>'
                '</div>'
            )

    # Technical Details Expander
    with st.expander("Technical Saliency Details"):
        st.write(f"**Target Layer:** `{current_case['target_layer']}`")
        st.write(
            "Grad-CAM computes the gradients of the predicted class score with respect to feature maps of the target convolutional layer. "
            "Global average pooling extracts channel importance weights, followed by ReLU rectification to isolate features "
            "that positively contribute to the target classification score."
        )

    # Comprehensive User Guide for Normal Users
    render_gradcam_user_guide()
    render_attribution_disclaimer()
    render_research_disclaimer()
