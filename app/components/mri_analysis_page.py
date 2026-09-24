"""
Page 2: MRI Analysis & Explainability Demonstration (Hero Feature).
Provides multi-stage validation, side-by-side preprocessing visualization,
single-architecture prediction, multi-model consensus agreement engine,
and Grad-CAM attribution mapping.
"""

import os
import io
from datetime import datetime
from PIL import Image
import streamlit as st
import matplotlib.cm as cm
import numpy as np

from app.utils.inference_engine import (
    MODEL_CONFIGS,
    CLASS_NAMES,
    CLASS_COLORS,
    load_cached_model,
    preprocess_image_for_inference,
    run_model_inference,
    generate_gradcam,
    get_inference_device
)
from app.utils.input_validator import (
    validate_input_image,
    compute_uncertainty_metrics,
    ValidationStatus
)
from app.utils.data_loader import get_sample_test_images
from app.utils.ui_helpers import render_html
from app.components.disclaimer import render_attribution_disclaimer, render_research_disclaimer
from app.components.gradcam_user_guide import render_gradcam_user_guide


def render_mri_analysis_page():
    # Page Header
    st.markdown("## MRI Analysis")
    st.markdown(
        '<div style="font-size: 0.95rem; color: #475569; margin-top: -0.4rem; margin-bottom: 1.25rem;">'
        'Upload a brain MRI to evaluate the trained research models with input validation, '
        'multi-model consensus agreement, and Grad-CAM explainability.'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Step-by-Step Workflow Ribbon
    ribbon_html = (
        '<div class="step-flow-ribbon">'
        '<div class="step-flow-item active"><span class="step-flow-num">01</span> Input Acquisition</div>'
        '<span style="color: #CBD5E1;">➔</span>'
        '<div class="step-flow-item"><span class="step-flow-num">02</span> Quality Gate</div>'
        '<span style="color: #CBD5E1;">➔</span>'
        '<div class="step-flow-item"><span class="step-flow-num">03</span> Preprocessing</div>'
        '<span style="color: #CBD5E1;">➔</span>'
        '<div class="step-flow-item"><span class="step-flow-num">04</span> Inference</div>'
        '<span style="color: #CBD5E1;">➔</span>'
        '<div class="step-flow-item"><span class="step-flow-num">05</span> Grad-CAM</div>'
        '</div>'
    )
    render_html(ribbon_html)
    
    # =========================================================================
    # 1. UPLOAD WORKSPACE
    # =========================================================================
    st.markdown("### 01 Input Acquisition")
    
    input_mode = st.radio(
        "Select Image Acquisition Method:",
        options=["Upload Brain MRI Scan", "Load Curated Research Sample"],
        horizontal=True
    )
    
    raw_uploaded_bytes = None
    input_source_name = "Uploaded Scan"
    source_pil_img = None
    
    if input_mode == "Upload Brain MRI Scan":
        dropzone_html = (
            '<div class="dropzone-3d-card">'
            '<div class="dropzone-icon">🧠</div>'
            '<div class="dropzone-title">DROP OR SELECT AXIAL BRAIN MRI</div>'
            '<div class="dropzone-sub">T1-Weighted Structural Neuroimaging • Single-Channel Grayscale or RGB (JPG, PNG, WEBP)</div>'
            '</div>'
        )
        render_html(dropzone_html)
        
        uploaded_file = st.file_uploader(
            "Drag and drop or select an axial brain MRI scan:",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported Formats: JPG, JPEG, PNG, WEBP"
        )
        st.caption("Accepted inputs: **JPG · JPEG · PNG · WEBP** (Single-channel or RGB axial brain MRI slice)")
        if uploaded_file is not None:
            raw_uploaded_bytes = uploaded_file.getvalue()
            input_source_name = uploaded_file.name
            try:
                source_pil_img = Image.open(io.BytesIO(raw_uploaded_bytes))
            except Exception:
                pass
    else:
        sample_catalog = get_sample_test_images()
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            stage_choice = st.selectbox(
                "Filter by Canonical Class:",
                options=list(sample_catalog.keys())
            )
        with col_s2:
            scans_in_stage = sample_catalog[stage_choice]
            scan_options = {f"{s['label']} ({s['filename']})": s for s in scans_in_stage}
            chosen_scan_label = st.selectbox("Select Benchmark Test Scan:", options=list(scan_options.keys()))
            chosen_scan_info = scan_options[chosen_scan_label]
            
        sample_path = chosen_scan_info["path"]
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as fp:
                raw_uploaded_bytes = fp.read()
            input_source_name = chosen_scan_info["filename"]
            try:
                source_pil_img = Image.open(sample_path)
            except Exception:
                pass
        else:
            st.error(f"Sample file not found at: {sample_path}")

    if raw_uploaded_bytes is None:
        placeholder_html = (
            '<div class="med-card" style="text-align: center; padding: 2.5rem 1.5rem; border: 2px dashed #CBD5E1; background: #FFFFFF; border-radius: 10px; margin-top: 0.8rem;">'
            '<div style="font-size: 1.4rem; margin-bottom: 0.35rem;">🧠</div>'
            '<div style="font-size: 1.05rem; font-weight: 750; color: #0F172A; margin-bottom: 0.35rem;">'
            'Awaiting Brain MRI Input'
            '</div>'
            '<div style="font-size: 0.86rem; color: #64748B; max-width: 520px; margin: 0 auto;">'
            'Upload an axial brain MRI scan or select a benchmark sample above to initiate automated validation and inference.'
            '</div>'
            '</div>'
        )
        render_html(placeholder_html)
        return

    # =========================================================================
    # 2. INPUT QUALITY VALIDATION CARD
    # =========================================================================
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("### 02 Input Quality Validation")
    
    val_result = validate_input_image(raw_uploaded_bytes, filename=input_source_name)
    metrics = val_result.metrics
    
    # Status Banner
    if val_result.status == ValidationStatus.PASS:
        callout_html = (
            '<div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-left: 4px solid #16A34A; border-radius: 8px; padding: 0.85rem 1.2rem; margin-bottom: 0.85rem;">'
            '<div style="font-weight: 800; font-size: 0.86rem; color: #166534; text-transform: uppercase; letter-spacing: 0.04em;">STATUS: INPUT VALIDATION PASSED</div>'
            '<div style="font-size: 0.84rem; color: #166534; margin-top: 0.15rem;">The scan satisfies all integrity, format, aspect ratio, and contrast requirements for neural network inference.</div>'
            '</div>'
        )
        render_html(callout_html)
    elif val_result.status == ValidationStatus.WARNING:
        callout_html = (
            '<div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #D97706; border-radius: 8px; padding: 0.85rem 1.2rem; margin-bottom: 0.85rem;">'
            '<div style="font-weight: 800; font-size: 0.86rem; color: #B45309; text-transform: uppercase; letter-spacing: 0.04em;">STATUS: SUITABILITY WARNING</div>'
            f'<div style="font-size: 0.84rem; color: #B45309; margin-top: 0.15rem;">{val_result.message}</div>'
            '</div>'
        )
        render_html(callout_html)
    else:  # REJECTED
        # Prominent Input Rejected Warning Box
        callout_html = (
            '<div style="background-color: #FEF2F2; border: 1.5px solid #F87171; border-left: 6px solid #DC2626; border-radius: 10px; padding: 1.1rem 1.4rem; margin-bottom: 1.2rem; box-shadow: 0 2px 8px rgba(220, 38, 38, 0.08);">'
            '<div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;">'
            '<span style="font-size: 1.3rem;">⚠️</span>'
            '<span style="font-weight: 800; font-size: 1.05rem; color: #991B1B; text-transform: uppercase; letter-spacing: 0.04em;">Input rejected</span>'
            '</div>'
            f'<div style="font-size: 0.96rem; font-weight: 700; color: #B91C1C; margin-bottom: 0.45rem;">{val_result.message}</div>'
            '<div style="font-size: 0.85rem; color: #7F1D1D; line-height: 1.55;">'
            'This research platform is trained <strong>exclusively on axial brain MRI scans</strong> (T1-weighted structural neuroimaging) '
            'and cannot make valid predictions on out-of-domain images. To preserve scientific validity and prevent arbitrary predictions, '
            'the inference pipeline and Grad-CAM generation have been completely halted.'
            '</div>'
            '</div>'
        )
        render_html(callout_html)
        st.warning("⚠️ **Input rejected:** This image does not appear to be a suitable brain MRI for this research model. Please upload a brain MRI image.")

        # Breakdown of Failed Checks
        failed_checks = [c for c in val_result.checks if not c.passed]
        if failed_checks:
            st.markdown("##### Verification Failure Details")
            for fc in failed_checks:
                st.markdown(
                    f'<div style="background: #FFF5F5; border: 1px solid #FED7D7; border-radius: 6px; padding: 0.55rem 0.85rem; margin-bottom: 0.4rem; font-size: 0.83rem; color: #9B2C2C;">'
                    f'<strong>✕ {fc.name}:</strong> {fc.message}'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Side-by-side: Ingested Image Preview + Input Domain Reference Guide
        col_img, col_guide = st.columns([1, 1.3])
        with col_img:
            st.markdown(
                '<div class="med-card" style="padding: 0.75rem; text-align: center; margin-bottom: 0.35rem; background: #FFFFFF;">'
                '<div style="font-size: 0.76rem; font-weight: 800; color: #DC2626; text-transform: uppercase; letter-spacing: 0.05em;">'
                'INGESTED IMAGE (INFERENCE BLOCKED)'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )
            if source_pil_img:
                st.image(source_pil_img, use_container_width=True, caption=f"Rejected Input: {input_source_name}")
            elif val_result.sanitized_image:
                st.image(val_result.sanitized_image, use_container_width=True, caption=f"Rejected Input: {input_source_name}")
                
        with col_guide:
            guide_html = (
                '<div class="med-card" style="padding: 1rem 1.1rem; background: #FFFFFF; border: 1px solid #E2E8F0;">'
                '<div style="font-size: 0.88rem; font-weight: 800; color: #0F172A; margin-bottom: 0.65rem;">'
                '📋 Input Domain Policy & Requirements'
                '</div>'
                '<div style="margin-bottom: 0.7rem;">'
                '<div style="font-size: 0.78rem; font-weight: 750; color: #166534; text-transform: uppercase; margin-bottom: 0.25rem;">'
                '✔ Supported Input Domain'
                '</div>'
                '<div style="font-size: 0.80rem; color: #334155; line-height: 1.45;">'
                '• Axial brain MRI slices (T1-weighted structural scans)<br>'
                '• Centered cranial cranium with dark background margins<br>'
                '• Monochromatic grayscale format<br>'
                '• Near-square proportions (~1:1 aspect ratio)'
                '</div>'
                '</div>'
                '<div>'
                '<div style="font-size: 0.78rem; font-weight: 750; color: #991B1B; text-transform: uppercase; margin-bottom: 0.25rem;">'
                '✖ Unsupported / Out-of-Domain Inputs'
                '</div>'
                '<div style="font-size: 0.80rem; color: #334155; line-height: 1.45;">'
                '• Dog, cat, or other animal photographs<br>'
                '• Human photographs, portraits, or selfies<br>'
                '• Digital artwork or AI-generated brain illustrations<br>'
                '• Screenshots, documents, or UI graphics<br>'
                '• Chest CT, Chest MRI, or lung scans<br>'
                '• Spine, abdomen, or pelvis scans<br>'
                '• Knee, shoulder, or extremity bone scans<br>'
                '• X-rays or unrelated medical imaging'
                '</div>'
                '</div>'
                '</div>'
            )
            render_html(guide_html)
            
        # Metadata Card
        fmt = metrics.get("format", "Unknown")
        res_str = f"{metrics.get('width', 0)} × {metrics.get('height', 0)} px"
        ch_str = f"{metrics.get('channels', 1)} ({metrics.get('mode', 'L')})"
        ar_val = metrics.get("aspect_ratio", 1.0)
        chroma_val = metrics.get("chromatic_divergence", 0.0)
        border_val = metrics.get("border_mean_intensity", 0.0)
        
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        q_c1, q_c2, q_c3, q_c4, q_c5 = st.columns(5)
        with q_c1:
            render_html(f'<div class="med-card" style="padding: 0.55rem; text-align: center;"><div style="font-size: 0.95rem; font-weight: 750;">{fmt}</div><div class="med-metric-lbl">Format</div></div>')
        with q_c2:
            render_html(f'<div class="med-card" style="padding: 0.55rem; text-align: center;"><div style="font-size: 0.95rem; font-weight: 750;">{res_str}</div><div class="med-metric-lbl">Resolution</div></div>')
        with q_c3:
            render_html(f'<div class="med-card" style="padding: 0.55rem; text-align: center;"><div style="font-size: 0.95rem; font-weight: 750;">{ar_val:.2f}:1</div><div class="med-metric-lbl">Aspect Ratio</div></div>')
        with q_c4:
            render_html(f'<div class="med-card" style="padding: 0.55rem; text-align: center;"><div style="font-size: 0.95rem; font-weight: 750; color: #DC2626 if {chroma_val} > 8 else #0F172A;">{chroma_val:.1f}</div><div class="med-metric-lbl">Chroma Score</div></div>')
        with q_c5:
            render_html(f'<div class="med-card" style="padding: 0.55rem; text-align: center;"><div style="font-size: 0.95rem; font-weight: 750; color: #DC2626 if {border_val} > 30 else #0F172A;">{border_val:.1f}</div><div class="med-metric-lbl">Border Mean</div></div>')
            
        st.info("💡 To analyze an MRI, please select one of the curated research samples above or upload an axial brain MRI scan.")
        return

    # Real Metadata Grid (Format, Resolution, Channels, Aspect ratio, Dynamic Range)
    fmt = metrics.get("format", "Unknown")
    res_str = f"{metrics.get('width', 0)} × {metrics.get('height', 0)} px"
    ch_str = f"{metrics.get('channels', 1)} ({metrics.get('mode', 'L')})"
    ar_val = metrics.get("aspect_ratio", 1.0)
    dyn_min = metrics.get("min_intensity", 0)
    dyn_max = metrics.get("max_intensity", 255)
    std_val = metrics.get("std_intensity", 0.0)
    
    q_c1, q_c2, q_c3, q_c4, q_c5 = st.columns(5)
    with q_c1:
        render_html(
            '<div class="med-card" style="padding: 0.65rem 0.8rem; text-align: center; background: #FFFFFF;">'
            f'<div style="font-size: 1.05rem; font-weight: 750; color: #0F172A;">{fmt}</div>'
            '<div class="med-metric-lbl">Format</div>'
            '</div>'
        )
    with q_c2:
        render_html(
            '<div class="med-card" style="padding: 0.65rem 0.8rem; text-align: center; background: #FFFFFF;">'
            f'<div style="font-size: 1.05rem; font-weight: 750; color: #0F172A;">{res_str}</div>'
            '<div class="med-metric-lbl">Resolution</div>'
            '</div>'
        )
    with q_c3:
        render_html(
            '<div class="med-card" style="padding: 0.65rem 0.8rem; text-align: center; background: #FFFFFF;">'
            f'<div style="font-size: 1.05rem; font-weight: 750; color: #0F172A;">{ch_str}</div>'
            '<div class="med-metric-lbl">Channels / Mode</div>'
            '</div>'
        )
    with q_c4:
        render_html(
            '<div class="med-card" style="padding: 0.65rem 0.8rem; text-align: center; background: #FFFFFF;">'
            f'<div style="font-size: 1.05rem; font-weight: 750; color: #0F172A;">{ar_val:.2f} : 1</div>'
            '<div class="med-metric-lbl">Aspect Ratio</div>'
            '</div>'
        )
    with q_c5:
        render_html(
            '<div class="med-card" style="padding: 0.65rem 0.8rem; text-align: center; background: #FFFFFF;">'
            f'<div style="font-size: 1.05rem; font-weight: 750; color: #0F172A;">[{dyn_min}, {dyn_max}]</div>'
            f'<div class="med-metric-lbl">Dynamic Range (σ={std_val:.1f})</div>'
            '</div>'
        )

    # Halt safely if rejected
    if not val_result.can_run_inference:
        st.error("Inference Blocked: Safe inference gate intercepted an invalid or out-of-domain input. Model execution has been halted to prevent arbitrary predictions.")
        return

    # Preprocessing Image Preparation
    image_to_process = val_result.sanitized_image
    tensor_input, display_img, meta = preprocess_image_for_inference(image_to_process)

    # =========================================================================
    # 3. PREPROCESSING VISUALIZATION (Side-by-Side Desktop / Stacked Mobile)
    # =========================================================================
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("### 03 Preprocessing Pipeline")
    st.caption("Grayscale normalization to 3-channel RGB, spatial resampling to 224×224, and ImageNet standardization.")
    
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        st.markdown(
            '<div class="med-card" style="padding: 0.75rem; text-align: center; margin-bottom: 0.35rem; background: #FFFFFF;">'
            '<div style="font-size: 0.76rem; font-weight: 800; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">'
            f'ORIGINAL SCAN ({meta["original_width"]} × {meta["original_height"]})'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )
        if source_pil_img:
            st.image(source_pil_img, use_container_width=True, caption=f"Raw Ingested Image: {input_source_name}")
        else:
            st.image(image_to_process, use_container_width=True, caption="Ingested Image")
            
    with col_pre2:
        st.markdown(
            '<div class="med-card" style="padding: 0.75rem; text-align: center; margin-bottom: 0.35rem; background: #FFFFFF;">'
            '<div style="font-size: 0.76rem; font-weight: 800; color: #2563EB; text-transform: uppercase; letter-spacing: 0.05em;">'
            'MODEL INPUT TENSOR (224 × 224 × 3 RGB)'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.image(display_img, use_container_width=True, caption="Standardized Tensor (Bilinear Resampled, ImageNet Normalized)")

    # =========================================================================
    # 4. MODEL SELECTION & INFERENCE
    # =========================================================================
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("### 04 Deep Learning Inference")
    
    col_sel, col_info = st.columns([1.2, 1.8])
    with col_sel:
        model_names = list(MODEL_CONFIGS.keys())
        default_index = model_names.index("ResNet18") if "ResNet18" in model_names else 0
        selected_model_name = st.selectbox(
            "Primary Architecture for Inspection:",
            options=model_names,
            index=default_index
        )
    with col_info:
        model_cfg = MODEL_CONFIGS.get(selected_model_name, {})
        # Derive formatted parameters safely from config or metadata fallbacks
        params_formatted = (
            model_cfg.get("params_formatted")
            or model_cfg.get("params")
            or (f"{model_cfg['total_parameters']:,} (~{model_cfg['total_parameters']/1e6:.2f}M)" if "total_parameters" in model_cfg else "N/A")
        )
        device, device_desc = get_inference_device()
        dev_tag = "CUDA GPU Accelerated" if device.type == "cuda" else "CPU Computation"
        st.markdown(
            f'<div class="med-card" style="padding: 0.65rem 0.95rem; margin-top: 1.6rem; background: #FFFFFF; border-left: 3px solid #2563EB;">'
            f'<div style="font-size: 0.82rem; color: #1E293B;">'
            f'<strong>{selected_model_name}</strong> • Parameters: {params_formatted} • Input: 224×224×3 • {dev_tag}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Load Model and Run Inference
    primary_model, load_msg = load_cached_model(selected_model_name)
    if primary_model is None:
        st.error(f"Model checkpoint is unavailable: {load_msg}")
        return

    with st.spinner(f"Running inference with {selected_model_name}..."):
        inference_res = run_model_inference(primary_model, tensor_input)
        
    pred_class = inference_res["predicted_class"]
    pred_idx = inference_res["predicted_idx"]
    confidence = inference_res["confidence"]
    probs = inference_res["probabilities"]
    latency = inference_res["latency_ms"]
    pred_color = CLASS_COLORS.get(pred_class, "#2563EB")
    
    uncertainty_info = compute_uncertainty_metrics(probs)

    # =========================================================================
    # 5. PREDICTION CARD (MODEL PREDICTION, NOT AI DIAGNOSIS)
    # =========================================================================
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    
    suitability_flag = ""
    if val_result.status == ValidationStatus.WARNING:
        suitability_flag = '<div style="margin-top: 0.4rem;"><span class="badge-chip badge-chip-warning">Caution: Input Suitability Unverified</span></div>'
        
    pred_card_html = (
        f'<div class="prediction-box" style="border: 1.5px solid {pred_color}; background: #FFFFFF; border-radius: 10px; padding: 1.25rem 1.4rem;">'
        '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1.0rem;">'
        '<div>'
        '<div style="font-size: 0.74rem; text-transform: uppercase; font-weight: 800; color: #64748B; letter-spacing: 0.06em;">'
        'MODEL PREDICTION'
        '</div>'
        f'<div style="font-size: 2.05rem; font-weight: 800; color: {pred_color}; margin: 0.2rem 0; line-height: 1.15;">'
        f'{pred_class}'
        '</div>'
        f'<div style="font-size: 0.84rem; color: #334155;">'
        f'Evaluated by <strong>{selected_model_name}</strong> • Latency: {latency:.1f} ms'
        '</div>'
        f'{suitability_flag}'
        '</div>'
        '<div style="display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap;">'
        '<div style="text-align: right; background: #F8FAFC; padding: 0.75rem 1.15rem; border-radius: 8px; border: 1px solid #E2E8F0;" title="Model Confidence: Softmax probability of the top predicted class, representing computational model certainty rather than clinical diagnosis.">'
        '<div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 750; color: #64748B; letter-spacing: 0.04em; cursor: help;">Model Confidence ℹ</div>'
        f'<div style="font-size: 1.85rem; font-weight: 800; color: {pred_color}; line-height: 1.2;">{confidence:.1%}</div>'
        '</div>'
        '<div style="text-align: right; background: #F8FAFC; padding: 0.75rem 1.15rem; border-radius: 8px; border: 1px solid #E2E8F0;">'
        '<div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 750; color: #64748B; letter-spacing: 0.04em;">Margin (Δp)</div>'
        f'<div style="font-size: 1.40rem; font-weight: 800; color: #0F172A; line-height: 1.3;">{uncertainty_info["prediction_margin"]*100:.1f}%</div>'
        '</div>'
        '<div style="text-align: right; background: #F8FAFC; padding: 0.75rem 1.15rem; border-radius: 8px; border: 1px solid #E2E8F0;">'
        '<div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 750; color: #64748B; letter-spacing: 0.04em;">Uncertainty</div>'
        f'<div style="font-size: 1.25rem; font-weight: 800; color: #0F172A; margin-top: 0.2rem;">{uncertainty_info["uncertainty_level"]}</div>'
        f'<div style="font-size: 0.70rem; color: #64748B;">H: {uncertainty_info["normalized_entropy"]:.2f}</div>'
        '</div>'
        '</div>'
        '</div>'
        '</div>'
    )
    render_html(pred_card_html)
    
    st.markdown(
        '<div style="font-size: 0.80rem; color: #64748B; margin-top: 0.45rem; font-style: italic;">'
        'These values represent computational model outputs and statistical uncertainty indicators, not clinical diagnostic certainty.'
        '</div>',
        unsafe_allow_html=True
    )
    
    # =========================================================================
    # 6. SOFTMAX PROBABILITY DISTRIBUTION (All 4 Classes Visible)
    # =========================================================================
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Softmax Probability Distribution")
    
    for c_name in CLASS_NAMES:
        c_prob = probs[c_name]
        c_pct = c_prob * 100.0
        c_color = CLASS_COLORS[c_name]
        is_winner = (c_name == pred_class)
        marker = f' <span style="font-size: 0.70rem; font-weight: 750; color: {c_color}; background: {c_color}18; padding: 2px 7px; border-radius: 4px;">PREDICTED</span>' if is_winner else ""
        
        bar_html = (
            '<div style="margin-bottom: 0.65rem;">'
            '<div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.86rem; margin-bottom: 0.2rem;">'
            f'<span><strong style="color: #0F172A;">{c_name}</strong>{marker}</span>'
            f'<span style="font-weight: 750; font-family: monospace; color: #0F172A;">{c_pct:.2f}%</span>'
            '</div>'
            '<div style="height: 9px; background: #E2E8F0; border-radius: 5px; overflow: hidden;">'
            f'<div style="width: {max(c_pct, 0.4):.2f}%; height: 100%; background: {c_color}; border-radius: 5px;"></div>'
            '</div>'
            '</div>'
        )
        render_html(bar_html)

    # =========================================================================
    # 7. MODEL AGREEMENT ENGINE (Live Cross-Architecture Consensus)
    # =========================================================================
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("### 05 Cross-Architecture Model Agreement")
    st.caption("Live concurrent evaluation across all three trained models to detect architectural consensus or decision boundary divergence.")
    
    # Run inference across all 3 models using existing cached checkpoints
    agreement_results = {}
    with st.spinner("Evaluating scan across MobileNetV2, EfficientNet-B0, and ResNet18..."):
        for m_name in ["MobileNetV2", "EfficientNet-B0", "ResNet18"]:
            m_obj, _ = load_cached_model(m_name)
            if m_obj is not None:
                agreement_results[m_name] = run_model_inference(m_obj, tensor_input)
            else:
                agreement_results[m_name] = None

    # Check consensus
    predicted_classes = [res["predicted_class"] for res in agreement_results.values() if res is not None]
    unique_classes = set(predicted_classes)
    all_agree = (len(unique_classes) == 1 and len(predicted_classes) == 3)
    
    if all_agree:
        consensus_badge_html = (
            '<div class="agreement-badge-agree">'
            '<span>✓</span> <strong>3 / 3 ARCHITECTURES AGREE</strong>'
            '</div>'
        )
        consensus_text = f"All three distinct model backbones independently predicted <strong>{predicted_classes[0]}</strong>."
    else:
        consensus_badge_html = (
            '<div class="agreement-badge-disagree">'
            '<span>⚠</span> <strong>MODEL DISAGREEMENT DETECTED</strong>'
            '</div>'
        )
        consensus_text = "Different architectures produced different predictions for this scan, reflecting representational divergence near the clinical stage decision boundary."

    agreement_container_html = (
        '<div class="agreement-card-container">'
        '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 0.85rem;">'
        '<div style="font-weight: 750; font-size: 0.96rem; color: #0F172A;">Consensus Diagnostics</div>'
        f'{consensus_badge_html}'
        '</div>'
        f'<div style="font-size: 0.84rem; color: #334155; margin-bottom: 1.0rem; line-height: 1.5;">{consensus_text}</div>'
        '</div>'
    )
    render_html(agreement_container_html)
    
    # 3 Individual Model Outcome Columns
    col_ag1, col_ag2, col_ag3 = st.columns(3)
    for col, m_name in zip([col_ag1, col_ag2, col_ag3], ["MobileNetV2", "EfficientNet-B0", "ResNet18"]):
        res = agreement_results.get(m_name)
        with col:
            if res:
                c_pred = res["predicted_class"]
                c_conf = res["confidence"]
                c_color = CLASS_COLORS.get(c_pred, "#2563EB")
                m_card_html = (
                    '<div class="med-card" style="padding: 0.95rem; background: #FFFFFF; border-top: 3px solid #0F4C81; height: 100%;">'
                    f'<div style="font-size: 0.80rem; font-weight: 750; color: #64748B; text-transform: uppercase;">{m_name}</div>'
                    f'<div style="font-size: 1.15rem; font-weight: 800; color: {c_color}; margin: 0.3rem 0;">{c_pred}</div>'
                    f'<div style="font-size: 0.82rem; color: #334155;">Confidence: <strong>{c_conf:.1%}</strong></div>'
                    f'<div style="font-size: 0.74rem; color: #64748B; margin-top: 0.2rem;">Latency: {res["latency_ms"]:.1f} ms</div>'
                    '</div>'
                )
                render_html(m_card_html)
            else:
                st.warning(f"{m_name} output unavailable")

    # =========================================================================
    # 8. EXPLAINABILITY (Grad-CAM Visual Attribution)
    # =========================================================================
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown(
        '### 06 Grad-CAM Explainability '
        '<span title="Grad-CAM (Gradient-weighted Class Activation Mapping): Visualizes which image regions contributed most strongly to the model\'s prediction." style="font-size: 0.78rem; font-weight: normal; color: #2563EB; background: #EFF6FF; border: 1px solid #BFDBFE; padding: 2px 8px; border-radius: 4px; cursor: help; vertical-align: middle;">ℹ What is Grad-CAM?</span>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="font-size: 0.86rem; color: #334155; margin-bottom: 0.75rem;">'
        'Highlighted regions indicate image areas contributing to the model prediction. '
        'Grad-CAM computes gradients of the target class score with respect to feature activation maps.'
        '</div>',
        unsafe_allow_html=True
    )
    
    with st.spinner("Computing gradient-weighted class activation mapping..."):
        try:
            cam_np, overlay_img, target_layer_desc = generate_gradcam(
                model=primary_model,
                model_name=selected_model_name,
                input_tensor=tensor_input,
                display_img=display_img,
                target_class=pred_idx
            )
            
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.markdown(
                    '<div class="med-card" style="padding: 0.75rem; text-align: center; margin-bottom: 0.35rem; background: #FFFFFF;">'
                    '<div style="font-size: 0.76rem; font-weight: 800; color: #0F4C81; text-transform: uppercase; letter-spacing: 0.05em;">ORIGINAL MRI</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.image(display_img, use_container_width=True, caption="The original brain MRI used for analysis.")
            with col_p2:
                st.markdown(
                    '<div class="med-card" style="padding: 0.75rem; text-align: center; margin-bottom: 0.35rem; background: #FFFFFF;">'
                    '<div style="font-size: 0.76rem; font-weight: 800; color: #0F4C81; text-transform: uppercase; letter-spacing: 0.05em;">GRAD-CAM HEATMAP</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                colored_cam = (cm.jet(cam_np)[:, :, :3] * 255.0).astype(np.uint8)
                st.image(colored_cam, use_container_width=True, caption="A color map showing the relative strength of image-region contributions to the model prediction.")
            with col_p3:
                st.markdown(
                    '<div class="med-card" style="padding: 0.75rem; text-align: center; margin-bottom: 0.35rem; background: #FFFFFF;">'
                    '<div style="font-size: 0.76rem; font-weight: 800; color: #0F4C81; text-transform: uppercase; letter-spacing: 0.05em;">GRAD-CAM OVERLAY</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.image(overlay_img, use_container_width=True, caption=f"The attribution map overlaid on the MRI to make contributing regions easier to locate. (Layer: {target_layer_desc})")
                
            # Comprehensive User Guide for Normal Users
            render_gradcam_user_guide()
            render_attribution_disclaimer()
            
            with st.expander("Technical Telemetry & Target Layer Hooks"):
                st.json({
                    "model": selected_model_name,
                    "target_layer": target_layer_desc,
                    "input_tensor_shape": list(tensor_input.shape),
                    "predicted_stage": pred_class,
                    "model_confidence": f"{confidence:.4f}",
                    "normalized_entropy": uncertainty_info["normalized_entropy"],
                    "prediction_margin": uncertainty_info["prediction_margin"],
                    "uncertainty_level": uncertainty_info["uncertainty_level"],
                    "probabilities": probs,
                    "device": str(device)
                })
                
        except Exception as cam_err:
            st.warning(f"Grad-CAM generation issue: {cam_err}")

    # Export Research Summary Button
    st.markdown("---")
    summary_report_text = f"""================================================================================
ACADEMIC RESEARCH PREDICTION SUMMARY
Project: Explainable Deep Learning-Based Multi-Stage Alzheimer's Disease Detection
System: Alzheimer's Disease Detection & Explainability System
================================================================================

Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Input File: {input_source_name}
Input Dimensions: {meta['original_width']} x {meta['original_height']} px

INPUT VALIDATION:
Validation Status: {val_result.status.value}
Headline: {val_result.headline}
Advisory: {val_result.message}

MODEL PREDICTION:
Selected Architecture: {selected_model_name}
Target Layer: {model_cfg['target_layer_desc']}
Predicted Class: {pred_class}
Model Confidence: {confidence:.2%}
Uncertainty Level: {uncertainty_info['uncertainty_level']} (Normalized Entropy: {uncertainty_info['normalized_entropy']:.3f})
Prediction Margin: {uncertainty_info['prediction_margin']*100:.2f}%
Inference Latency: {latency:.2f} ms

SOFTMAX PROBABILITIES:
- Non-Demented:         {probs['Non-Demented']:.4f} ({probs['Non-Demented']*100:.2f}%)
- Very Mild Demented:   {probs['Very Mild Demented']:.4f} ({probs['Very Mild Demented']*100:.2f}%)
- Mild Demented:        {probs['Mild Demented']:.4f} ({probs['Mild Demented']*100:.2f}%)
- Moderate Demented:    {probs['Moderate Demented']:.4f} ({probs['Moderate Demented']*100:.2f}%)

CROSS-ARCHITECTURE AGREEMENT:
Consensus Status: {"3/3 Models Agree" if all_agree else "Model Disagreement Detected"}
Unique Predicted Stages: {list(unique_classes)}

GRAD-CAM EXPLANATION:
Method: Gradient-weighted Class Activation Mapping
Target Layer: {model_cfg['target_layer_desc']}
Interpretation: Highlighted regions indicate areas that contributed more strongly to the model's prediction.

ACADEMIC & RESEARCH DISCLAIMER:
This document is generated by an academic engineering research prototype for educational 
and algorithmic benchmarking purposes. Grad-CAM visualizes receptive field gradient 
attributions and does NOT constitute clinical or biological proof of hippocampal atrophy, 
ventricular dilation, cortical thinning, anatomical biomarkers, or disease causality.
This system has NOT been clinically validated and MUST NOT be used for clinical diagnosis,
screening, or patient management decisions.
================================================================================
"""
    st.download_button(
        label="Download Research Prediction Summary (.txt)",
        data=summary_report_text,
        file_name=f"research_prediction_summary_{input_source_name.replace('.jpg', '').replace('.png', '').replace('.webp', '')}.txt",
        mime="text/plain"
    )
    
    render_research_disclaimer()
