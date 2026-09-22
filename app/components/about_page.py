"""
Page 8: About & Academic Disclaimer.
Provides structured project documentation across:
- Project Overview & Objectives
- Deep Learning Technology Stack
- Evaluated Neural Architectures
- Dataset Composition & Quarantined Partitions
- Research Scope
- Technical Limitations & Sample Size Caveat
- Formal Academic Disclaimer
"""

import streamlit as st
from app.utils.ui_helpers import render_html


def render_about_page():
    # Page Header
    header_html = (
        '<div style="margin-bottom: 1.25rem;">'
        '<div class="workflow-badge" style="margin-bottom: 0.5rem;">'
        'PROJECT DOCUMENTATION & ETHICAL GOVERNANCE'
        '</div>'
        '<h2 style="margin: 0.15rem 0 0.35rem 0; font-size: 1.65rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">'
        'About the Research Platform'
        '</h2>'
        '<div style="font-size: 0.88rem; color: #64748B; line-height: 1.5;">'
        'System specifications, deep learning architecture configurations, dataset governance, '
        'and academic research disclaimers.'
        '</div>'
        '</div>'
    )
    render_html(header_html)
    
    # 1. Project Overview
    st.markdown("### Project Overview")
    proj_html = (
        '<div class="med-card" style="background: #FFFFFF;">'
        '<div style="font-weight: 700; font-size: 1.05rem; color: #0F4C81; margin-bottom: 0.4rem;">'
        "Explainable Deep Learning-Based Multi-Stage Alzheimer’s Disease Detection Using Brain MRI"
        '</div>'
        '<p style="color: #334155; font-size: 0.90rem; line-height: 1.65; margin: 0;">'
        'An academic research software platform designed to demonstrate multi-stage classification '
        'of neurodegenerative stages from axial structural brain MRI scans. The platform integrates '
        'transfer learning across three distinct convolutional architectures, post-hoc statistical calibration, '
        'visual explainability via Grad-CAM, cross-model consensus analysis, failure mode profiling, '
        'and robust 5-stage input validation gates.'
        '</p>'
        '</div>'
    )
    render_html(proj_html)
    
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 2. Technology Stack & 3. Models
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("### Technology Stack")
        tech_html = (
            '<div class="med-card" style="background: #FFFFFF;">'
            '<div style="font-size: 0.86rem; color: #334155; line-height: 1.85;">'
            '• <strong>Deep Learning Framework:</strong> PyTorch 2.6.0 with Torchvision 0.21.0<br>'
            '• <strong>Acceleration Environment:</strong> CUDA 12.4 on NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)<br>'
            '• <strong>Web Application:</strong> Streamlit 1.64.0 (Light Medical Research Design System)<br>'
            '• <strong>Image Processing:</strong> Pillow (PIL), NumPy, SciPy<br>'
            '• <strong>Calibration & Diagnostics:</strong> Temperature Scaling, Expected Calibration Error (ECE)<br>'
            '• <strong>Runtime Environment:</strong> Python 3.12 (64-bit Windows)'
            '</div>'
            '</div>'
        )
        render_html(tech_html)
        
    with col_t2:
        st.markdown("### Evaluated Architectures")
        model_html = (
            '<div class="med-card" style="background: #FFFFFF;">'
            '<div style="font-size: 0.86rem; color: #334155; line-height: 1.85;">'
            '• <strong>MobileNetV2:</strong> 2.23M params &bull; Inverted residual bottlenecks &bull; 2.93 ms latency<br>'
            '• <strong>EfficientNet-B0:</strong> 4.01M params &bull; Compound depth/width scaling &bull; 2.20 ms latency<br>'
            '• <strong>ResNet-18:</strong> 11.18M params &bull; Residual identity skip connections &bull; 1.77 ms latency<br>'
            '• <strong>Optimization:</strong> AdamW (lr=1e-4, wd=1e-4) with FP16 Automatic Mixed Precision<br>'
            '• <strong>Explainability:</strong> Grad-CAM on final convolutional layer<br>'
            '• <strong>Calibration:</strong> Validation-fitted scalar temperature parameter'
            '</div>'
            '</div>'
        )
        render_html(model_html)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 4. Dataset Governance & Splits
    st.markdown("### Dataset Governance & Quarantined Partitions")
    dataset_html = (
        '<div class="med-card" style="background: #FFFFFF;">'
        '<div style="font-size: 0.88rem; color: #334155; line-height: 1.75;">'
        'Audited benchmark cohort of exactly <strong>6,400 unique axial brain MRI scans</strong> (128 &times; 128 px, single-channel):<br>'
        '• <strong>Training Split (70%):</strong> 4,480 images (Non: 2,240 | Very Mild: 1,568 | Mild: 627 | Moderate: 45) with 24.89&times; class-weighted loss.<br>'
        '• <strong>Validation Split (15%):</strong> 960 images (Non: 480 | Very Mild: 336 | Mild: 134 | Moderate: 10) for checkpoint selection and temperature tuning.<br>'
        '• <strong>Held-Out Test Split (15%):</strong> 960 images (Non: 480 | Very Mild: 336 | Mild: 135 | Moderate: 9) quarantined with SHA-256 verified 0.00% leakage.'
        '</div>'
        '</div>'
    )
    render_html(dataset_html)
    
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 5. Research Scope & 6. Limitations
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("### Research Scope")
        scope_html = (
            '<div class="med-card" style="background: #FFFFFF;">'
            '<p style="font-size: 0.86rem; color: #334155; line-height: 1.65; margin: 0;">'
            'Investigates the comparative diagnostic utility, parameter efficiency, calibration reliability, '
            'and visual attribution patterns of transfer-learned convolutional architectures on 2D axial '
            'structural neuroimaging slices under strict class imbalance mitigation.'
            '</p>'
            '</div>'
        )
        render_html(scope_html)
        
    with col_s2:
        st.markdown("### Technical Limitations")
        limit_html = (
            '<div class="med-card" style="background: #FFFFFF;">'
            '<p style="font-size: 0.86rem; color: #334155; line-height: 1.65; margin: 0;">'
            'Evaluates 2D slice crops rather than 3D volumetric sequences. Lacks patient identifiers and longitudinal tracking. '
            'The Moderate Demented class has test support n = 9, introducing statistical variance. '
            'Visual heatmaps represent mathematical filter activations, not verified physiological biomarkers.'
            '</p>'
            '</div>'
        )
        render_html(limit_html)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 7. Formal Academic Disclaimer
    st.markdown("### Academic Compliance & Ethical Disclaimer")
    disclaimer_html = (
        '<div class="disclaimer-alert" style="padding: 1.25rem 1.5rem;">'
        '<div class="disclaimer-alert-title" style="font-size: 0.95rem; font-weight: 800; letter-spacing: 0.02em;">'
        'ACADEMIC RESEARCH NOTICE — NOT A MEDICAL DEVICE'
        '</div>'
        '<p style="font-size: 0.88rem; line-height: 1.65; margin: 0.45rem 0 0 0; color: #78350F;">'
        'This software application is an academic engineering research prototype developed for algorithm benchmarking, '
        'interpretability demonstration, and pedagogical presentation. It is <strong>NOT a cleared medical device</strong> and has not been evaluated, '
        'approved, or certified by the US FDA, European CE Mark, Indian CDSCO, or any regulatory health agency.<br><br>'
        'This software must <strong>NEVER be used for clinical diagnosis, patient triage, prognosis, medication guidance, or medical decision-making</strong>. '
        'All outputs represent algorithmic estimations on 2D axial MRI crops and must be verified by licensed healthcare practitioners.'
        '</p>'
        '</div>'
    )
    render_html(disclaimer_html)
