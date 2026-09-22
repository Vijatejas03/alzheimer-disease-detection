"""
Page 1: Overview / Research Platform Dashboard.
Provides high-level system summary, interactive research pipeline,
verified KPI metrics, architecture snapshots, and dementia stage spectrum.
"""

import streamlit as st
from app.utils.ui_helpers import render_html
from app.components.disclaimer import render_research_disclaimer


def render_home_page():
    # =========================================================================
    # 1. HERO SECTION
    # =========================================================================
    hero_html = (
        '<div class="hero-container">'
        '<div class="hero-badge-row">'
        '<span class="badge-chip badge-chip-info">ACADEMIC RESEARCH PROTOTYPE</span>'
        '<span class="badge-chip" style="background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1;">DATASET: 6,400 SCANS</span>'
        '<span class="badge-chip" style="background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1;">3 CANDIDATE MODELS</span>'
        '</div>'
        '<div class="hero-title">Explainable Alzheimer\'s MRI Analysis</div>'
        '<div class="hero-subtitle">'
        'An academic deep-learning research platform for four-stage dementia classification, '
        'visual attribution and model evaluation using structural brain MRI scans.'
        '</div>'
        '</div>'
    )
    render_html(hero_html)

    # Hero Action Buttons
    col_btn1, col_btn2, _ = st.columns([1.1, 1.3, 2.6])
    with col_btn1:
        if st.button("⊕  Analyze MRI Scan", use_container_width=True, type="primary"):
            st.session_state["nav_page"] = "MRI Analysis"
            st.rerun()
    with col_btn2:
        if st.button("⊞  Explore Model Benchmarks", use_container_width=True):
            st.session_state["nav_page"] = "Model Comparison"
            st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 2. VERIFIED RESEARCH PLATFORM KPIs
    # =========================================================================
    st.markdown("### Research Platform Scope & Key Metrics")
    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
    
    with kpi_c1:
        render_html(
            '<div class="med-metric-card">'
            '<div class="med-metric-val">6,400</div>'
            '<div class="med-metric-lbl">Dataset Images</div>'
            '</div>'
        )
    with kpi_c2:
        render_html(
            '<div class="med-metric-card">'
            '<div class="med-metric-val">3</div>'
            '<div class="med-metric-lbl">Models Evaluated</div>'
            '</div>'
        )
    with kpi_c3:
        render_html(
            '<div class="med-metric-card">'
            '<div class="med-metric-val">960</div>'
            '<div class="med-metric-lbl">Held-Out Test Scans</div>'
            '</div>'
        )
    with kpi_c4:
        render_html(
            '<div class="med-metric-card">'
            '<div class="med-metric-val">4</div>'
            '<div class="med-metric-lbl">Classification Stages</div>'
            '</div>'
        )
    with kpi_c5:
        render_html(
            '<div class="med-metric-card">'
            '<div class="med-metric-val">35 / 35</div>'
            '<div class="med-metric-lbl">Automated Tests Passing</div>'
            '</div>'
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 3. RESEARCH PIPELINE WORKFLOW (Step-by-Step Visualization)
    # =========================================================================
    st.markdown("### End-to-End Research Pipeline")
    st.caption("Computational sequence from raw image ingestion to interpretability and controlled robustness analysis.")
    
    steps = [
        ("01", "MRI Input", "Grayscale axial slice"),
        ("02", "Quality Gate", "5-stage safety checks"),
        ("03", "Preprocessing", "Standardize to 224x224"),
        ("04", "Inference", "FP16 CUDA Forward"),
        ("05", "Prediction", "4-class probability"),
        ("06", "Uncertainty", "Entropy & margin"),
        ("07", "Grad-CAM", "Attribution heatmap"),
        ("08", "Evaluation", "Benchmarking & audits")
    ]
    
    step_cards_html = ['<div class="pipeline-flow-container">']
    for num, title, desc in steps:
        step_cards_html.append(
            '<div class="pipeline-step-card">'
            f'<div class="pipeline-step-num">{num}</div>'
            f'<div class="pipeline-step-title">{title}</div>'
            f'<div style="font-size: 0.72rem; color: #64748B; margin-top: 0.2rem;">{desc}</div>'
            '</div>'
        )
    step_cards_html.append('</div>')
    render_html("".join(step_cards_html))

    # Full Research Workflow Ribbon Callout
    workflow_ribbon_html = (
        '<div class="med-card" style="padding: 0.90rem 1.15rem; background: #FFFFFF; border-left: 4px solid #2563EB; margin-bottom: 1.5rem;">'
        '<div style="font-size: 0.76rem; font-weight: 800; color: #0F4C81; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem;">'
        'Complete Research Workflow Architecture'
        '</div>'
        '<div style="font-size: 0.84rem; color: #334155; line-height: 1.6;">'
        '<strong>MRI</strong> → <strong>Input Quality Validation</strong> → <strong>Preprocessing</strong> → '
        '<strong>Deep Learning Inference</strong> → <strong>Four-Class Prediction</strong> → <strong>Confidence + Uncertainty</strong> → '
        '<strong>Model Agreement</strong> → <strong>Grad-CAM Explainability</strong> → <strong>Calibration</strong> → '
        '<strong>Error Analysis</strong> → <strong>Robustness Evaluation</strong>'
        '</div>'
        '</div>'
    )
    render_html(workflow_ribbon_html)

    # =========================================================================
    # 4. MODEL SNAPSHOT: Three Architectural Backbones
    # =========================================================================
    st.markdown("### Candidate Model Snapshots")
    st.caption("Empirical measurements across 960 held-out test scans (zero data leakage). No overall ranking is asserted.")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        m1_html = (
            '<div class="model-snapshot-box" style="border-top-color: #0284C7;">'
            '<div class="model-snapshot-title">MobileNetV2</div>'
            '<div class="model-snapshot-tag">Inverted Residuals • Lightweight</div>'
            '<div style="font-size: 0.86rem; color: #334155; line-height: 1.85;">'
            '• <strong>Parameters:</strong> 2,228,996<br>'
            '• <strong>Test Accuracy:</strong> 93.13%<br>'
            '• <strong>Macro F1:</strong> 0.9434<br>'
            '• <strong>ROC-AUC:</strong> 0.9936<br>'
            '• <strong>Latency:</strong> 2.93 ms'
            '</div>'
            '</div>'
        )
        render_html(m1_html)
        
    with col_m2:
        m2_html = (
            '<div class="model-snapshot-box" style="border-top-color: #2563EB;">'
            '<div class="model-snapshot-title">EfficientNet-B0</div>'
            '<div class="model-snapshot-tag">Compound Scaling • MBConv + SE</div>'
            '<div style="font-size: 0.86rem; color: #334155; line-height: 1.85;">'
            '• <strong>Parameters:</strong> 4,012,672<br>'
            '• <strong>Test Accuracy:</strong> 98.23%<br>'
            '• <strong>Macro F1:</strong> 0.9876<br>'
            '• <strong>ROC-AUC:</strong> 0.9989<br>'
            '• <strong>Latency:</strong> 2.20 ms'
            '</div>'
            '</div>'
        )
        render_html(m2_html)
        
    with col_m3:
        m3_html = (
            '<div class="model-snapshot-box" style="border-top-color: #0F4C81;">'
            '<div class="model-snapshot-title">ResNet-18</div>'
            '<div class="model-snapshot-tag">Residual Connections • Dense Conv</div>'
            '<div style="font-size: 0.86rem; color: #334155; line-height: 1.85;">'
            '• <strong>Parameters:</strong> 11,178,564<br>'
            '• <strong>Test Accuracy:</strong> 98.44%<br>'
            '• <strong>Macro F1:</strong> 0.9860<br>'
            '• <strong>ROC-AUC:</strong> 0.9991<br>'
            '• <strong>Latency:</strong> 1.77 ms'
            '</div>'
            '</div>'
        )
        render_html(m3_html)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 5. FOUR-STAGE DEMENTIA SPECTRUM
    # =========================================================================
    st.markdown("### Four-Stage Classification Spectrum")
    
    spec_c1, spec_c2, spec_c3, spec_c4 = st.columns(4)
    with spec_c1:
        render_html(
            '<div class="med-card" style="border-left: 4px solid #2563EB; height: 100%;">'
            '<div style="font-weight: 750; color: #2563EB; font-size: 0.95rem; margin-bottom: 0.25rem;">Non-Demented</div>'
            '<div style="font-size: 0.74rem; font-weight: 700; color: #64748B; margin-bottom: 0.5rem;">CONTROL COHORT (50.0%)</div>'
            '<p style="font-size: 0.82rem; color: #334155; line-height: 1.5; margin: 0;">'
            'Structurally normal ventricles and cortical volume; serves as healthy cognitive baseline.'
            '</p>'
            '</div>'
        )
    with spec_c2:
        render_html(
            '<div class="med-card" style="border-left: 4px solid #6366F1; height: 100%;">'
            '<div style="font-weight: 750; color: #6366F1; font-size: 0.95rem; margin-bottom: 0.25rem;">Very Mild</div>'
            '<div style="font-size: 0.74rem; font-weight: 700; color: #64748B; margin-bottom: 0.5rem;">EARLY STAGE (35.0%)</div>'
            '<p style="font-size: 0.82rem; color: #334155; line-height: 1.5; margin: 0;">'
            'Subtle sulcal enlargement and border-case boundaries; most frequent site of inter-model confusion.'
            '</p>'
            '</div>'
        )
    with spec_c3:
        render_html(
            '<div class="med-card" style="border-left: 4px solid #D97706; height: 100%;">'
            '<div style="font-weight: 750; color: #D97706; font-size: 0.95rem; margin-bottom: 0.25rem;">Mild Demented</div>'
            '<div style="font-size: 0.74rem; font-weight: 700; color: #64748B; margin-bottom: 0.5rem;">INTERMEDIATE (14.0%)</div>'
            '<p style="font-size: 0.82rem; color: #334155; line-height: 1.5; margin: 0;">'
            'Clearer ventricular expansion and cortical space widening; high recognition precision across models.'
            '</p>'
            '</div>'
        )
    with spec_c4:
        render_html(
            '<div class="med-card" style="border-left: 4px solid #DC2626; height: 100%;">'
            '<div style="font-weight: 750; color: #DC2626; font-size: 0.95rem; margin-bottom: 0.25rem;">Moderate Demented</div>'
            '<div style="font-size: 0.74rem; font-weight: 700; color: #DC2626; margin-bottom: 0.5rem;">ADVANCED (1.0%, n=9*)</div>'
            '<p style="font-size: 0.82rem; color: #334155; line-height: 1.5; margin: 0;">'
            'Severe structural degeneration. <em>*Test support n=9; conclusions carry substantial statistical uncertainty.</em>'
            '</p>'
            '</div>'
        )

    # Restrained Research Notice
    render_research_disclaimer()
