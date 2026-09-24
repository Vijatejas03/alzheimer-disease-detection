"""
Page 1: Overview / Action-First Research Platform Dashboard.
Redesigned with futuristic 3D medical-AI visual, action-first CTAs,
clickable quick actions, visual workflow, project snapshot, and candidate models.
"""

import streamlit as st
from app.utils.ui_helpers import render_html
from app.utils.navigation import navigate_to
from app.components.brain_3d_visual import get_3d_brain_svg
from app.components.disclaimer import render_research_disclaimer


def render_home_page():
    # =========================================================================
    # 1. 3D HERO SECTION (ACTION-FIRST)
    # =========================================================================
    hero_col_left, hero_col_right = st.columns([1.25, 1.0], gap="large")

    with hero_col_left:
        hero_left_html = (
            '<div class="hero-3d-left">'
            '<div class="hero-badge-pill">'
            '<span class="hero-badge-dot"></span>'
            'ACADEMIC AI RESEARCH PLATFORM'
            '</div>'
            '<h1 class="hero-3d-title">'
            'Explainable Alzheimer’s<br>'
            '<span class="hero-gradient-text">MRI Analysis</span>'
            '</h1>'
            '<p class="hero-3d-desc">'
            'Analyze structural brain MRI scans using deep learning and explore '
            'model predictions with visual explanations.'
            '</p>'
            '<div class="hero-metric-tag-row">'
            '<span class="hero-tag">⚡ 3 Neural Networks</span>'
            '<span class="hero-tag">🔬 Grad-CAM Saliency</span>'
            '<span class="hero-tag">🛡️ Input Quality Gate</span>'
            '</div>'
            '</div>'
        )
        render_html(hero_left_html)

        # Primary Action CTAs - Responsive functional Streamlit buttons
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        cta_col1, cta_col2 = st.columns([1.15, 1.0], gap="small")
        with cta_col1:
            st.button(
                "🧠  ANALYZE MRI SCAN",
                key="hero_btn_analyze",
                use_container_width=True,
                type="primary",
                on_click=navigate_to,
                args=("MRI Analysis",)
            )
        with cta_col2:
            st.button(
                "📊  EXPLORE RESULTS",
                key="hero_btn_results",
                use_container_width=True,
                on_click=navigate_to,
                args=("Evaluation",)
            )

    with hero_col_right:
        render_html(get_3d_brain_svg())

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 2. QUICK ACTION CARDS (4 Interactive Modules)
    # =========================================================================
    st.markdown('<div class="section-title-3d">Quick Actions</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub-3d">Direct access to primary diagnostic, explainability, and evaluation workspaces.</div>', unsafe_allow_html=True)

    qa1, qa2, qa3, qa4 = st.columns(4, gap="small")

    with qa1:
        render_html(
            '<div class="quick-action-card qa-card-blue">'
            '<div class="qa-icon-wrap">🧠</div>'
            '<div class="qa-card-title">ANALYZE MRI</div>'
            '<div class="qa-card-desc">Upload and analyze an axial brain MRI scan.</div>'
            '</div>'
        )
        st.button(
            "Launch Analyzer →",
            key="qa_btn_analyze",
            use_container_width=True,
            on_click=navigate_to,
            args=("MRI Analysis",)
        )

    with qa2:
        render_html(
            '<div class="quick-action-card qa-card-cyan">'
            '<div class="qa-icon-wrap">🔍</div>'
            '<div class="qa-card-title">EXPLAIN</div>'
            '<div class="qa-card-desc">Explore Grad-CAM model attribution heatmaps.</div>'
            '</div>'
        )
        st.button(
            "View Attribution →",
            key="qa_btn_explain",
            use_container_width=True,
            on_click=navigate_to,
            args=("Explainability",)
        )

    with qa3:
        render_html(
            '<div class="quick-action-card qa-card-purple">'
            '<div class="qa-icon-wrap">📊</div>'
            '<div class="qa-card-title">COMPARE</div>'
            '<div class="qa-card-desc">Compare the three candidate neural models.</div>'
            '</div>'
        )
        st.button(
            "Compare Models →",
            key="qa_btn_compare",
            use_container_width=True,
            on_click=navigate_to,
            args=("Model Comparison",)
        )

    with qa4:
        render_html(
            '<div class="quick-action-card qa-card-emerald">'
            '<div class="qa-icon-wrap">📈</div>'
            '<div class="qa-card-title">EVALUATE</div>'
            '<div class="qa-card-desc">Explore held-out test set benchmark results.</div>'
            '</div>'
        )
        st.button(
            "Audit Metrics →",
            key="qa_btn_evaluate",
            use_container_width=True,
            on_click=navigate_to,
            args=("Evaluation",)
        )

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 3. HOW IT WORKS: 5-STEP VISUAL WORKFLOW
    # =========================================================================
    st.markdown('<div class="section-title-3d">How It Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub-3d">End-to-end computational pipeline from raw neuroimaging to visual attribution.</div>', unsafe_allow_html=True)

    workflow_steps = [
        ("01", "MRI INPUT", "Upload axial T1 brain slice", "🧠"),
        ("02", "PREPROCESSING", "Standardize to 224×224×3", "⚙️"),
        ("03", "DEEP LEARNING", "Forward inference pass", "⚡"),
        ("04", "PREDICTION", "4-stage probability score", "🎯"),
        ("05", "GRAD-CAM", "Visual attribution heatmap", "🔍")
    ]

    wf_cols = st.columns(5, gap="small")
    for idx, col in enumerate(wf_cols):
        num, title, desc, icon = workflow_steps[idx]
        with col:
            arrow = '<div class="wf-connector">➔</div>' if idx < 4 else ''
            render_html(
                f'<div class="workflow-card-3d">'
                f'<div class="wf-step-badge">{num}</div>'
                f'<div class="wf-step-icon">{icon}</div>'
                f'<div class="wf-step-title">{title}</div>'
                f'<div class="wf-step-desc">{desc}</div>'
                f'{arrow}'
                f'</div>'
            )

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 4. PROJECT SNAPSHOT: VERIFIED METRICS
    # =========================================================================
    st.markdown('<div class="section-title-3d">Project Snapshot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub-3d">Strictly verified empirical cohort and experimental specifications.</div>', unsafe_allow_html=True)

    snap1, snap2, snap3, snap4 = st.columns(4, gap="small")
    with snap1:
        render_html(
            '<div class="snapshot-card-3d">'
            '<div class="snapshot-val">6,400</div>'
            '<div class="snapshot-lbl">MRI SCANS</div>'
            '<div class="snapshot-sub">128×128 Axial T1 Cohort</div>'
            '</div>'
        )
    with snap2:
        render_html(
            '<div class="snapshot-card-3d">'
            '<div class="snapshot-val">3</div>'
            '<div class="snapshot-lbl">CANDIDATE MODELS</div>'
            '<div class="snapshot-sub">MobileNetV2 • EfficientNet • ResNet</div>'
            '</div>'
        )
    with snap3:
        render_html(
            '<div class="snapshot-card-3d">'
            '<div class="snapshot-val">4</div>'
            '<div class="snapshot-lbl">CLASSES</div>'
            '<div class="snapshot-sub">Non • Very Mild • Mild • Moderate</div>'
            '</div>'
        )
    with snap4:
        render_html(
            '<div class="snapshot-card-3d">'
            '<div class="snapshot-val">960</div>'
            '<div class="snapshot-lbl">HELD-OUT TEST SCANS</div>'
            '<div class="snapshot-sub">0.00% Hash Overlap (Quarantined)</div>'
            '</div>'
        )

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 5. CANDIDATE MODEL CARDS
    # =========================================================================
    st.markdown('<div class="section-title-3d">Candidate Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub-3d">Three distinct architectural paradigms benchmarked on the identical held-out test split.</div>', unsafe_allow_html=True)

    mod_col1, mod_col2, mod_col3 = st.columns(3, gap="medium")

    with mod_col1:
        render_html(
            '<div class="model-card-3d card-mobilenet">'
            '<div class="model-badge-top">LIGHTWEIGHT EDGE ARCHITECTURE</div>'
            '<div class="model-card-name">MobileNetV2</div>'
            '<div class="model-card-arch">Inverted Residuals • Depthwise Separable</div>'
            '<div class="model-stats-grid">'
            '<div class="stat-box"><span class="stat-num">2.23M</span><span class="stat-lbl">Parameters</span></div>'
            '<div class="stat-box"><span class="stat-num">93.13%</span><span class="stat-lbl">Test Accuracy</span></div>'
            '<div class="stat-box"><span class="stat-num">0.9434</span><span class="stat-lbl">Macro F1</span></div>'
            '<div class="stat-box"><span class="stat-num">2.93 ms</span><span class="stat-lbl">Latency</span></div>'
            '</div>'
            '</div>'
        )
        st.button(
            "Inspect MobileNetV2 →",
            key="btn_inspect_mb",
            use_container_width=True,
            on_click=navigate_to,
            args=("Model Comparison",)
        )

    with mod_col2:
        render_html(
            '<div class="model-card-3d card-efficientnet">'
            '<div class="model-badge-top">COMPOUND SCALED ARCHITECTURE</div>'
            '<div class="model-card-name">EfficientNet-B0</div>'
            '<div class="model-card-arch">Compound Scaling • MBConv + Squeeze-Excite</div>'
            '<div class="model-stats-grid">'
            '<div class="stat-box"><span class="stat-num">4.01M</span><span class="stat-lbl">Parameters</span></div>'
            '<div class="stat-box"><span class="stat-num">98.23%</span><span class="stat-lbl">Test Accuracy</span></div>'
            '<div class="stat-box"><span class="stat-num">0.9876</span><span class="stat-lbl">Macro F1</span></div>'
            '<div class="stat-box"><span class="stat-num">2.20 ms</span><span class="stat-lbl">Latency</span></div>'
            '</div>'
            '</div>'
        )
        st.button(
            "Inspect EfficientNet-B0 →",
            key="btn_inspect_eff",
            use_container_width=True,
            on_click=navigate_to,
            args=("Model Comparison",)
        )

    with mod_col3:
        render_html(
            '<div class="model-card-3d card-resnet">'
            '<div class="model-badge-top">RESIDUAL CONNECTIONS</div>'
            '<div class="model-card-name">ResNet-18</div>'
            '<div class="model-card-arch">Identity Skip Connections • Dense Convolutions</div>'
            '<div class="model-stats-grid">'
            '<div class="stat-box"><span class="stat-num">11.18M</span><span class="stat-lbl">Parameters</span></div>'
            '<div class="stat-box"><span class="stat-num">98.44%</span><span class="stat-lbl">Test Accuracy</span></div>'
            '<div class="stat-box"><span class="stat-num">0.9860</span><span class="stat-lbl">Macro F1</span></div>'
            '<div class="stat-box"><span class="stat-num">1.77 ms</span><span class="stat-lbl">Latency</span></div>'
            '</div>'
            '</div>'
        )
        st.button(
            "Inspect ResNet-18 →",
            key="btn_inspect_res",
            use_container_width=True,
            on_click=navigate_to,
            args=("Model Comparison",)
        )

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 6. FOUR-STAGE DEMENTIA SPECTRUM
    # =========================================================================
    st.markdown('<div class="section-title-3d">Four-Stage Clinical Spectrum</div>', unsafe_allow_html=True)

    spec_c1, spec_c2, spec_c3, spec_c4 = st.columns(4, gap="small")
    with spec_c1:
        render_html(
            '<div class="spectrum-card-3d spec-non">'
            '<div class="spec-stage-title">Non-Demented</div>'
            '<div class="spec-stage-badge">CONTROL COHORT (50.0%)</div>'
            '<div class="spec-desc">Cognitively normal control cohort; serves as the baseline classification benchmark.</div>'
            '</div>'
        )
    with spec_c2:
        render_html(
            '<div class="spectrum-card-3d spec-verymild">'
            '<div class="spec-stage-title">Very Mild</div>'
            '<div class="spec-stage-badge">EARLY STAGE (35.0%)</div>'
            '<div class="spec-desc">Earliest stage of detectable cognitive decline; primary boundary for inter-model confusion.</div>'
            '</div>'
        )
    with spec_c3:
        render_html(
            '<div class="spectrum-card-3d spec-mild">'
            '<div class="spec-stage-title">Mild Demented</div>'
            '<div class="spec-stage-badge">INTERMEDIATE (14.0%)</div>'
            '<div class="spec-desc">Intermediate impairment stage; exhibits consistent cross-model recognition precision.</div>'
            '</div>'
        )
    with spec_c4:
        render_html(
            '<div class="spectrum-card-3d spec-mod">'
            '<div class="spec-stage-title">Moderate</div>'
            '<div class="spec-stage-badge">ADVANCED (1.0%, n=9*)</div>'
            '<div class="spec-desc">Advanced impairment stage cohort. <em>*Test support n=9; conclusions carry statistical caveats.</em></div>'
            '</div>'
        )

    # Restrained Research Notice
    render_research_disclaimer()
