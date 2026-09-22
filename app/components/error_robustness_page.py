"""
Page 6: Error Analysis & Controlled Perturbation Robustness.
Provides research-grade failure mode diagnostics, high-confidence error profiling,
cross-architecture overlap, and controlled scanner variation robustness on the 960-image test set.
"""

from pathlib import Path
import streamlit as st

from app.utils.data_loader import (
    load_error_analysis_summary,
    load_robustness_results
)
from app.utils.ui_helpers import render_html
from app.components.disclaimer import render_research_disclaimer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SLUG_MAP = {
    "EfficientNet-B0": "efficientnet_b0",
    "ResNet18": "resnet18",
    "MobileNetV2": "mobilenet_v2"
}


def render_error_robustness_page():
    # Page Header
    header_html = (
        '<div style="margin-bottom: 1.25rem;">'
        '<div class="workflow-badge" style="margin-bottom: 0.5rem;">'
        'FAILURE MODE PROFILING & CONTROLLED PERTURBATION BENCHMARK'
        '</div>'
        '<h2 style="margin: 0.15rem 0 0.35rem 0; font-size: 1.65rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">'
        'Error Analysis & Model Robustness'
        '</h2>'
        '<div style="font-size: 0.88rem; color: #64748B; line-height: 1.5;">'
        'Auditing failure modes, prediction uncertainty when incorrect, cross-architecture disagreement, '
        'and resilience to synthetic acquisition variations across the 960-image held-out test cohort.'
        '</div>'
        '</div>'
    )
    render_html(header_html)

    err_summary = load_error_analysis_summary()
    robust_summary = load_robustness_results()

    if not err_summary:
        st.warning("Error analysis data is not available.")
        return

    # Top-Level Tri-Model Failure & Reliability KPIs
    overlap_info = err_summary.get("cross_model_overlap", {})
    high_conf_info = err_summary.get("high_confidence_error_counts", {})
    low_conf_info = err_summary.get("low_confidence_correct_counts", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_html(
            '<div class="med-metric-card">'
            f'<div class="med-metric-val">{overlap_info.get("correct_all_three", 875)} / 960</div>'
            '<div class="med-metric-lbl">Tri-Model Consensus (91.15%)</div>'
            '</div>'
        )
    with c2:
        render_html(
            '<div class="med-metric-card">'
            f'<div class="med-metric-val">{high_conf_info.get("total_ge_0_80", 40)}</div>'
            '<div class="med-metric-lbl">High-Conf Errors (&ge;80%)</div>'
            '</div>'
        )
    with c3:
        render_html(
            '<div class="med-metric-card">'
            f'<div class="med-metric-val">{low_conf_info.get("total_lt_0_70", 54)}</div>'
            '<div class="med-metric-lbl">Low-Conf Correct (&lt;70%)</div>'
            '</div>'
        )
    with c4:
        render_html(
            '<div class="med-metric-card">'
            f'<div class="med-metric-val" style="color: #0D9488;">{overlap_info.get("misclassified_all_three", 1)} / 960</div>'
            '<div class="med-metric-lbl">All-Three Failure (0.10%)</div>'
            '</div>'
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Cross-Architecture Error Distribution Overview
    cross_overview_html = (
        '<div class="med-card" style="background: #FFFFFF; margin-bottom: 1.25rem;">'
        '<div style="font-weight: 700; color: #0F4C81; margin-bottom: 0.4rem; font-size: 0.95rem;">Cross-Architecture Error Distribution Summary</div>'
        '<div style="font-size: 0.88rem; color: #334155; line-height: 1.65;">'
        '• <strong>MobileNetV2:</strong> 66 total errors (93.13% test acc) &bull; 24 high-confidence errors &bull; 56 unique errors not shared with other models.<br>'
        '• <strong>EfficientNet-B0:</strong> 17 total errors (98.23% test acc) &bull; 10 high-confidence errors &bull; 10 unique errors.<br>'
        '• <strong>ResNet18:</strong> 15 total errors (98.44% test acc) &bull; 6 high-confidence errors &bull; 7 unique errors.<br>'
        '• <strong>Complementary Ensemble Behavior:</strong> 73 scans are failed by only 1 model, 11 scans by 2 models, and <strong>only 1 scan in the entire 960-image test set is failed by all three models</strong> (<code>verymild_862.jpg</code>). This demonstrates strong architectural complementarity.'
        '</div>'
        '</div>'
    )
    render_html(cross_overview_html)

    # Architecture Selector
    st.markdown("### Architecture-Specific Error Inspection")
    selected_display_name = st.selectbox(
        "Select Model to Inspect:",
        options=list(SLUG_MAP.keys()),
        index=0
    )
    slug = SLUG_MAP[selected_display_name]

    m_err_data = err_summary.get("class_wise_metrics", {}).get(selected_display_name, {})
    high_err_counts = high_conf_info.get("by_model", {}).get(selected_display_name, 0)
    low_corr_counts = low_conf_info.get("by_model", {}).get(selected_display_name, 0)

    if m_err_data:
        # 4 Architecture Specific Metric Cards
        total_errors = sum(c["incorrect_predictions"] for c in m_err_data.get("class_metrics", {}).values())
        test_acc_pct = (960 - total_errors) / 960 * 100.0

        e_c1, e_c2, e_c3, e_c4 = st.columns(4)
        with e_c1:
            render_html(
                '<div class="med-metric-card">'
                f'<div class="med-metric-val">{total_errors} / 960</div>'
                f'<div class="med-metric-lbl">Total Errors ({test_acc_pct:.2f}% Acc)</div>'
                '</div>'
            )
        with e_c2:
            render_html(
                '<div class="med-metric-card">'
                f'<div class="med-metric-val">{high_err_counts}</div>'
                '<div class="med-metric-lbl">High-Conf Errors (&ge;80%)</div>'
                '</div>'
            )
        with e_c3:
            render_html(
                '<div class="med-metric-card">'
                f'<div class="med-metric-val">{low_corr_counts}</div>'
                '<div class="med-metric-lbl">Low-Conf Correct (&lt;70%)</div>'
                '</div>'
            )
        with e_c4:
            unique_cnt = overlap_info.get("unique_errors", {}).get(selected_display_name, 0)
            render_html(
                '<div class="med-metric-card">'
                f'<div class="med-metric-val">{unique_cnt}</div>'
                '<div class="med-metric-lbl">Unique Architecture Errors</div>'
                '</div>'
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Figures: Error Confidence Distribution & Cross-Model Overlap
        col_err_conf, col_overlap = st.columns(2)
        err_conf_path = PROJECT_ROOT / "results" / "figures" / "error_analysis" / f"{slug}_error_confidence.png"
        overlap_fig_path = PROJECT_ROOT / "results" / "figures" / "error_analysis" / "cross_model_error_overlap.png"

        with col_err_conf:
            st.markdown("#### Error Confidence & Uncertainty Distribution")
            if err_conf_path.exists():
                st.image(str(err_conf_path), use_container_width=True, caption=f"Confidence & Entropy Separation — {selected_display_name}")
            else:
                st.info("Error confidence figure not found.")

        with col_overlap:
            st.markdown("#### Cross-Architecture Error Overlap")
            if overlap_fig_path.exists():
                st.image(str(overlap_fig_path), use_container_width=True, caption="Model Error Overlap Across All 3 Architectures")
            else:
                st.info("Cross-model error overlap figure not found.")

        # Class-Wise Error & Confusion Breakdown Table
        st.markdown("#### Class-Wise Error & Confusion Breakdown")
        cm_metrics = m_err_data.get("class_metrics", {})
        if cm_metrics:
            err_table_lines = [
                "| Clinical Stage | Support | Correct | Incorrect | Accuracy | Precision | Recall (Sens.) | False Pos. | False Neg. |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
            ]
            for c_name, c_vals in cm_metrics.items():
                disp = c_name.replace("_", " ")
                sup = c_vals.get("total_samples", 0)
                corr = c_vals.get("correct_predictions", 0)
                inc = c_vals.get("incorrect_predictions", 0)
                acc = f"{c_vals.get('accuracy', 0)*100:.2f}%"
                prec = f"{c_vals.get('precision', 0)*100:.2f}%"
                rec = f"{c_vals.get('recall', 0)*100:.2f}%"
                fp = c_vals.get("false_positives", 0)
                fn = c_vals.get("false_negatives", 0)
                err_table_lines.append(f"| **{disp}** | {sup} | {corr} | {inc} | {acc} | {prec} | {rec} | {fp} | {fn} |")
            st.markdown("\n".join(err_table_lines))

        # Top Confusion Pairs
        conf_pairs = m_err_data.get("confusion_pairs", [])
        if conf_pairs:
            pair_bullets = []
            for p in conf_pairs[:4]:
                t_cls = p["true_class"].replace("_", " ")
                p_cls = p["predicted_class"].replace("_", " ")
                cnt = p["count"]
                pct = p["percentage_of_true_class"]
                pair_bullets.append(f"• **{t_cls} → {p_cls}**: {cnt} cases ({pct:.2f}% of {t_cls} test cases)")

            conf_html = (
                '<div class="med-card" style="background: #F8FAFC; border-left: 4px solid #0F4C81; margin-top: 0.8rem; margin-bottom: 1.25rem;">'
                '<div style="font-weight: 700; color: #0F4C81; margin-bottom: 0.35rem; font-size: 0.90rem;">Primary Model Confusion Patterns</div>'
                '<div style="font-size: 0.86rem; color: #334155; line-height: 1.6;">'
                + "<br>".join(pair_bullets) +
                '<br><em>*Note: Distinguishing Non-Demented from Very Mild Demented accounts for the overwhelming majority of errors across all architectures. No model ever confused Non-Demented with Moderate Demented. Moderate Demented test support = 9; conclusions for this class carry statistical uncertainty.</em>'
                '</div>'
                '</div>'
            )
            render_html(conf_html)

    # ==========================================
    # CONTROLLED ROBUSTNESS EVALUATION
    # ==========================================
    st.markdown("---")
    st.markdown("### Controlled Perturbation Robustness Testing")
    
    rob_intro_html = (
        '<div style="font-size: 0.88rem; color: #64748B; margin-bottom: 1.0rem;">'
        '10 controlled image perturbations applied to the 960-image test set simulating real-world acquisition noise, '
        'receiver coil gain variation, slice thickness downsampling, and patient head micro-rotations.'
        '</div>'
    )
    render_html(rob_intro_html)

    m_robust = robust_summary.get(selected_display_name, [])
    robust_fig_path = PROJECT_ROOT / "results" / "figures" / "robustness" / "model_robustness_comparison.png"

    col_rob_table, col_rob_img = st.columns([1.1, 1.0])
    with col_rob_table:
        if m_robust:
            rob_table_lines = [
                "| Perturbation Condition | Perturbed Acc | Change (Δ Acc) | Macro F1 |",
                "| :--- | :---: | :---: | :---: |"
            ]
            for r in m_robust:
                lbl = r.get("label", "")
                p_acc = f"{r.get('perturbed_accuracy', 0)*100:.2f}%"
                delta = f"{r.get('accuracy_change', 0)*100:+.2f}%"
                f1 = f"{r.get('perturbed_macro_f1', 0):.4f}"
                rob_table_lines.append(f"| **{lbl}** | {p_acc} | {delta} | {f1} |")
            st.markdown("\n".join(rob_table_lines))

    with col_rob_img:
        if robust_fig_path.exists():
            st.image(str(robust_fig_path), use_container_width=True, caption="Cross-Model Robustness Across All 10 Perturbations")
        else:
            st.info("Robustness figure not found.")

    rob_note_html = (
        '<div class="med-card" style="background: #F8FAFC; border-left: 4px solid #0D9488; margin-top: 1.0rem;">'
        '<div style="font-weight: 700; color: #0D9488; margin-bottom: 0.35rem; font-size: 0.90rem;">Controlled Robustness Findings & Insights</div>'
        '<div style="font-size: 0.86rem; color: #334155; line-height: 1.6;">'
        '• <strong>Photometric Resilience:</strong> All architectures exhibit high stability under simulated RF gain variations (brightness/contrast &plusmn;15%, &le;1.77% accuracy drop).<br>'
        '• <strong>Gaussian Noise Disparity:</strong> ResNet-18 maintained <strong>95.83% accuracy</strong> under deterministic additive Gaussian noise (&sigma;=0.03), whereas MobileNetV2 and EfficientNet-B0 suffered substantial high-frequency degradation. Residual identity skip connections preserve low-frequency structural anatomy far better than depthwise separable convolutions.<br>'
        '• <strong>Head Motion & Downsampling:</strong> Micro-rotations (&plusmn;5&deg;) and 2&times; downsampling (112&times;112 px) produced minimal performance degradation (&le;2.5%), demonstrating clinical scan tolerance.<br>'
        '• <em>Research Protocol: Perturbations were evaluated strictly in an auxiliary controlled experiment; official benchmark test labels and weights remain 100% untouched.</em>'
        '</div>'
        '</div>'
    )
    render_html(rob_note_html)

    render_research_disclaimer()
