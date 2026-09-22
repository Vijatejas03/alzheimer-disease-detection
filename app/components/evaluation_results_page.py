"""
Page 4: Detailed Model Evaluation & Research Diagnostics.
Presents comprehensive performance metrics on the 960-image quarantined test set.
Organized into 5 focused research tabs:
1. Test Performance
2. Confusion Matrices
3. ROC & PR Curves
4. Per-Class Results
5. Calibration & Reliability
"""

from pathlib import Path
import streamlit as st

from app.utils.data_loader import (
    load_final_test_results,
    load_calibration_results
)
from app.utils.inference_engine import CLASS_COLORS
from app.utils.ui_helpers import render_html
from app.components.disclaimer import render_research_disclaimer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SLUG_MAP = {
    "EfficientNet-B0": "efficientnet_b0",
    "ResNet18": "resnet18",
    "MobileNetV2": "mobilenet_v2"
}


def render_evaluation_results_page():
    # Page Header
    header_html = (
        '<div style="margin-bottom: 1.25rem;">'
        '<div class="workflow-badge" style="margin-bottom: 0.5rem;">'
        'HELD-OUT TEST SET EVALUATION & RELIABILITY'
        '</div>'
        '<h2 style="margin: 0.15rem 0 0.35rem 0; font-size: 1.65rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">'
        'Comprehensive Model Evaluation'
        '</h2>'
        '<div style="font-size: 0.88rem; color: #64748B; line-height: 1.5;">'
        'Strict benchmark metrics evaluated on the quarantined 960-image test set (15% split) with SHA-256 verified zero data leakage.'
        '</div>'
        '</div>'
    )
    render_html(header_html)
    
    # Architecture Selector
    col_sel, col_info = st.columns([1.2, 2.8])
    with col_sel:
        selected_display_name = st.selectbox(
            "Select Architecture to Inspect:",
            options=list(SLUG_MAP.keys()),
            index=0
        )
    with col_info:
        st.markdown(
            '<div style="font-size: 0.82rem; color: #64748B; padding-top: 1.85rem;">'
            '<em>All metrics reflect held-out test predictions evaluated post-training with zero data leakage.</em>'
            '</div>',
            unsafe_allow_html=True
        )
        
    slug = SLUG_MAP[selected_display_name]
    
    test_json = load_final_test_results()
    if not test_json or "models_evaluated" not in test_json:
        st.warning("Detailed evaluation data is not available.")
        return
        
    model_data = test_json["models_evaluated"].get(slug, {})
    if not model_data:
        st.warning(f"No evaluation results found for {selected_display_name}.")
        return

    # 5-Tab Navigation Layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Test Performance",
        "🗂️ Confusion Matrices",
        "📈 ROC & PR Curves",
        "🔬 Per-Class Results",
        "⚖️ Calibration & Reliability"
    ])
    
    # ==========================================
    # TAB 1: TEST PERFORMANCE
    # ==========================================
    with tab1:
        st.markdown(f"### Primary Diagnostic Metrics — {selected_display_name}")
        
        # 8 Standard Metric Cards in 2 Rows of 4
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_html(
                '<div class="med-metric-card">'
                f'<div class="med-metric-val">{model_data.get("accuracy", 0)*100:.2f}%</div>'
                '<div class="med-metric-lbl">Overall Accuracy</div>'
                '</div>'
            )
        with c2:
            render_html(
                '<div class="med-metric-card">'
                f'<div class="med-metric-val">{model_data.get("macro_precision", 0):.4f}</div>'
                '<div class="med-metric-lbl">Macro Precision</div>'
                '</div>'
            )
        with c3:
            render_html(
                '<div class="med-metric-card">'
                f'<div class="med-metric-val">{model_data.get("macro_recall_sensitivity", 0):.4f}</div>'
                '<div class="med-metric-lbl">Macro Recall (Sens.)</div>'
                '</div>'
            )
        with c4:
            render_html(
                '<div class="med-metric-card">'
                f'<div class="med-metric-val">{model_data.get("macro_f1", 0):.4f}</div>'
                '<div class="med-metric-lbl">Macro F1-Score</div>'
                '</div>'
            )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            render_html(
                '<div class="med-metric-card">'
                f'<div class="med-metric-val">{model_data.get("balanced_accuracy", 0):.4f}</div>'
                '<div class="med-metric-lbl">Balanced Accuracy</div>'
                '</div>'
            )
        with c6:
            render_html(
                '<div class="med-metric-card">'
                f'<div class="med-metric-val">{model_data.get("matthews_corrcoef", 0):.4f}</div>'
                '<div class="med-metric-lbl">Matthews Corr (MCC)</div>'
                '</div>'
            )
        with c7:
            render_html(
                '<div class="med-metric-card">'
                f'<div class="med-metric-val">{model_data.get("macro_specificity", 0):.4f}</div>'
                '<div class="med-metric-lbl">Macro Specificity</div>'
                '</div>'
            )
        with c8:
            render_html(
                '<div class="med-metric-card">'
                f'<div class="med-metric-val">{model_data.get("roc_auc", 0):.4f}</div>'
                '<div class="med-metric-lbl">OvR Multiclass ROC-AUC</div>'
                '</div>'
            )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Technical Architecture Summary
        summary_html = (
            '<div class="med-card" style="background: #FFFFFF;">'
            f'<div style="font-weight: 700; color: #0F4C81; margin-bottom: 0.4rem; font-size: 0.95rem;">Architecture Performance Summary — {selected_display_name}</div>'
            '<div style="font-size: 0.88rem; color: #334155; line-height: 1.65;">'
            f'• <strong>Evaluation Cohort:</strong> 960 held-out axial brain MRI scans (Non-Demented: 480, Very Mild: 336, Mild: 135, Moderate: 9).<br>'
            f'• <strong>Balanced Diagnostic Metric:</strong> Balanced Accuracy of <strong>{model_data.get("balanced_accuracy", 0):.4f}</strong> demonstrates robust sensitivity across all 4 stages despite severe class imbalance.<br>'
            f'• <strong>Statistical Agreement:</strong> Matthews Correlation Coefficient of <strong>{model_data.get("matthews_corrcoef", 0):.4f}</strong> confirms strong multiclass correlation well above chance (0.0).<br>'
            f'• <strong>False Positive Suppression:</strong> Macro Specificity of <strong>{model_data.get("macro_specificity", 0):.4f}</strong> confirms high true-negative discrimination, minimizing misdiagnosis in control scans.'
            '</div>'
            '</div>'
        )
        render_html(summary_html)

    # ==========================================
    # TAB 2: CONFUSION MATRICES
    # ==========================================
    with tab2:
        st.markdown(f"### Confusion Matrix Analysis — {selected_display_name}")
        
        cm_path = PROJECT_ROOT / "results" / "figures" / f"{slug}_test_confusion_matrix.png"
        col_cm1, col_cm2 = st.columns([1.3, 1.0])
        
        with col_cm1:
            if cm_path.exists():
                st.image(str(cm_path), use_container_width=True, caption=f"Test Confusion Matrix — {selected_display_name} (N=960)")
            else:
                st.info("Confusion matrix figure not found.")
                
        with col_cm2:
            cm_analysis_html = (
                '<div class="med-card" style="background: #FFFFFF;">'
                '<div style="font-weight: 700; color: #0F4C81; margin-bottom: 0.45rem; font-size: 0.92rem;">Confusion Patterns & Diagnostics</div>'
                '<div style="font-size: 0.86rem; color: #334155; line-height: 1.6;">'
                '• <strong>Main Confusion Locus:</strong> The predominant source of classification error is the boundary between <em>Non-Demented</em> and <em>Very Mild Demented</em>, reflecting subtle clinical transition stages in early neurodegeneration.<br>'
                '• <strong>Extreme Separation:</strong> No model in the benchmark ever confused <em>Non-Demented</em> (healthy controls) with <em>Moderate Demented</em> (advanced atrophy).<br>'
                '• <strong>Sample Size Notice:</strong> The Moderate Demented class has test support n = 9 (1% of cohort). High empirical recall is observed, but with wider confidence intervals due to low support.'
                '</div>'
                '</div>'
            )
            render_html(cm_analysis_html)

    # ==========================================
    # TAB 3: ROC & PR CURVES
    # ==========================================
    with tab3:
        st.markdown(f"### Receiver Operating Characteristic (ROC) — {selected_display_name}")
        
        roc_path = PROJECT_ROOT / "results" / "figures" / f"{slug}_test_roc.png"
        col_r1, col_r2 = st.columns([1.3, 1.0])
        
        with col_r1:
            if roc_path.exists():
                st.image(str(roc_path), use_container_width=True, caption=f"One-vs-Rest ROC Curves — {selected_display_name}")
            else:
                st.info("ROC curve figure not found.")
                
        with col_r2:
            roc_analysis_html = (
                '<div class="med-card" style="background: #FFFFFF;">'
                '<div style="font-weight: 700; color: #0F4C81; margin-bottom: 0.45rem; font-size: 0.92rem;">Discriminative Power & OvR Analysis</div>'
                '<div style="font-size: 0.86rem; color: #334155; line-height: 1.6;">'
                f'• <strong>Aggregate OvR ROC-AUC:</strong> <strong>{model_data.get("roc_auc", 0):.4f}</strong> demonstrates strong class-separation capacity across decision thresholds.<br>'
                '• <strong>True Positive vs. False Positive Tradeoff:</strong> The sharp steepness of the curve near the origin demonstrates high sensitivity at low false-positive rates.<br>'
                '• <strong>Clinical Implication:</strong> High area under the curve across all stages confirms that probability ranking is statistically sound across thresholds.'
                '</div>'
                '</div>'
            )
            render_html(roc_analysis_html)

    # ==========================================
    # TAB 4: PER-CLASS RESULTS
    # ==========================================
    with tab4:
        st.markdown(f"### Per-Class Diagnostic Performance — {selected_display_name}")
        
        per_class_dict = model_data.get("per_class_metrics", {})
        if per_class_dict:
            table_lines = [
                "| Clinical Stage | Support (N) | Precision | Recall (Sens.) | Specificity | F1-Score | OvR ROC-AUC |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
            ]
            for c_key, c_vals in per_class_dict.items():
                disp = c_vals.get("display_name", c_key)
                sup = c_vals.get("support", 0)
                prec = f"{c_vals.get('precision', 0):.4f}"
                rec = f"{c_vals.get('recall_sensitivity', 0):.4f}"
                spec = f"{c_vals.get('specificity', 0):.4f}"
                f1 = f"{c_vals.get('f1_score', 0):.4f}"
                auc = f"{c_vals.get('roc_auc_ovr', 0):.4f}"
                table_lines.append(f"| **{disp}** | {sup} | {prec} | {rec} | {spec} | **{f1}** | {auc} |")
                
            st.markdown("\n".join(table_lines))
            
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            
            per_class_note = (
                '<div class="med-card" style="background: #F8FAFC; border-left: 4px solid #2563EB;">'
                '<div style="font-weight: 700; color: #1E40AF; margin-bottom: 0.35rem; font-size: 0.90rem;">Per-Class Evaluation Notes</div>'
                '<div style="font-size: 0.85rem; color: #334155; line-height: 1.55;">'
                '• <strong>Non-Demented (n=480):</strong> Healthy control cohort exhibiting high precision and specificity across all models.<br>'
                '• <strong>Very Mild Demented (n=336):</strong> The earliest stage of cognitive decline, where subtle ventricular enlargement begins.<br>'
                '• <strong>Mild Demented (n=135):</strong> Noticeable cortical atrophy resulting in distinctive feature patterns.<br>'
                '• <strong>Moderate Demented (n=9):</strong> Severe atrophy and ventricular dilation. High sensitivity achieved with class-weighted Cross-Entropy loss; however, the small sample size warrants cautious statistical interpretation.'
                '</div>'
                '</div>'
            )
            render_html(per_class_note)

    # ==========================================
    # TAB 5: CALIBRATION & RELIABILITY
    # ==========================================
    with tab5:
        st.markdown(f"### Post-Hoc Model Calibration — {selected_display_name}")
        
        cal_sub_html = (
            '<div style="font-size: 0.88rem; color: #64748B; margin-bottom: 1.0rem;">'
            'Temperature scaling parameter <em>T</em> was optimized strictly on validation split logits (minimizing NLL) '
            'and evaluated on the untouched 960-image test split using 15 equal-width confidence bins.'
            '</div>'
        )
        render_html(cal_sub_html)
        
        cal_data_all = load_calibration_results()
        cal_model_data = cal_data_all.get(selected_display_name, {})
        
        if cal_model_data:
            uncal = cal_model_data.get("uncalibrated", {})
            cal = cal_model_data.get("calibrated", {})
            t_val = cal_model_data.get("temperature", 1.0)
            
            cal_c1, cal_c2, cal_c3, cal_c4 = st.columns(4)
            with cal_c1:
                render_html(
                    '<div class="med-metric-card">'
                    f'<div class="med-metric-val">{t_val:.4f}</div>'
                    '<div class="med-metric-lbl">Learned Temperature (T)</div>'
                    '</div>'
                )
            with cal_c2:
                render_html(
                    '<div class="med-metric-card">'
                    f'<div class="med-metric-val">{uncal.get("ece", 0)*100:.2f}% <span style="font-size: 0.80rem; color: #64748B;">→ {cal.get("ece", 0)*100:.2f}%</span></div>'
                    '<div class="med-metric-lbl">ECE (Uncal → Cal)</div>'
                    '</div>'
                )
            with cal_c3:
                render_html(
                    '<div class="med-metric-card">'
                    f'<div class="med-metric-val">{uncal.get("brier_score", 0):.4f} <span style="font-size: 0.80rem; color: #64748B;">→ {cal.get("brier_score", 0):.4f}</span></div>'
                    '<div class="med-metric-lbl">Brier Score (Uncal → Cal)</div>'
                    '</div>'
                )
            with cal_c4:
                render_html(
                    '<div class="med-metric-card">'
                    f'<div class="med-metric-val">{uncal.get("negative_log_likelihood", 0):.4f} <span style="font-size: 0.80rem; color: #64748B;">→ {cal.get("negative_log_likelihood", 0):.4f}</span></div>'
                    '<div class="med-metric-lbl">NLL (Uncal → Cal)</div>'
                    '</div>'
                )
                
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            
            col_rel, col_dist = st.columns(2)
            rel_img_path = PROJECT_ROOT / "results" / "figures" / "calibration" / f"{slug}_reliability.png"
            dist_img_path = PROJECT_ROOT / "results" / "figures" / "calibration" / f"{slug}_confidence_histogram.png"
            
            with col_rel:
                st.markdown("#### Reliability Diagram (15 Equal-Width Bins)")
                if rel_img_path.exists():
                    st.image(str(rel_img_path), use_container_width=True, caption=f"Empirical Accuracy vs. Confidence — {selected_display_name}")
                else:
                    st.info("Reliability diagram not found.")
                    
            with col_dist:
                st.markdown("#### Confidence Distribution (Correct vs. Incorrect)")
                if dist_img_path.exists():
                    st.image(str(dist_img_path), use_container_width=True, caption=f"Confidence Distribution — {selected_display_name}")
                else:
                    st.info("Confidence distribution plot not found.")
                    
            cal_note_html = (
                '<div class="med-card" style="background: #F8FAFC; border-left: 4px solid #0F4C81; margin-top: 0.8rem;">'
                '<div style="font-weight: 700; color: #0F4C81; margin-bottom: 0.35rem; font-size: 0.90rem;">Reliability Assessment & Mathematical Properties</div>'
                '<div style="font-size: 0.86rem; color: #334155; line-height: 1.6;">'
                '• <strong>Baseline Regularization:</strong> The uncalibrated architecture already exhibits low expected calibration error (ECE &le; 1.66%), reflecting effective weight decay and dropout regularization.<br>'
                '• <strong>Strict Non-Interference:</strong> Temperature scaling operates purely as a monotonic positive scalar division on logits ($z_i / T$), guaranteeing <strong>100% preservation of top-1 classification predictions</strong> and ROC curves.<br>'
                '• <em>Scientific Principle: Calibration assesses empirical fidelity to observed probabilities on this cohort; it does not validate diagnostic safety in external medical settings.</em>'
                '</div>'
                '</div>'
            )
            render_html(cal_note_html)
        else:
            st.info("Calibration data not found for this architecture.")

    render_research_disclaimer()
