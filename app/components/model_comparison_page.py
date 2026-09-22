"""
Page 3: Architecture Benchmark & Quantitative Comparison.
Compares MobileNetV2, EfficientNet-B0, and ResNet-18 across the held-out test set.
"""

from pathlib import Path
import streamlit as st
import pandas as pd

from app.utils.data_loader import load_final_model_comparison
from app.utils.ui_helpers import render_html
from app.components.disclaimer import render_research_disclaimer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def render_model_comparison_page():
    st.markdown("## Architecture Benchmark")
    st.markdown(
        '<div style="font-size: 0.95rem; color: #475569; margin-top: -0.4rem; margin-bottom: 1.25rem;">'
        'Controlled empirical evaluation of MobileNetV2, EfficientNet-B0, and ResNet-18 on the 960-image '
        'quarantined test set (zero data leakage). No overall ranking is asserted.'
        '</div>',
        unsafe_allow_html=True
    )
    
    # 3 Model Snapshot Cards
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        m1_html = (
            '<div class="model-snapshot-box" style="border-top-color: #0284C7;">'
            '<div class="model-snapshot-title">MobileNetV2</div>'
            '<div class="model-snapshot-tag">Inverted Residuals • Depthwise Separable</div>'
            '<div style="font-size: 0.86rem; color: #334155; line-height: 1.85;">'
            '• <strong>Parameters:</strong> 2,228,996 (2.23M)<br>'
            '• <strong>Test Accuracy:</strong> 93.13%<br>'
            '• <strong>Macro F1:</strong> 0.9434<br>'
            '• <strong>ROC-AUC:</strong> 0.9936<br>'
            '• <strong>Inference Latency:</strong> 2.93 ms'
            '</div>'
            '</div>'
        )
        render_html(m1_html)
        
    with col_m2:
        m2_html = (
            '<div class="model-snapshot-box" style="border-top-color: #2563EB;">'
            '<div class="model-snapshot-title">EfficientNet-B0</div>'
            '<div class="model-snapshot-tag">Compound Scaled • MBConv + SE</div>'
            '<div style="font-size: 0.86rem; color: #334155; line-height: 1.85;">'
            '• <strong>Parameters:</strong> 4,012,672 (4.01M)<br>'
            '• <strong>Test Accuracy:</strong> 98.23%<br>'
            '• <strong>Macro F1:</strong> 0.9876<br>'
            '• <strong>ROC-AUC:</strong> 0.9989<br>'
            '• <strong>Inference Latency:</strong> 2.20 ms'
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
            '• <strong>Parameters:</strong> 11,178,564 (11.18M)<br>'
            '• <strong>Test Accuracy:</strong> 98.44%<br>'
            '• <strong>Macro F1:</strong> 0.9860<br>'
            '• <strong>ROC-AUC:</strong> 0.9991<br>'
            '• <strong>Inference Latency:</strong> 1.77 ms'
            '</div>'
            '</div>'
        )
        render_html(m3_html)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # COMPREHENSIVE 10-METRIC BENCHMARK MATRIX
    # =========================================================================
    st.markdown("### Benchmark Evaluation Matrix")
    st.caption("Official held-out test split (N = 960) evaluation across 10 diagnostic metrics.")
    
    rows = load_final_model_comparison()
    if rows:
        md_lines = [
            "| Architecture | Parameters | Accuracy | Precision | Recall | Macro F1 | Balanced Acc | MCC | Specificity | ROC-AUC | Latency |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]
        param_map = {
            "MobileNetV2": "2.23M",
            "EfficientNet-B0": "4.01M",
            "ResNet-18": "11.18M",
            "ResNet18": "11.18M"
        }
        lat_map = {
            "MobileNetV2": "2.93 ms",
            "EfficientNet-B0": "2.20 ms",
            "ResNet-18": "1.77 ms",
            "ResNet18": "1.77 ms"
        }
        for r in rows:
            m_name = r.get("model_name", "")
            params = param_map.get(m_name, f"{int(r.get('total_parameters', 0)):,}")
            acc = f"{float(r.get('accuracy', 0))*100:.2f}%"
            prec = f"{float(r.get('macro_precision', 0)):.4f}"
            rec = f"{float(r.get('macro_recall', 0)):.4f}"
            f1 = f"{float(r.get('macro_f1', 0)):.4f}"
            b_acc = f"{float(r.get('balanced_accuracy', 0)):.4f}"
            mcc = f"{float(r.get('matthews_corrcoef', 0)):.4f}"
            spec = f"{float(r.get('macro_specificity', 0)):.4f}"
            auc = f"{float(r.get('roc_auc', 0)):.4f}"
            lat = lat_map.get(m_name, f"{float(r.get('average_latency_ms', 0)):.2f} ms")
            md_lines.append(f"| **{m_name}** | {params} | {acc} | {prec} | {rec} | **{f1}** | {b_acc} | {mcc} | {spec} | {auc} | {lat} |")
            
        st.markdown("\n".join(md_lines))
    else:
        st.warning("Comparison metrics are currently unavailable.")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # COMPARATIVE METRIC VISUALIZATIONS
    # =========================================================================
    st.markdown("### Metric Trade-off Comparisons")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("#### Accuracy vs. Parameter Footprint")
        chart_data_acc = pd.DataFrame({
            "Architecture": ["MobileNetV2", "EfficientNet-B0", "ResNet-18"],
            "Accuracy (%)": [93.13, 98.23, 98.44],
            "Parameters (M)": [2.23, 4.01, 11.18]
        })
        st.bar_chart(chart_data_acc.set_index("Architecture")["Accuracy (%)"], color="#2563EB")
        st.caption("ResNet-18 and EfficientNet-B0 achieve comparable accuracy (>98.2%), while MobileNetV2 requires only 2.23M parameters.")
        
    with col_c2:
        st.markdown("#### Latency vs. ROC-AUC")
        chart_data_lat = pd.DataFrame({
            "Architecture": ["MobileNetV2", "EfficientNet-B0", "ResNet-18"],
            "Inference Latency (ms)": [2.93, 2.20, 1.77]
        })
        st.bar_chart(chart_data_lat.set_index("Architecture")["Inference Latency (ms)"], color="#0F4C81")
        st.caption("ResNet-18 exhibits the lowest inference latency (1.77 ms on RTX 3050 GPU) due to simple non-depthwise residual tensor operations.")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    # Architectural Insight Note
    insight_html = (
        '<div class="med-card" style="background: #F8FAFC; border-left: 4px solid #2563EB; margin-top: 0.5rem;">'
        '<div style="font-weight: 750; color: #0F4C81; font-size: 0.90rem; margin-bottom: 0.35rem;">Architectural Analysis & Research Findings</div>'
        '<p style="font-size: 0.86rem; color: #334155; line-height: 1.6; margin: 0;">'
        '• <strong>Compound Scaling Advantage:</strong> EfficientNet-B0 achieved 98.23% accuracy and the highest Balanced Accuracy (0.9903) with under 4.02M parameters.<br>'
        '• <strong>Residual Simplicity:</strong> ResNet-18 achieved 98.44% accuracy with the lowest latency (1.77 ms) and demonstrated superior physical robustness against noise.<br>'
        '• <strong>Efficiency Baseline:</strong> MobileNetV2 achieved 93.13% accuracy at only 2.23M parameters, demonstrating strong utility for edge computing constraints.<br>'
        '• <em>Methodological Principle: Each architecture presents distinct trade-offs across throughput, footprint, and robustness. No singular model is declared an overall winner.</em>'
        '</p>'
        '</div>'
    )
    render_html(insight_html)

    render_research_disclaimer()
