"""
Page 7: Research Methodology & Scientific Workflow.
Presents the complete experimental pipeline in 11 structured scientific stages:
01 Dataset Ingestion & Spectrum
02 Cryptographic Hash Audit (Zero Leakage)
03 Preprocessing & Resampling
04 Quarantined Data Partitioning (70/15/15)
05 Model Training & Class Imbalance Mitigation
06 Comprehensive Test Set Evaluation
07 Grad-CAM Explainability & Saliency
08 Safe Inference Gate & Uncertainty Screening
09 Post-Hoc Calibration & Reliability
10 Error Profiling & Controlled Robustness
11 Scientific Limitations & Academic Scope
"""

import streamlit as st
from app.utils.ui_helpers import render_html
from app.components.disclaimer import render_research_disclaimer


def render_methodology_page():
    # Page Header
    header_html = (
        '<div style="margin-bottom: 1.25rem;">'
        '<div class="workflow-badge" style="margin-bottom: 0.5rem;">'
        'SCIENTIFIC EXPERIMENTAL PIPELINE & RIGOR'
        '</div>'
        '<h2 style="margin: 0.15rem 0 0.35rem 0; font-size: 1.65rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">'
        'Research Methodology'
        '</h2>'
        '<div style="font-size: 0.88rem; color: #64748B; line-height: 1.5;">'
        'Complete end-to-end experimental protocol from raw neuroimaging cohort ingestion '
        'to visual explainability, post-hoc calibration, error analysis, and safe inference safeguards.'
        '</div>'
        '</div>'
    )
    render_html(header_html)
    
    stages = [
        (
            "01",
            "Dataset Ingestion & Disease Spectrum",
            "The benchmark cohort comprises exactly 6,400 axial brain MRI scans (128 × 128 px, single-channel) "
            "spanning four clinical stages: Non-Demented (3,200), Very Mild Demented (2,240), Mild Demented (896), "
            "and Moderate Demented (64).",
            """
| Disease Stage | Unique Scans | Percentage | Clinical Role in Spectrum |
| :--- | :---: | :---: | :--- |
| **Non-Demented** | 3,200 | 50.0% | Normal healthy control group |
| **Very Mild Demented** | 2,240 | 35.0% | Early subtle neurodegeneration |
| **Mild Demented** | 896 | 14.0% | Intermediate cognitive impairment |
| **Moderate Demented** | 64 | 1.0% | Advanced structural neurodegeneration |
"""
        ),
        (
            "02",
            "Cryptographic Data Audit & Zero Leakage Verification",
            "To guarantee unbiased evaluation, all 6,400 images underwent cryptographic SHA-256 hash auditing. "
            "Zero duplicate hashes and exactly 0.00% data leakage were verified across the partitions. "
            "The test partition was immediately quarantined and remained untouched throughout model development.",
            None
        ),
        (
            "03",
            "Image Preprocessing & Standardization",
            "Single-channel grayscale scans are converted to 3-channel RGB to match ImageNet pre-trained backbones, "
            "resampled to 224 × 224 via bilinear interpolation, and standardized using canonical ImageNet statistics "
            "(Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]).",
            None
        ),
        (
            "04",
            "Quarantined Stratified Partitioning (70/15/15)",
            "Stratified random splitting was executed with fixed random seed 42 to preserve exact class proportions:\n\n"
            "• **Training Set (70%):** 4,480 images (Non: 2,240 | Very Mild: 1,568 | Mild: 627 | Moderate: 45)\n\n"
            "• **Validation Set (15%):** 960 images (Non: 480 | Very Mild: 336 | Mild: 134 | Moderate: 10)\n\n"
            "• **Held-Out Test Set (15%):** 960 images (Non: 480 | Very Mild: 336 | Mild: 135 | Moderate: 9)",
            None
        ),
        (
            "05",
            "Model Training & Class Imbalance Mitigation",
            "Three convolutional backbones (MobileNetV2, EfficientNet-B0, ResNet-18) were trained using AdamW "
            "(lr=1e-4, weight_decay=1e-4), FP16 Automatic Mixed Precision on NVIDIA RTX 3050 GPU, and class-weighted "
            "Cross-Entropy loss (weighting Moderate Demented by 24.89×) to overcome severe class imbalance.",
            None
        ),
        (
            "06",
            "Comprehensive Test Set Benchmark",
            "Models were evaluated strictly on the untouched 960-image test set across 11 diagnostic metrics: "
            "Accuracy, Macro Precision, Macro Recall, Macro F1, Balanced Accuracy, Specificity, Matthews Correlation "
            "Coefficient (MCC), and One-vs-Rest Multiclass ROC-AUC.",
            None
        ),
        (
            "07",
            "Grad-CAM Explainability & Visual Saliency",
            "Gradient-weighted Class Activation Mapping computes gradients of the target class score with respect "
            "to feature maps of the final convolutional layer (`features.8` for EfficientNet-B0, `layer4.1` for ResNet-18, "
            "`features.18` for MobileNetV2), isolating anatomical regions that contributed positively to the classification.",
            None
        ),
        (
            "08",
            "Safe Inference Gate & Uncertainty Screening",
            "A 5-stage validation gate intercepts corrupt, blank, extreme ratio (>4:1), or high-chroma non-medical "
            "inputs before inference. Mathematical prediction uncertainty is calculated via normalized Shannon entropy "
            "and prediction margin.",
            None
        ),
        (
            "09",
            "Post-Hoc Model Calibration & Reliability",
            "Empirical softmax confidence was statistically calibrated via temperature scaling ($z_i / T$). "
            "Parameter $T$ was optimized strictly on validation split logits by minimizing Negative Log-Likelihood, "
            "reducing Expected Calibration Error (ECE) across all architectures while preserving 100% of top-1 predictions.",
            None
        ),
        (
            "10",
            "Failure Mode Profiling & Controlled Robustness",
            "Comprehensive error analysis profiled failure patterns (e.g. Non-Demented vs Very Mild Demented confusion), "
            "high-confidence errors, and cross-model agreement. Robustness was evaluated across 10 controlled synthetic "
            "perturbations (noise, brightness, contrast, rotation, downsampling), demonstrating ResNet-18 noise resilience "
            "and high tri-model complementarity (only 1 shared error out of 960 scans).",
            None
        ),
        (
            "11",
            "Scientific Limitations & Research Scope",
            "The system evaluates 2D axial MRI slices rather than continuous 3D volumetric sequences. "
            "The anonymized public dataset lacks longitudinal patient tracking and demographic covariates. "
            "The Moderate Demented class has test support n = 9, introducing statistical uncertainty. "
            "Visual heatmaps reflect mathematical filter activations rather than verified physiological biomarkers.",
            None
        )
    ]
    
    for num, title, explanation, table_md in stages:
        stage_html = (
            '<div class="stage-step-row">'
            f'<div class="stage-step-num">{num}</div>'
            '<div class="stage-step-content">'
            f'<div class="stage-step-title">{title}</div>'
            f'<p class="stage-step-text">{explanation}</p>'
            '</div>'
            '</div>'
        )
        render_html(stage_html)
        if table_md:
            with st.expander(f"Inspect {title} Breakdown"):
                st.markdown(table_md)

    render_research_disclaimer()
