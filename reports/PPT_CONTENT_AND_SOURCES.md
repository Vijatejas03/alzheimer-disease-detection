# PPT Content & Sources
## Alzheimer_Final_Project_Presentation.pptx

**Generated:** `reports/generate_ppt.py`
**Output:** `reports/Alzheimer_Final_Project_Presentation.pptx`
**Slide count:** 24
**File size:** ~1.98 MB
**Aspect ratio:** 16:9 (13.33 × 7.5 inches)

---

## Slide-by-Slide Summary

| # | Title | Key Visual Elements |
|---|-------|-------------------|
| 1 | Title Slide | Brain MRI background (Grad-CAM overlay), institution badge, team placeholder, tech stack pills |
| 2 | Project at a Glance | 7-stage pipeline cards, 4 key metric highlight tiles |
| 3 | Problem Statement | 5 problem cards + global burden stats panel |
| 4 | Motivation | 4 motivation cards (explainability, multi-model, imbalance, OOD) |
| 5 | Project Objectives | 6 numbered objective cards (2-col grid) |
| 6 | Literature Survey | 6-paper reference table, research gap callout |
| 7 | Reference/Baseline | Pipeline flow diagram, 6 limitation cards |
| 8 | Research Gap | Feature comparison table (Baseline vs Ours) |
| 9 | Proposed System | Full 7-stage architecture flow + reject branch + sub-component row |
| 10 | Dataset | Class distribution bar chart, image format specs, split cards |
| 11 | Preprocessing | 6-stage pipeline flow, 5 key guarantee cards |
| 12 | Class Imbalance | Visual bar chart (50× imbalance), 4 solution cards |
| 13 | Model Architectures | 3 model cards (params, backbone, head, strength, metrics) |
| 14 | Training Setup | Hardware table + 10-field hyperparameter table |
| 15 | Evaluation Metrics | 6 metric cards with results and rationale |
| 16 | Model Comparison | Metrics table + model_comparison chart (actual PNG) + insights |
| 17 | Confusion Matrices | All 3 confusion matrix PNGs (actual from results/figures/) |
| 18 | Grad-CAM | 3 Grad-CAM overlay PNGs (EfficientNet-B0) + color legend + explanation |
| 19 | Calibration & Errors | Calibration table + reliability diagram PNG + error analysis |
| 20 | Robustness | Main robustness comparison PNG + 5 perturbation categories |
| 21 | OOD Safeguard | 8-stage pipeline list + accept/reject lists + empirical validation |
| 22 | Streamlit App | 8-page feature cards + tech stack + live URL |
| 23 | Deployment | CI/CD pipeline flow + QR code + GitHub/Streamlit details |
| 24 | Limitations + Conclusion | 3-column: Limitations / Future Work / Conclusion + Thank You banner |

---

## Real Metrics Used

### Dataset
- Total: **6,400** unique images · 128×128 grayscale JPEG
- Non-Demented: **3,200** · Very Mild: **2,240** · Mild: **896** · Moderate: **64**
- Split: **70% / 15% / 15%** (4,480 / 960 / 960) · 0 hash-overlap

### Model Parameters
- MobileNetV2: **2,228,996**
- EfficientNet-B0: **4,012,672**
- ResNet-18: **11,178,564**

### Held-Out Test Results (960 images)
| Model | Accuracy | Macro F1 | Balanced Acc | MCC | ROC-AUC | Latency |
|-------|----------|----------|--------------|-----|---------|---------|
| MobileNetV2 | 93.13% | 94.34% | 94.25% | 0.887 | 0.9936 | 2.93ms |
| EfficientNet-B0 | 98.23% | 98.76% | 99.03% | 0.971 | 0.9989 | 2.20ms |
| ResNet-18 | 98.44% | 98.60% | 98.29% | 0.974 | 0.9991 | 1.77ms |

### Error Analysis
- MobileNetV2: 66 errors · EfficientNet-B0: 17 errors · ResNet-18: 15 errors
- >68% of errors at Non-Demented ↔ Very Mild boundary
- Moderate Demented test support: 9 samples

### Training Config
- PyTorch · CUDA 12.4 · RTX 3050 Laptop (4GB VRAM)
- Batch 16 · 25 epochs · LR 1e-4 · WD 1e-4 · Seed 42 · AMP

---

## Visual Assets Embedded

### Actual PNG files from `results/figures/`:
- `overall_model_comparison.png` — Slide 16
- `mobilenet_v2_test_confusion_matrix.png` — Slide 17
- `efficientnet_b0_test_confusion_matrix.png` — Slide 17
- `resnet18_test_confusion_matrix.png` — Slide 17
- `gradcam/efficientnet_b0/non_1119_correct_overlay.png` — Slide 18
- `gradcam/efficientnet_b0/verymild_44_correct_overlay.png` — Slide 18
- `gradcam/efficientnet_b0/mild_430_correct_overlay.png` — Slide 18
- `calibration/all_models_reliability_comparison.png` — Slide 19
- `robustness/model_robustness_comparison.png` — Slide 20
- `error_analysis/efficientnet_b0_error_confidence.png` — Slide 19
- `gradcam/efficientnet_b0/mild_430_correct_overlay.png` — Slide 1 (background motif)

### QR Code:
- Generated dynamically at build time for: `https://alzheimer-xai-vijay.streamlit.app`
- Appears on Slide 23

---

## Literature References (Slide 6)

All papers are real published works — NOT invented:

1. **Farooq et al. (2017)** — "A Deep CNN Based Multi-Class Classification of Alzheimer's Disease Using MRI" — IEEE ISICS
2. **Wen et al. (2020)** — "Convolutional Neural Networks for Classification of Alzheimer's Disease: Overview and Reproducible Evaluation" — Medical Image Analysis
3. **Shanmugam et al. (2022)** — Transfer learning ResNet-50 for AD staging on Kaggle MRI dataset
4. **Islam & Zhang (2018)** — "Brain MRI Analysis for Alzheimer's Disease Diagnosis Using an Ensemble System of Deep Convolutional Neural Networks" — Brain Informatics
5. **Spasov et al. (2019)** — "A Parameter-Efficient Deep Learning Approach to Predict Conversion from Mild Cognitive Impairment to Alzheimer's Disease" — NeuroImage
6. **Odusami et al. (2021)** — "Analysis of Features of Alzheimer's Disease: Detection of Early Stage from Functional Brain Changes in Magnetic Resonance Images using a Finetuned ResNet18 Network" — Diagnostics

---

## Placeholders Remaining (Must Fill Before Submission)

> **IMPORTANT:** Update these before printing or presenting!

1. **Slide 1 — Team Members:** Replace `[Your Name Here]`, `[USN Here]`, `[Team Member 2]`
2. **Slide 1 — Guide:** Replace `[Project Guide Name, Designation]`
3. **Slide 1 — College:** Replace `[College Name]`
4. **All slides — Footer:** Currently shows project short name — update if needed

---

## Academic Framing (As Specified)

- System described as: **"academic research prototype"**, NOT "medical diagnosis AI"
- Grad-CAM described as: **"model attribution visualization"**, NOT "Alzheimer's biomarker detection"
- Accuracy described as: **"held-out test accuracy on this dataset and split"**, NOT "clinical accuracy"
- OOD safeguard: **"academic input-domain safeguard"**, NOT "universal OOD detector"
- Moderate class results: noted with caveat (only 9 test samples)

---

## Deployment Info

- **GitHub:** https://github.com/Vijatejas03/alzheimer-disease-detection
- **Live URL:** https://alzheimer-xai-vijay.streamlit.app
