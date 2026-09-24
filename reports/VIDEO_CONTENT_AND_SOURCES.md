# Video Content & Sources
## Alzheimer_Project_Tutorial_Demo.mp4

---

## Video Properties

| Property | Value |
|----------|-------|
| **Filename** | `Alzheimer_Project_Tutorial_Demo.mp4` |
| **Duration** | 40.5 seconds |
| **Resolution** | 1920 × 1080 (Full HD 1080p) |
| **Frame rate** | 30 fps |
| **Total frames** | 1,215 |
| **File size** | ~29.3 MB |
| **Codec** | MP4V (OpenCV VideoWriter) |
| **Aspect ratio** | 16:9 landscape |
| **Generated** | `reports/generate_video.py` |

---

## Scene Breakdown

| # | Scene | Duration | Content |
|---|-------|----------|---------|
| 1 | QR Code / Access | ~3.5s | QR code for live URL, project title, team badge |
| 2 | App Opening | ~3.5s | Simulated browser interface, key project stats |
| 3 | Home / Overview | ~4.0s | 8-page navigation menu, 3-model cards, pipeline flow |
| 4 | MRI Analysis | ~4.0s | Actual MRI sample, analysis controls, model selector |
| 5 | Prediction Results | ~4.5s | Class probabilities, 3-model agreement, uncertainty |
| 6 | Grad-CAM | ~5.5s | 3 real Grad-CAM overlays, color legend, attribution disclaimer |
| 7 | Model Comparison | ~5.5s | Metrics table (real numbers), all 3 confusion matrix PNGs |
| 8 | Final Frame | ~5.5s | Tagline, Grad-CAM showcase, team info, QR code, live URL |
| — | Fade transitions | ~4.0s (cumulative) | Smooth 0.5s crossfades between all scenes |

---

## Project Assets Used

### MRI Samples (from `data/test_samples/`)
- `non_1263.jpg` — used in Scene 4 (MRI Analysis) and Scene 5 (Prediction)
- `verymild_1576.jpg` — used in Scene 6 (Grad-CAM panel)
- `mild_33.jpg` — used in Scene 6 (Grad-CAM panel)

### Grad-CAM Overlays (from `results/figures/gradcam/efficientnet_b0/`)
- `non_1119_correct_overlay.png` — Non-Demented Grad-CAM (Scenes 6, 8)
- `verymild_44_correct_overlay.png` — Very Mild Demented Grad-CAM (Scenes 6, 8)
- `mild_430_correct_overlay.png` — Mild Demented Grad-CAM (Scenes 1, 6, 8)

### Confusion Matrices (from `results/figures/`)
- `mobilenet_v2_test_confusion_matrix.png` — Scene 7
- `efficientnet_b0_test_confusion_matrix.png` — Scene 7
- `resnet18_test_confusion_matrix.png` — Scene 7

### QR Code
- Generated dynamically for: `https://alzheimer-xai-vijay.streamlit.app`
- Appears in: Scene 1 (large, centered) and Scene 8 (bottom right)

---

## Verified Metrics Used in Video

All metrics shown in the video are taken directly from `results/metrics/final_model_comparison.csv` and `results/metrics/final_test_results.json`. No values were fabricated.

| Model | Accuracy | Macro F1 | Balanced Acc | MCC | ROC-AUC |
|-------|----------|----------|--------------|-----|---------|
| MobileNetV2 | 93.13% | 94.34% | 94.25% | 0.887 | 0.9936 |
| EfficientNet-B0 | 98.23% | 98.76% | 99.03% | 0.971 | 0.9989 |
| ResNet-18 | 98.44% | 98.60% | 98.29% | 0.974 | 0.9991 |

Error counts shown: MobileNetV2=66, EfficientNet-B0=17, ResNet-18=15

---

## Team Information Shown

- **Team Leader:** Vijaytejas A C | 1VK23CS074
- **Members:** Parvati Revannavar | Moulya S | Priyadarshini K
- **Guide:** Dr. Vidya A, HOD-CSE
- **Institution:** Vivekananda Institute of Technology | 2025–26

---

## Application URL

- **Live URL:** https://alzheimer-xai-vijay.streamlit.app
- Displayed in: Scene 1, Scene 2, Scene 8

---

## Compliance Confirmations

| Check | Status |
|-------|--------|
| ML implementation NOT modified | CONFIRMED |
| Trained model weights NOT modified | CONFIRMED |
| Dataset NOT modified | CONFIRMED |
| Results/metrics NOT modified | CONFIRMED |
| Streamlit app NOT modified | CONFIRMED |
| No fabricated metrics used | CONFIRMED |
| No fabricated medical claims | CONFIRMED |
| No claim of clinical validation | CONFIRMED |
| Grad-CAM correctly described as model attribution | CONFIRMED |
| System correctly described as research prototype | CONFIRMED |
| Patient identifiable information NOT used | CONFIRMED |
| Video uses only curated demo MRI samples | CONFIRMED |

---

## Academic Framing Applied

- System described as **"AI-based Brain MRI Research Prototype"** — NOT a medical device
- Grad-CAM sections include disclaimer: **"Model attribution ≠ clinical diagnosis"**
- Accuracy described as held-out test accuracy on this dataset and split
- Moderate class caveat noted in confusion matrix section
