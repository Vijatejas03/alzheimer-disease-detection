# Project Methodology & Baseline Documentation

**Project:** Explainable Deep Learning-Based Multi-Stage Alzheimer's Disease Detection Using Brain MRI  
**Type:** Final-Year Major Project (Academic / Research Prototype)

---

## 1. Baseline (Senior Project) Summary

The senior/reference project provides the methodological starting point for this work.

### 1.1 Senior Project Approach
| Aspect | Senior Project |
|:---|:---|
| **Framework** | TensorFlow / Keras |
| **Architecture** | Single MobileNet model |
| **Task** | 4-class MRI classification |
| **Web Interface** | Flask |
| **Explainability** | None reported |
| **Model Comparison** | None (single model only) |
| **Evaluation** | Accuracy only (reported) |
| **Class Imbalance** | Not explicitly addressed |
| **Reproducibility** | Not systematically documented |

### 1.2 Identified Limitations of the Senior Project
1. **Single architecture assumption** — No comparative benchmarking; performance of the chosen architecture is not validated against alternatives.
2. **Black-box output** — No explainability mechanism to inspect model reasoning.
3. **Incomplete evaluation** — Reporting only accuracy is insufficient for imbalanced multi-class medical imaging datasets. Sensitivity, specificity, MCC, and balanced accuracy are absent.
4. **Class imbalance not addressed** — Standard cross-entropy without class weighting can produce biased majority-class predictions.
5. **Reproducibility gap** — Absence of fixed random seeds, documented train/val/test split strategy, or experiment configuration tracking.

> **Note:** The senior project accuracy figures are their experimental results.
> We will compute our own independent results after training on our own dataset split.
> We will not report the senior project's accuracy as our result under any circumstances.

---

## 2. Our Proposed Improvements

| Aspect | Our Project |
|:---|:---|
| **Framework** | PyTorch |
| **Architectures** | MobileNetV2, EfficientNet-B0, ResNet-18 (comparative study) |
| **Task** | 4-class MRI classification |
| **Web Interface** | Streamlit |
| **Explainability** | Grad-CAM (image-region attribution) |
| **Model Comparison** | Controlled experimental benchmarking across 3 architectures |
| **Evaluation** | Accuracy, Precision, Recall/Sensitivity, Specificity, F1, Balanced Accuracy, MCC, Confusion Matrix |
| **Class Imbalance** | Class-weighted cross-entropy loss (effectiveness to be validated experimentally) |
| **Reproducibility** | Fixed seeds, JSON config, stratified splitting, experiment metadata logging |

---

## 3. Dataset

### 3.1 Expected Structure
```
data/dataset/
├── Non_Demented/
├── Very_Mild_Demented/
├── Mild_Demented/
└── Moderate_Demented/
```

### 3.2 Dataset Inspection (Required Before Training)
Before any model training, the dataset must be inspected using:
```bash
python inspect_dataset.py
```
This reports:
- Per-class image counts
- Image dimension statistics
- Format distribution
- Corrupted/unreadable files
- Duplicate filenames or identical files across classes
- Class imbalance ratio
- Recommended train/val/test split sizes

### 3.3 Dataset Source
The dataset used in this project comes from the Kaggle **Alzheimer MRI Preprocessed Dataset** (or equivalent 4-class Brain MRI classification dataset). The dataset must be placed in `data/dataset/` with subdirectories matching the four class names above.

> The dataset is excluded from the Git repository by `.gitignore`. It must be
> downloaded separately by any user who clones this repository.

---

## 4. Pipeline Architecture

```
Input MRI Image
       │
       ▼
[ Input Validation ]
  - Format check (.jpg / .jpeg / .png)
  - Dimension range check
  - Pixel variance check (detect blank images)
       │
       ▼
[ Preprocessing ]
  - Resize to 224 × 224 (standard CNN input)
  - RGB conversion
  - Optional contour crop (remove black margins)
  - ImageNet mean/std normalization
       │
       ▼
[ Data Augmentation — Training split only ]
  - Random horizontal flip
  - Random rotation (±10°)
  - Random affine (slight translation & scale)
  - ColorJitter (brightness, contrast)
  ─ No augmentation on val/test ─
       │
       ▼
[ Deep Learning Model ]
  Candidate 1: MobileNetV2  (ImageNet pretrained → fine-tuned)
  Candidate 2: EfficientNet-B0 (ImageNet pretrained → fine-tuned)
  Candidate 3: ResNet-18    (ImageNet pretrained → fine-tuned)
       │
       ▼
[ Prediction & Class Probabilities ]
  - Softmax over 4 classes
  - Confidence bar chart
       │
       ▼
[ Grad-CAM Explainability ]
  - Gradients of predicted class score w.r.t. final conv layer activations
  - Heatmap overlaid on original scan
  ⚠ Heatmap shows regions contributing to model prediction.
  ⚠ NOT a validated clinical biomarker map.
       │
       ▼
[ Evaluation — on held-out test set ]
  - Accuracy, Precision, Recall, Specificity
  - F1-Score, Balanced Accuracy, MCC
  - Confusion Matrix (4×4)
```

---

## 5. Model Selection Strategy

The final model recommendation is determined **solely by experimental test-set results**, not by prior assumption.

**Selection criteria (priority order):**
1. Macro F1-Score (most robust to class imbalance)
2. Balanced Accuracy
3. Matthews Correlation Coefficient (MCC)
4. Inference latency per image (ms)
5. Parameter count

The comparison table will be populated after all three models are trained and evaluated.

---

## 6. Explainability — Scientific Framing

Grad-CAM highlights **image regions that had the highest gradient activation** relative to the predicted class score at the final convolutional layer.

**What this means:**
- It shows *where the model looked* when making a prediction.
- It is a model introspection tool, not a clinical measurement.

**What this does NOT mean:**
- Highlighted regions are not automatically confirmed anatomical biomarkers.
- The system does not diagnose hippocampal atrophy, ventricular dilation, or cortical thinning.
- Grad-CAM results have not been validated by radiologists.

---

## 7. Evaluation Metrics

All metrics are computed on the held-out **test set only** (not the validation set used during training).

| Metric | Description |
|:---|:---|
| Accuracy | Overall correct / total |
| Precision (Macro) | Mean precision across all 4 classes |
| Recall / Sensitivity (Macro) | Mean recall across all 4 classes |
| Recall per class | Individual class-level recall |
| Specificity (Macro) | Mean specificity across all 4 classes |
| F1-Score (Macro & Weighted) | Harmonic mean of precision and recall |
| Balanced Accuracy | Mean recall per class (equal class weight) |
| MCC | Matthews Correlation Coefficient (multi-class) |
| Confusion Matrix | 4×4 true vs predicted stage heatmap |

> Results will only appear in the application and documentation **after** actual model training and evaluation on the test set.

---

## 8. Reproducibility Checklist

- [ ] Fixed random seed: `42` (used in Python, NumPy, PyTorch)
- [ ] Stratified train/val/test split: `70% / 15% / 15%`
- [ ] Experiment configuration saved to `results/metrics/<model>_experiment_metadata.json`
- [ ] Training history (loss, accuracy per epoch) saved to `results/metrics/`
- [ ] Best checkpoint saved to `results/models/<model>_best.pt`
- [ ] Confusion matrix figure saved to `results/figures/`
- [ ] Dataset inspection report saved to `results/metrics/dataset_inspection.json`

---

## 9. Limitations

1. **2D axial slice classification only** — This system classifies individual 2D slices, not full volumetric MRI studies.
2. **Dataset dependency** — Performance is bounded by dataset quality, class balance, and acquisition variability.
3. **Single dataset** — No external validation set from a different hospital or scanner.
4. **Grad-CAM scope** — Visual explanations are not radiologist-validated.
5. **Not a clinical tool** — This system is a research prototype. It has not undergone clinical validation or regulatory review.

---

## 10. Ethical & Non-Clinical Disclaimer

> **IMPORTANT:**  
> This software is an **academic research prototype** built for computer science final-year project purposes only.  
> It is **not** a medical device, clinical diagnostic tool, or healthcare product.  
> Predictions generated by this system must **never** be used for clinical diagnosis, patient staging, treatment planning, or any medical decision.  
> Always consult a qualified healthcare professional or radiologist for clinical evaluation.
