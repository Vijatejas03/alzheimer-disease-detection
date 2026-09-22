# Held-Out Test Evaluation & Multi-Model Comparative Report

**Project:** Explainable Deep Learning-Based Multi-Stage Alzheimer's Disease Detection Using Brain MRI  
**Evaluation Date:** 2026-09-22 02:02:18  
**Hardware Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (4095.5 MiB VRAM)  
**Held-Out Test Sample Count:** Exactly 960 images (quarantined during all training runs)  

---

## 1. Executive Summary Table

| Model Architecture | Parameters | Accuracy | Macro Precision | Macro Recall | Macro F1 | Balanced Acc | MCC | Multiclass ROC-AUC | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | 2,228,996 | **93.13%** | 0.9444 | 0.9425 | **0.9434** | 0.9425 | 0.8868 | **0.9936** | 2.93 ms |
| **EfficientNet-B0** | 4,012,672 | **98.23%** | 0.9851 | 0.9903 | **0.9876** | 0.9903 | 0.9711 | **0.9989** | 2.20 ms |
| **ResNet-18** | 11,178,564 | **98.44%** | 0.9893 | 0.9829 | **0.9860** | 0.9829 | 0.9743 | **0.9991** | 1.77 ms |

---

## 2. Detailed Per-Class Breakdown (Held-Out Test Set)

### MobileNetV2

| Class Stage | Support | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC (OvR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Non-Demented (Control)** | 480 | 0.9457 | 0.9437 | 0.9458 | 0.9447 | 0.9886 |
| **Very Mild Demented (Early Stage)** | 336 | 0.9145 | 0.9226 | 0.9535 | 0.9185 | 0.9888 |
| **Mild Demented (Intermediate Stage)** | 135 | 0.9173 | 0.9037 | 0.9867 | 0.9104 | 0.9970 |
| **Moderate Demented (Advanced Stage)** | 9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### EfficientNet-B0

| Class Stage | Support | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC (OvR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Non-Demented (Control)** | 480 | 0.9915 | 0.9729 | 0.9917 | 0.9821 | 0.9979 |
| **Very Mild Demented (Early Stage)** | 336 | 0.9708 | 0.9881 | 0.9840 | 0.9794 | 0.9979 |
| **Mild Demented (Intermediate Stage)** | 135 | 0.9783 | 1.0000 | 0.9964 | 0.9890 | 0.9999 |
| **Moderate Demented (Advanced Stage)** | 9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### ResNet-18

| Class Stage | Support | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC (OvR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Non-Demented (Control)** | 480 | 0.9855 | 0.9938 | 0.9854 | 0.9896 | 0.9988 |
| **Very Mild Demented (Early Stage)** | 336 | 0.9792 | 0.9821 | 0.9888 | 0.9807 | 0.9979 |
| **Mild Demented (Intermediate Stage)** | 135 | 0.9923 | 0.9556 | 0.9988 | 0.9736 | 0.9997 |
| **Moderate Demented (Advanced Stage)** | 9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## 3. Confusion Matrices

#### MobileNetV2 Confusion Matrix (Row: True, Col: Pred)
```
[[453,  19,   8,   0],
 [ 23, 310,   3,   0],
 [  3,  10, 122,   0],
 [  0,   0,   0,   9]]
```

#### EfficientNet-B0 Confusion Matrix (Row: True, Col: Pred)
```
[[467,  10,   3,   0],
 [  4, 332,   0,   0],
 [  0,   0, 135,   0],
 [  0,   0,   0,   9]]
```

#### ResNet-18 Confusion Matrix (Row: True, Col: Pred)
```
[[477,   3,   0,   0],
 [  5, 330,   1,   0],
 [  2,   4, 129,   0],
 [  0,   0,   0,   9]]
```

---

## 4. Academic & Research Disclaimer

> **Notice:** This project is an academic and research prototype developed for multi-stage dementia stage detection from structural MRI. It is NOT clinically certified, medically approved, or intended for independent medical diagnosis.
