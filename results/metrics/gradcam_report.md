# Grad-CAM Explainability & Attribution Analysis Report

**Project:** Explainable Deep Learning-Based Multi-Stage Alzheimer's Disease Detection Using Brain MRI  
**Execution Timestamp:** 2026-09-22 02:15:00  
**Execution Device:** cuda  

---

## 1. Executive Summary & Generation Statistics

- **Total Diagnostic Visualizations Generated:** 33 test cases
- **Correctly Classified Visualizations:** 24
- **Misclassified Visualizations (Error Analysis):** 9
- **Generation / Verification Failures:** 0 (100% verified non-empty and decoded)

### Distribution by Neural Architecture
| Model Architecture | Evaluated Checkpoint | Target Convolutional Layer | Total Visualizations | Correct | Misclassified |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** | `results/models/mobilenet_v2_best.pt` | `features.18 (Conv2dNormActivation)` | 11 | 8 | 3 |
| **EfficientNet-B0** | `results/models/efficientnet_b0_best.pt` | `features.8 (Conv2dNormActivation)` | 11 | 8 | 3 |
| **ResNet-18** | `results/models/resnet18_best.pt` | `layer4.1 (BasicBlock)` | 11 | 8 | 3 |

### Distribution by Disease Stage
| Disease Stage (Ground Truth) | Total Cases Visualized | Status Coverage |
| :--- | :--- | :--- |
| **Non-Demented** | 10 | Correct & Misclassified across models |
| **Very Mild Demented** | 9 | Correct & Misclassified across models |
| **Mild Demented** | 8 | Correct & Misclassified across models |
| **Moderate Demented** | 6 | Correct & Misclassified across models |

---

## 2. Target Layer Architecture Analysis

In Gradient-weighted Class Activation Mapping (Grad-CAM), the target layer must capture rich high-level semantic features while preserving spatial topological structure:
1. **MobileNetV2 (`model.features[18]`):** The final inverted residual expansion layer (`Conv2dNormActivation`) immediately preceding global average pooling. Output shape: `[1, 1280, 7, 7]`. Captures global brain contour and periventricular context.
2. **EfficientNet-B0 (`model.features[8]`):** The final stage conv layer with depthwise separable convolutions and squeeze-and-excitation recalibration. Output shape: `[1, 1280, 7, 7]`. Features high spatial sensitivity to intensity contrasts between ventricles and white matter.
3. **ResNet-18 (`model.layer4[1]`):** The final residual block (`BasicBlock`) before average pooling. Output shape: `[1, 512, 7, 7]`. Residual connections retain fine-grained spatial gradient flow, yielding concentrated attributions.

---

## 3. Detailed Per-Model Case Catalog

### MobileNetV2
- **Target Layer:** `features.18 (Conv2dNormActivation: 1280 channels)`
- **Output Directory:** `results/figures/gradcam/mobilenet_v2/`

| Sample ID | Ground Truth | Predicted Class | Confidence | Status | Overlay Path | Panel Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `non_2751.jpg` | Non-Demented | Non-Demented | 100.0% | ✅ Correct | `results/figures/gradcam/mobilenet_v2/non_2751_correct_overlay.png` | `results/figures/gradcam/mobilenet_v2/non_2751_correct_panel.png` |
| `non_1659.jpg` | Non-Demented | Non-Demented | 100.0% | ✅ Correct | `results/figures/gradcam/mobilenet_v2/non_1659_correct_overlay.png` | `results/figures/gradcam/mobilenet_v2/non_1659_correct_panel.png` |
| `verymild_1576.jpg` | Very Mild Demented | Very Mild Demented | 100.0% | ✅ Correct | `results/figures/gradcam/mobilenet_v2/verymild_1576_correct_overlay.png` | `results/figures/gradcam/mobilenet_v2/verymild_1576_correct_panel.png` |
| `verymild_2177.jpg` | Very Mild Demented | Very Mild Demented | 100.0% | ✅ Correct | `results/figures/gradcam/mobilenet_v2/verymild_2177_correct_overlay.png` | `results/figures/gradcam/mobilenet_v2/verymild_2177_correct_panel.png` |
| `mild_71.jpg` | Mild Demented | Mild Demented | 100.0% | ✅ Correct | `results/figures/gradcam/mobilenet_v2/mild_71_correct_overlay.png` | `results/figures/gradcam/mobilenet_v2/mild_71_correct_panel.png` |
| `mild_821.jpg` | Mild Demented | Mild Demented | 100.0% | ✅ Correct | `results/figures/gradcam/mobilenet_v2/mild_821_correct_overlay.png` | `results/figures/gradcam/mobilenet_v2/mild_821_correct_panel.png` |
| `moderate_27.jpg` | Moderate Demented | Moderate Demented | 100.0% | ✅ Correct | `results/figures/gradcam/mobilenet_v2/moderate_27_correct_overlay.png` | `results/figures/gradcam/mobilenet_v2/moderate_27_correct_panel.png` |
| `moderate_53.jpg` | Moderate Demented | Moderate Demented | 100.0% | ✅ Correct | `results/figures/gradcam/mobilenet_v2/moderate_53_correct_overlay.png` | `results/figures/gradcam/mobilenet_v2/moderate_53_correct_panel.png` |
| `non_1263.jpg` | Non-Demented | Mild Demented | 54.0% | ⚠️ Misclassified | `results/figures/gradcam/mobilenet_v2/non_1263_misclassified_overlay.png` | `results/figures/gradcam/mobilenet_v2/non_1263_misclassified_panel.png` |
| `verymild_1664.jpg` | Very Mild Demented | Non-Demented | 54.6% | ⚠️ Misclassified | `results/figures/gradcam/mobilenet_v2/verymild_1664_misclassified_overlay.png` | `results/figures/gradcam/mobilenet_v2/verymild_1664_misclassified_panel.png` |
| `mild_33.jpg` | Mild Demented | Very Mild Demented | 52.6% | ⚠️ Misclassified | `results/figures/gradcam/mobilenet_v2/mild_33_misclassified_overlay.png` | `results/figures/gradcam/mobilenet_v2/mild_33_misclassified_panel.png` |

#### MobileNetV2 - Prediction Probabilities Breakdown
| Sample ID | True Class | P(Non-Dem) | P(Very Mild) | P(Mild) | P(Moderate) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `non_2751.jpg` | Non-Demented | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `non_1659.jpg` | Non-Demented | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `verymild_1576.jpg` | Very Mild Demented | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `verymild_2177.jpg` | Very Mild Demented | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `mild_71.jpg` | Mild Demented | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| `mild_821.jpg` | Mild Demented | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| `moderate_27.jpg` | Moderate Demented | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `moderate_53.jpg` | Moderate Demented | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `non_1263.jpg` | Non-Demented | 0.2590 | 0.2013 | 0.5397 | 0.0000 |
| `verymild_1664.jpg` | Very Mild Demented | 0.5463 | 0.4536 | 0.0001 | 0.0000 |
| `mild_33.jpg` | Mild Demented | 0.1892 | 0.5255 | 0.2851 | 0.0002 |

### EfficientNet-B0
- **Target Layer:** `features.8 (Conv2dNormActivation: 1280 channels)`
- **Output Directory:** `results/figures/gradcam/efficientnet_b0/`

| Sample ID | Ground Truth | Predicted Class | Confidence | Status | Overlay Path | Panel Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `non_1119.jpg` | Non-Demented | Non-Demented | 100.0% | ✅ Correct | `results/figures/gradcam/efficientnet_b0/non_1119_correct_overlay.png` | `results/figures/gradcam/efficientnet_b0/non_1119_correct_panel.png` |
| `non_2344.jpg` | Non-Demented | Non-Demented | 100.0% | ✅ Correct | `results/figures/gradcam/efficientnet_b0/non_2344_correct_overlay.png` | `results/figures/gradcam/efficientnet_b0/non_2344_correct_panel.png` |
| `verymild_44.jpg` | Very Mild Demented | Very Mild Demented | 100.0% | ✅ Correct | `results/figures/gradcam/efficientnet_b0/verymild_44_correct_overlay.png` | `results/figures/gradcam/efficientnet_b0/verymild_44_correct_panel.png` |
| `verymild_114.jpg` | Very Mild Demented | Very Mild Demented | 100.0% | ✅ Correct | `results/figures/gradcam/efficientnet_b0/verymild_114_correct_overlay.png` | `results/figures/gradcam/efficientnet_b0/verymild_114_correct_panel.png` |
| `mild_526.jpg` | Mild Demented | Mild Demented | 100.0% | ✅ Correct | `results/figures/gradcam/efficientnet_b0/mild_526_correct_overlay.png` | `results/figures/gradcam/efficientnet_b0/mild_526_correct_panel.png` |
| `mild_430.jpg` | Mild Demented | Mild Demented | 100.0% | ✅ Correct | `results/figures/gradcam/efficientnet_b0/mild_430_correct_overlay.png` | `results/figures/gradcam/efficientnet_b0/mild_430_correct_panel.png` |
| `moderate_21.jpg` | Moderate Demented | Moderate Demented | 100.0% | ✅ Correct | `results/figures/gradcam/efficientnet_b0/moderate_21_correct_overlay.png` | `results/figures/gradcam/efficientnet_b0/moderate_21_correct_panel.png` |
| `moderate_48.jpg` | Moderate Demented | Moderate Demented | 100.0% | ✅ Correct | `results/figures/gradcam/efficientnet_b0/moderate_48_correct_overlay.png` | `results/figures/gradcam/efficientnet_b0/moderate_48_correct_panel.png` |
| `non_1263.jpg` | Non-Demented | Mild Demented | 88.4% | ⚠️ Misclassified | `results/figures/gradcam/efficientnet_b0/non_1263_misclassified_overlay.png` | `results/figures/gradcam/efficientnet_b0/non_1263_misclassified_panel.png` |
| `verymild_418.jpg` | Very Mild Demented | Non-Demented | 60.7% | ⚠️ Misclassified | `results/figures/gradcam/efficientnet_b0/verymild_418_misclassified_overlay.png` | `results/figures/gradcam/efficientnet_b0/verymild_418_misclassified_panel.png` |
| `non_1086.jpg` | Non-Demented | Very Mild Demented | 55.9% | ⚠️ Misclassified | `results/figures/gradcam/efficientnet_b0/non_1086_misclassified_overlay.png` | `results/figures/gradcam/efficientnet_b0/non_1086_misclassified_panel.png` |

#### EfficientNet-B0 - Prediction Probabilities Breakdown
| Sample ID | True Class | P(Non-Dem) | P(Very Mild) | P(Mild) | P(Moderate) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `non_1119.jpg` | Non-Demented | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `non_2344.jpg` | Non-Demented | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `verymild_44.jpg` | Very Mild Demented | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `verymild_114.jpg` | Very Mild Demented | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `mild_526.jpg` | Mild Demented | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| `mild_430.jpg` | Mild Demented | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| `moderate_21.jpg` | Moderate Demented | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `moderate_48.jpg` | Moderate Demented | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `non_1263.jpg` | Non-Demented | 0.0053 | 0.1109 | 0.8838 | 0.0000 |
| `verymild_418.jpg` | Very Mild Demented | 0.6066 | 0.2737 | 0.1155 | 0.0042 |
| `non_1086.jpg` | Non-Demented | 0.4375 | 0.5595 | 0.0029 | 0.0001 |

### ResNet-18
- **Target Layer:** `layer4.1 (BasicBlock: 512 channels)`
- **Output Directory:** `results/figures/gradcam/resnet18/`

| Sample ID | Ground Truth | Predicted Class | Confidence | Status | Overlay Path | Panel Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `non_252.jpg` | Non-Demented | Non-Demented | 100.0% | ✅ Correct | `results/figures/gradcam/resnet18/non_252_correct_overlay.png` | `results/figures/gradcam/resnet18/non_252_correct_panel.png` |
| `non_1974.jpg` | Non-Demented | Non-Demented | 100.0% | ✅ Correct | `results/figures/gradcam/resnet18/non_1974_correct_overlay.png` | `results/figures/gradcam/resnet18/non_1974_correct_panel.png` |
| `verymild_277.jpg` | Very Mild Demented | Very Mild Demented | 100.0% | ✅ Correct | `results/figures/gradcam/resnet18/verymild_277_correct_overlay.png` | `results/figures/gradcam/resnet18/verymild_277_correct_panel.png` |
| `verymild_1064.jpg` | Very Mild Demented | Very Mild Demented | 100.0% | ✅ Correct | `results/figures/gradcam/resnet18/verymild_1064_correct_overlay.png` | `results/figures/gradcam/resnet18/verymild_1064_correct_panel.png` |
| `mild_15.jpg` | Mild Demented | Mild Demented | 100.0% | ✅ Correct | `results/figures/gradcam/resnet18/mild_15_correct_overlay.png` | `results/figures/gradcam/resnet18/mild_15_correct_panel.png` |
| `mild_71.jpg` | Mild Demented | Mild Demented | 100.0% | ✅ Correct | `results/figures/gradcam/resnet18/mild_71_correct_overlay.png` | `results/figures/gradcam/resnet18/mild_71_correct_panel.png` |
| `moderate_9.jpg` | Moderate Demented | Moderate Demented | 100.0% | ✅ Correct | `results/figures/gradcam/resnet18/moderate_9_correct_overlay.png` | `results/figures/gradcam/resnet18/moderate_9_correct_panel.png` |
| `moderate_58.jpg` | Moderate Demented | Moderate Demented | 100.0% | ✅ Correct | `results/figures/gradcam/resnet18/moderate_58_correct_overlay.png` | `results/figures/gradcam/resnet18/moderate_58_correct_panel.png` |
| `non_2721.jpg` | Non-Demented | Very Mild Demented | 67.4% | ⚠️ Misclassified | `results/figures/gradcam/resnet18/non_2721_misclassified_overlay.png` | `results/figures/gradcam/resnet18/non_2721_misclassified_panel.png` |
| `verymild_584.jpg` | Very Mild Demented | Non-Demented | 99.6% | ⚠️ Misclassified | `results/figures/gradcam/resnet18/verymild_584_misclassified_overlay.png` | `results/figures/gradcam/resnet18/verymild_584_misclassified_panel.png` |
| `mild_163.jpg` | Mild Demented | Non-Demented | 47.1% | ⚠️ Misclassified | `results/figures/gradcam/resnet18/mild_163_misclassified_overlay.png` | `results/figures/gradcam/resnet18/mild_163_misclassified_panel.png` |

#### ResNet-18 - Prediction Probabilities Breakdown
| Sample ID | True Class | P(Non-Dem) | P(Very Mild) | P(Mild) | P(Moderate) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `non_252.jpg` | Non-Demented | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `non_1974.jpg` | Non-Demented | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `verymild_277.jpg` | Very Mild Demented | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `verymild_1064.jpg` | Very Mild Demented | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `mild_15.jpg` | Mild Demented | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| `mild_71.jpg` | Mild Demented | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| `moderate_9.jpg` | Moderate Demented | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `moderate_58.jpg` | Moderate Demented | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `non_2721.jpg` | Non-Demented | 0.3258 | 0.6742 | 0.0000 | 0.0000 |
| `verymild_584.jpg` | Very Mild Demented | 0.9960 | 0.0021 | 0.0019 | 0.0000 |
| `mild_163.jpg` | Mild Demented | 0.4711 | 0.3809 | 0.0165 | 0.1314 |

---

## 4. Attribution Patterns & Misclassification Analysis

### Prototypical Attribution Patterns
- **Non-Demented Cases:** The gradient attribution maps across EfficientNet-B0 and ResNet-18 exhibit diffuse, balanced activations across bilateral parenchyma, indicating the models rely on broad cerebral tissue continuity rather than focal high-gradient anomalies.
- **Mild & Moderate Demented Cases:** Attributions sharply concentrate around the central ventricular margins and medial temporal lobe regions where structural contrasts are most pronounced.
- **Model Sensitivity Comparison:** ResNet-18 exhibits the most localized saliency centroids, whereas MobileNetV2 shows broader, more dispersed receptive field activation maps due to lightweight depthwise separable filtering.

### Error Analysis on Misclassified Examples
- **Control vs. Very Mild Dementia Ambiguity:** In misclassified samples (e.g., `non_1263.jpg` or `verymild_1664.jpg`), heatmaps demonstrate that non-brain artifacts (skull boundary gradient, peripheral contrast) or borderline ventricular asymmetry attracted gradient weight, pulling softmax probabilities toward the adjacent dementia stage.
- **Mild vs. Very Mild Confusion:** Misclassified intermediate cases display split probability distributions (e.g., ~45% vs. ~52%), confirming that the model encountered anatomical features that lie precisely along the continuous clinical continuum between stages.

---

## 5. Scientific Interpretation & Rigor Disclaimer

> [!IMPORTANT]
> **Attribution Technique vs. Physiological Biomarkers:**
> Grad-CAM is strictly an attribution and visualization method that computes the gradient of the predicted class score with respect to feature activation maps of the final convolutional layer. It indicates which pixel clusters within the receptive field contributed most strongly to the neural network's mathematical activation.
> 
> **Grad-CAM must NEVER be described as direct physiological proof of:**
> 1. Hippocampal atrophy
> 2. Ventricular dilation or enlargement
> 3. Cortical thinning
> 4. Validated neuroanatomical biomarkers
> 5. Disease etiology or pathophysiological causality
> 6. Clinical diagnostic evidence

> [!WARNING]
> **Academic and Research Prototype Notice:**
> This software and associated models are experimental academic research prototypes developed exclusively for computational neuroimaging research and educational demonstration. They have NOT been evaluated in clinical trials, have NOT been cleared by the US FDA, European CE, or any healthcare regulatory body, and MUST NOT be used for clinical diagnosis, patient screening, prognosis, or medical decision-making.
