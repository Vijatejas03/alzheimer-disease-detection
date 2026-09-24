# User Tutorial Walkthrough Script & Video Specification
## Alzheimer’s Disease Detection & Explainability System

**Video File:** [`reports/Alzheimer_Live_App_User_Tutorial.mp4`](file:///C:/Users/vijay/.gemini/antigravity/scratch/Alzheimer_Disease_Detection/reports/Alzheimer_Live_App_User_Tutorial.mp4)  
**Live Application URL:** [https://alzheimer-xai-vijay.streamlit.app](https://alzheimer-xai-vijay.streamlit.app)  
**Project:** Explainable Deep Learning-Based Multi-Stage Alzheimer’s Disease Detection Using Brain MRI  
**Team Leader:** Vijaytejas A C (`1VK23CS074`)  
**Institution:** Vivekananda Institute of Technology  

---

## 1. Video Technical Specifications

| Parameter | Value |
| :--- | :--- |
| **Duration** | **76.40 seconds** (1 minute 16 seconds) |
| **Resolution** | **1920 × 1080 (Full HD 1080p, 16:9)** |
| **Frame Rate** | 25.0 FPS |
| **Total Frames** | 1,910 frames |
| **Video Codec** | MP4 (mp4v / H.264 compatible) |
| **File Size** | 39.87 MB |
| **Visual Style** | **Real Live Screen Recording** with desktop browser chrome, smooth ease-in/ease-out animated OS cursor, click ripples, real scroll physics, and lower-third tutorial captions |
| **Recording Target** | Real live Streamlit Community Cloud instance (`https://alzheimer-xai-vijay.streamlit.app`) |

---

## 2. Complete Step-by-Step User Journey & Narration Script

### Scene 1 — Open the Project (00:00 – 00:05)
* **On-Screen Action:** The user opens a web browser, navigates to the address bar, and enters the project URL: `https://alzheimer-xai-vijay.streamlit.app`. The live cloud-hosted application connects and renders in full resolution.
* **On-Screen Caption:** `Step 1 — Open the Project | Navigating to https://alzheimer-xai-vijay.streamlit.app`
* **Suggested Voiceover / Narration:**  
  > *"To begin, open your web browser and navigate to our live application link at alzheimer-xai-vijay.streamlit.app. The platform loads instantly in any modern browser on desktop or mobile."*

---

### Scene 2 — Explore the Home Page (00:05 – 00:13)
* **On-Screen Action:** The live overview page is displayed. The mouse cursor navigates smoothly across the title lockup, system architecture badge, and four primary performance metric highlight cards (`6,400 MRI Scans`, `3 CNN Architectures`, `98.44% Accuracy`, `0.9991 ROC-AUC`). The view scrolls down smoothly to show the multi-stage pipeline flow, then returns to the navigation bar.
* **On-Screen Caption:** `Step 2 — Explore the Project | Home & System Overview`
* **Suggested Voiceover / Narration:**  
  > *"On the home page, you can review the research platform overview, dataset distribution, and high-level architectural metrics across our three fine-tuned neural networks."*

---

### Scene 3 — Open MRI Analysis (00:13 – 00:19)
* **On-Screen Action:** The cursor moves to the left sidebar navigation menu, hovers over `⊕ MRI Analysis`, and clicks the radio button with an animated cyan click ripple. The dedicated MRI Analysis workspace loads smoothly.
* **On-Screen Caption:** `Step 3 — Open MRI Analysis | Navigate to MRI Analysis in the sidebar`
* **Suggested Voiceover / Narration:**  
  > *"From the left navigation sidebar, click on 'MRI Analysis' to open the interactive neuroimaging diagnosis and explainability workspace."*

---

### Scene 4 — Upload a Brain MRI (00:19 – 00:26)
* **On-Screen Action:** The user moves to the file input component. The file chooser activates and uploads a verified clinical test scan: `data/test_samples/mild_33.jpg`. The application immediately ingests the file and renders the raw axial neuroimaging preview.
* **On-Screen Caption:** `Step 4 — Upload a Brain MRI | Scan loaded: mild_33.jpg (Axial Grayscale Neuroimaging)`
* **Suggested Voiceover / Narration:**  
  > *"Click the upload area to select an axial T1-weighted brain MRI scan from your device, or choose from the curated research benchmark samples."*

---

### Scene 5 — Input Quality Validation (00:26 – 00:32)
* **On-Screen Action:** The view scrolls down to section `02 Input Quality Validation`. The green status banner displays `STATUS: INPUT VALIDATION PASSED`. The metadata grid highlights verified scan dimensions, aspect ratio, chromatic divergence score, and dark background margins.
* **On-Screen Caption:** `Validation — Input Quality Validation Passed | Integrity, aspect ratio, background darkness, and contrast verified`
* **Suggested Voiceover / Narration:**  
  > *"The automated 8-stage input domain safeguard immediately verifies scan integrity, aspect ratio, and cranial geometry, blocking non-brain photos before any neural network execution."*

---

### Scene 6 & 7 — Select Model & Run Inference (00:32 – 00:40)
* **On-Screen Action:** The user scrolls to `04 Deep Learning Inference`. The cursor hovers over the architecture dropdown, selecting the primary model (with options for `ResNet18`, `EfficientNet-B0`, and `MobileNetV2`). The system executes PyTorch forward inference and softmax calculation in under 3 milliseconds.
* **On-Screen Caption:** `Step 5 & 6 — Select Model & Run Inference | Automated neural network forward pass & softmax probability computation`
* **Suggested Voiceover / Narration:**  
  > *"Select your desired CNN architecture—such as ResNet18, EfficientNet-B0, or MobileNetV2. Deep learning inference runs automatically to extract spatial representations."*

---

### Scene 8 — View Model Prediction (00:40 – 00:47)
* **On-Screen Action:** The view scrolls down to `05 Model Prediction`. The prediction card highlights the predicted dementia stage with calibrated confidence percentage, prediction latency, and 3-model agreement consensus indicator.
* **On-Screen Caption:** `Step 7 — View Model Prediction | Predicted Dementia Stage & Consensus Agreement`
* **Suggested Voiceover / Narration:**  
  > *"Section five displays the predicted clinical stage along with calibrated confidence and multi-model consensus across all three neural networks."*

---

### Scene 9 — View Class Probabilities (00:47 – 00:53)
* **On-Screen Action:** The cursor highlights the full 4-class multi-stage probability distribution chart showing exact softmax probabilities across:
  1. `Non-Demented`
  2. `Very Mild Demented`
  3. `Mild Demented`
  4. `Moderate Demented`
* **On-Screen Caption:** `Step 7 — Class Probability Distribution | Probabilities across Non-Demented, Very Mild, Mild, and Moderate stages`
* **Suggested Voiceover / Narration:**  
  > *"View the comprehensive probability distribution across all four diagnostic stages to evaluate model uncertainty."*

---

### Scene 10 — Grad-CAM Explainability (00:53 – 01:01)
* **On-Screen Action:** The view scrolls down to section `06 Model Attribution (Grad-CAM)`. The user inspects the three side-by-side visualization panels:
  - Original Preprocessed Brain MRI
  - Grad-CAM Heatmap (Jet Colormap)
  - Blended Alpha Overlay
  The color legend and scientific attribution disclaimer are highlighted.
* **On-Screen Caption:** `Step 8 — Grad-CAM Saliency Explanation | Warm colors (Red/Yellow) highlight regions driving neural network prediction`
* **Suggested Voiceover / Narration:**  
  > *"Section six provides visual explainability using Grad-CAM. Warmer colors highlight the anatomical regions that contributed most to the model's decision. Please note: Grad-CAM represents model attribution, not a definitive medical diagnosis."*

---

### Scene 11 — Model Comparison Page (01:01 – 01:07)
* **On-Screen Action:** The cursor moves to the sidebar and clicks `⊞ Model Comparison`. The held-out test set benchmark table appears, comparing Accuracy, Macro F1, Balanced Accuracy, Matthews Correlation Coefficient, ROC-AUC, and Inference Latency across MobileNetV2, EfficientNet-B0, and ResNet18.
* **On-Screen Caption:** `Step 9 — Compare Models | Held-Out Test Benchmarks: MobileNetV2 vs EfficientNet-B0 vs ResNet18`
* **Suggested Voiceover / Narration:**  
  > *"Navigate to 'Model Comparison' to examine verified held-out test benchmarks across our three architectures, with ResNet18 achieving 98.44% accuracy and EfficientNet-B0 achieving 99.03% balanced accuracy."*

---

### Scene 12 — Evaluation Results Page (01:07 – 01:13)
* **On-Screen Action:** The cursor clicks `▲ Evaluation`. The screen scrolls through per-class confusion matrices, one-vs-rest ROC curves, and reliability calibration curves evaluated on the 960-image held-out test split.
* **On-Screen Caption:** `Step 10 — Explore Evaluation Results | Confusion Matrices, Per-Class ROC Curves, and Calibration`
* **Suggested Voiceover / Narration:**  
  > *"The 'Evaluation' page presents full confusion matrices, multi-class ROC curves, and calibration diagrams demonstrating robust classification across all stages."*

---

### Scene 13 — Additional Research Pages (01:13 – 01:18)
* **On-Screen Action:** Rapid click-through across remaining research sections:
  - `⚡ Error & Robustness` (5 perturbation stress categories: blur, noise, contrast, brightness, rotation)
  - `≡ Methodology` (Dataset splitting, ImageNet standardization, weighted loss)
  - `ℹ About` (Team details, research disclosures, institutional affiliation)
* **On-Screen Caption:** `Research — Error & Robustness, Methodology, About | Testing model stability under blur, noise, and lighting shifts`
* **Suggested Voiceover / Narration:**  
  > *"You can also explore 'Error & Robustness' for empirical stress-testing, 'Methodology' for training specifications, and 'About' for project context."*

---

### Scene 14 — Complete User Journey Recap (01:18 – 01:24)
* **On-Screen Action:** High-contrast summary recap slide displaying the 10-step sequence, live project URL, team leader name, guide, and college details, followed by a smooth fade to black.
* **On-Screen Caption:** `Project Access: https://alzheimer-xai-vijay.streamlit.app | Team Leader: Vijaytejas A C`
* **Suggested Voiceover / Narration:**  
  > *"That completes the walkthrough of the Alzheimer’s Disease Detection and Explainability System. Scan the QR code or visit alzheimer-xai-vijay.streamlit.app to test the platform yourself."*

---

## 3. Project Integrity & Data Safety Audit

| Verification Check | Status | Evidence |
| :--- | :--- | :--- |
| **Live Web App Tested** | **VERIFIED** | Live Streamlit Community Cloud instance (`https://alzheimer-xai-vijay.streamlit.app`) interacted with in headless Chromium |
| **Real Test MRI Input Used** | **VERIFIED** | Ingested `data/test_samples/mild_33.jpg` (SHA-256 verified project test scan) |
| **Real Live Prediction Captured** | **VERIFIED** | Captured live inference response and Grad-CAM output generated by deployed PyTorch model |
| **No ML Code Modified** | **VERIFIED** | Zero modifications to model weights, architectures, train/val/test splits, or benchmark metrics |
| **No Fabricated Data / Screenshots**| **VERIFIED** | All browser viewports rendered directly from Chromium DOM snapshots |
| **Safety & Medical Disclaimers** | **VERIFIED** | Explicit disclaimers embedded: *"Model attribution ≠ clinical diagnosis"*, *"Academic research prototype"* |
| **Clean Git Status** | **VERIFIED** | Working tree clean on `main` branch |
