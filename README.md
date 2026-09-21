# Explainable Deep Learning-Based Multi-Stage Alzheimer’s Disease Detection Using Brain MRI

## 🎓 Final-Year Major Project Overview
This project presents an **Explainable Deep Learning framework** for the multi-stage classification of Alzheimer’s Disease from structural Brain Magnetic Resonance Imaging (MRI) scans. The system incorporates comparative candidate architecture benchmarking, class-imbalance mitigation, comprehensive statistical evaluation, and visual interpretability via **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

---

## 📌 Project Highlights & Research Objectives
- **4-Stage Classification** (class labels as defined by the dataset source):
  - Non-Demented
  - Very Mild Demented
  - Mild Demented
  - Moderate Demented
- **Candidate Architectures**: Controlled comparative benchmarking across **MobileNet**, **EfficientNet**, and **ResNet** (final selection strictly determined by empirical test performance).
- **Explainable AI (XAI)**: Generates Grad-CAM visual heatmaps showing image regions that contributed to the model's prediction. These are model explanations and must not be interpreted as clinically validated biomarkers.
- **Imbalance Handling**: Class-weighted cross-entropy loss and medical data augmentations.
- **Evaluation Suite**: Rigorous evaluation using Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, Balanced Accuracy, Matthews Correlation Coefficient (MCC), and Confusion Matrices.
- **Interactive Web Interface**: Streamlit-based interface for MRI upload, model inference, Grad-CAM visualization, and dataset exploration.
- **Academic Rigor**: Strict non-clinical disclaimer; no fabricated metrics or clinical claims.

---

## 🔬 Literature Gap & Our Proposed Improvements

| Aspect | Senior / Baseline Projects | Our Proposed Improvements |
| :--- | :--- | :--- |
| **Model Interpretability** | Black-box models with no visual justification | **Grad-CAM visual heatmaps** showing which image regions contributed to the model's prediction. Highlighted regions are model explanations and must not be interpreted as confirmed anatomical biomarkers. |
| **Class Imbalance** | Ignored, causing bias toward majority classes | **Class-weighted loss** & stratified augmentation pipelines |
| **Model Comparison** | Arbitrarily chooses one architecture | Controlled empirical study comparing **MobileNet**, **EfficientNet**, and **ResNet** |
| **Evaluation Metrics** | Standard overall accuracy only | Comprehensive metrics: **Balanced Accuracy, Specificity, MCC, Macro-F1, Confusion Matrix** |
| **Integrity & Framing** | Often claims medical diagnosis | **Academic / Research prototype** with explicit ethical and non-clinical disclaimers |

---

## 🏗️ System Architecture & Pipeline

`
MRI Scan
   │
   ▼
[ Input Validation ] ──► Format, dimensions, pixel distribution checks
   │
   ▼
[ Preprocessing ] ────► Normalization, bounding-box cropping, resizing (224x224)
   │
   ▼
[ Augmentation & Imbalance Handling ] ──► Class weights, affine & contrast transforms
   │
   ▼
[ Deep Learning Candidate Models ] ──► MobileNet / EfficientNet / ResNet
   │
   ▼
[ Prediction & Probabilities ] ──► Multi-class Softmax across 4 dementia stages
   │
   ▼
[ Grad-CAM Explainability ] ────► Gradient-weighted feature map heatmaps overlaid on scan
   │
   ▼
[ Comprehensive Evaluation ] ───► Accuracy, Precision, Sensitivity, Specificity, MCC, Confusion Matrix
   │
   ▼
[ Streamlit Web Application ] ──► Interactive UI for research evaluation and visual exploration
`

---

## 📂 Project Directory Structure

`
Alzheimer_Disease_Detection/
├── data/
│   ├── dataset/                        # Place dataset here in 4 class subdirectories
│   │   ├── Non_Demented/
│   │   ├── Very_Mild_Demented/
│   │   ├── Mild_Demented/
│   │   └── Moderate_Demented/
│   ├── raw/                            # Directory for downloaded archives or raw scans
│   └── test_samples/                   # Reference test images for demonstration
├── src/
│   ├── data/
│   │   ├── validation.py               # Input validation & dataset integrity checker
│   │   ├── preprocessing.py            # Image transforms, resizing, normalization
│   │   ├── augmentation.py             # Safe medical data augmentation pipelines
│   │   └── dataset.py                  # PyTorch Dataset loader & class weights calculator
│   ├── models/
│   │   ├── architectures.py            # MobileNet, EfficientNet, and ResNet definitions
│   │   └── model_factory.py            # Unified model loading & checkpoint management
│   ├── training/
│   │   ├── loss.py                     # Class-weighted Cross-Entropy & focal loss
│   │   └── train.py                    # Modular PyTorch training & validation loops
│   ├── evaluation/
│   │   ├── metrics.py                  # Full metric suite (Acc, Prec, Rec, Spec, F1, BalAcc, MCC)
│   │   └── evaluate.py                 # Multi-model test set evaluation & benchmark logger
│   ├── explainability/
│   │   └── gradcam.py                  # Grad-CAM heatmap extraction and colormap overlay
│   └── utils/
│       ├── academic_report.py          # Academic/Research prediction summary generator
│       └── logger.py                   # Structured console and file logger
├── app/
│   ├── app.py                          # Main Streamlit Web Application
│   ├── components/                     # Modular UI tabs and subcomponents
│   └── styles/                         # Custom UI styling
├── results/
│   ├── models/                         # Saved model checkpoints (.pt / .pth)
│   ├── metrics/                        # Computed test metrics JSON and logs
│   └── figures/                        # Confusion matrix charts and training curves
├── tests/                              # Pytest unit tests for all modules
├── requirements.txt                    # Project Python dependencies
├── run_app.bat                         # One-click Windows launcher for Streamlit UI
└── README.md                           # Project documentation
`

---

## 📦 Setup & Installation

### 1. Clone or Open Workspace
Open the workspace directory in VS Code or your terminal:
`ash
cd C:\Users\vijay\.gemini\antigravity\scratch\Alzheimer_Disease_Detection
`

### 2. Install Required Dependencies
`ash
pip install -r requirements.txt
`

### 3. Placing the Dataset
Place your MRI image dataset in the data/dataset/ directory organized by class:
- data/dataset/Non_Demented/
- data/dataset/Very_Mild_Demented/
- data/dataset/Mild_Demented/
- data/dataset/Moderate_Demented/

Supported image formats: .jpg, .jpeg, .png.

---

## 🚀 Running the Project

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Inspect the Dataset (REQUIRED before training)
```bash
python inspect_dataset.py
# Or double-click run_inspect.bat on Windows
```

### Step 3 — Train Models (after dataset is confirmed)
```bash
# Configured via experiment_config.json
python src/training/train.py
```

### Step 4 — Launch the Web Application
```bash
streamlit run app/app.py
```

### Running Unit Tests
```bash
pytest tests/
```

---

## 📄 Documentation Files

| File | Purpose |
|:---|:---|
| [`README.md`](README.md) | Project overview and quick-start |
| [`METHODOLOGY.md`](METHODOLOGY.md) | Full methodology, baseline comparison, pipeline details |
| [`GIT_SETUP.md`](GIT_SETUP.md) | Git installation, initialisation, and GitHub push guide |
| [`experiment_config.json`](experiment_config.json) | Default training hyperparameters (editable) |
| [`inspect_dataset.py`](inspect_dataset.py) | Dataset integrity and imbalance analysis tool |

---

## ⚠️ Academic & Non-Clinical Disclaimer
> **IMPORTANT NOTICE**: This software is an **academic engineering research project** developed solely for educational, experimental, and scientific investigation purposes. It is **NOT** a certified medical device, clinical diagnostic instrument, or healthcare product. Predictions generated by this system must **never** be used for clinical diagnosis, patient staging, treatment planning, or medical decisions.
