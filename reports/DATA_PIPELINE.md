# Comprehensive Data Pipeline Report

**Project:** Explainable Deep Learning-Based Multi-Stage Alzheimer's Disease Detection Using Brain MRI  
**Stage:** Step 2 — Dataset Splitting, Preprocessing, Augmentation & Pipeline Verification  
**Target Architectures:** MobileNetV2, EfficientNet-B0, ResNet-18  

---

## 1. Dataset Source
- **Origin Platform:** Kaggle (`tourist55/alzheimers-dataset-4-class-of-images`).
- **Archive Preserved:** `data/raw/archive.zip` (59,143,418 bytes, ~56.40 MB).
- **Physical Dataset Path:** `data/dataset/` with canonical class directories (`Non_Demented`, `Very_Mild_Demented`, `Mild_Demented`, `Moderate_Demented`).
- **Image Specifications:** 128 × 128 pixels, 1 channel (Grayscale mode `'L'`), JPEG encoding (`.jpg`).

---

## 2. Original Dataset Size
The original downloaded ZIP archive (`archive.zip`) contained **12,800 entries** partitioned into two top-level directories:
- `Alzheimer_s Dataset/train/`: 6,400 images
- `Alzheimer_s Dataset/test/`: 6,400 images

---

## 3. Duplicate Test-Folder Discovery
During cryptographic SHA-256 analysis across all 12,800 entries:
- **Critical Finding:** Every single file in `Alzheimer_s Dataset/test/` is an exact, byte-for-byte duplicate of the corresponding file in `Alzheimer_s Dataset/train/` (6,400 common relative file names, 6,400 matching SHA-256 hashes, 100% duplicate rate).
- **Impact:** Evaluating models on the Kaggle-provided `test/` folder would yield artificial, deceptive test accuracy (100% data leakage).
- **Remediation:** The duplicate `test/` directory was discarded. Only the true unique 6,400 images were retained in `data/dataset/`.

---

## 4. Unique Dataset Size
The verified, authentic dataset contains exactly **6,400 unique images** with zero intra-dataset SHA-256 collisions:
- **Non-Demented:** 3,200 images (50.00%)
- **Very Mild Demented:** 2,240 images (35.00%)
- **Mild Demented:** 896 images (14.00%)
- **Moderate Demented:** 64 images (1.00%)
- **Total Unique Images:** **6,400 images** (100.00%)

---

## 5. Train / Validation / Test Strategy
To establish rigorous, uncompromised model evaluation, a deterministic stratified split was constructed directly from the unique 6,400 images:
- **Training Set (70%):** 4,480 images
- **Validation Set (15%):** 960 images
- **Held-out Test Set (15%):** 960 images
- **Random Seed:** Fixed to `42` (`random.Random("42_<class_name>")` per class partition).
- **Manifest-driven Architecture:** Split allocations are stored in version-controlled CSV manifests (`reports/splits/train.csv`, `validation.csv`, `test.csv`) rather than creating redundant physical disk copies.

---

## 6. Exact Class Distributions

| Class | Total Dataset | Train Set (70%) | Validation Set (15%) | Test Set (15%) |
| :--- | ---: | ---: | ---: | ---: |
| **Non-Demented** | 3,200 (50.00%) | 2,240 (50.00%) | 480 (50.00%) | 480 (50.00%) |
| **Very Mild Demented** | 2,240 (35.00%) | 1,568 (35.00%) | 336 (35.00%) | 336 (35.00%) |
| **Mild Demented** | 896 (14.00%) | 627 (13.99%) | 134 (13.96%) | 135 (14.06%) |
| **Moderate Demented** | 64 (1.00%) | 45 (1.01%) | 10 (1.04%) | 9 (0.94%) |
| **Total Images** | **6,400 (100%)** | **4,480 (100%)** | **960 (100%)** | **960 (100%)** |

---

## 7. Image Preprocessing Pipeline
- **Raw Input:** $128 \times 128 \times 1$ Grayscale JPEG.
- **Channel Expansion:** Converted to RGB ($3$ channels) to enable transfer learning on pretrained vision backbones.
- **Resolution Standardization:** Bilinear interpolation resizing to target dimensions $224 \times 224 \times 3$.
- **Tensor Conversion:** Converted to PyTorch `torch.FloatTensor` in dynamic range $[0.0, 1.0]$.
- **Statistical Normalization:** Normalized using standard ImageNet mean and standard deviation:
  - $\mu = [0.485, 0.456, 0.406]$
  - $\sigma = [0.229, 0.224, 0.225]$
- **Determinism:** Validation and test splits receive purely deterministic preprocessing with zero random perturbations.

---

## 8. Data Augmentation Strategy (Training Split Only)
Data augmentation is applied **exclusively on the training split** and generated dynamically on-the-fly inside `DataLoader`:
1. **Random Horizontal Flip ($p=0.5$):** Reflects bilateral symmetry of axial brain slices.
2. **Random Rotation ($\pm 10^\circ$):** Simulates slight patient head orientation angles within the scanner bore.
3. **Random Affine ($5\%$ translation, $0.95-1.05$ scaling):** Simulates subtle anatomical voxel shift and brain volume scale differences.
4. **ColorJitter ($10\%$ brightness, $10\%$ contrast):** Simulates scanner intensity nonuniformity and RF coil variation.
5. **Validation/Test Isolation:** Both validation and test pipelines strictly bypass stochastic augmentations.

---

## 9. Class Imbalance Strategy
Due to the extreme 50:1 imbalance between Non-Demented ($3,200$) and Moderate Demented ($64$):
- **Class-Weighted Cross-Entropy:** Inverse class frequencies are calculated from the training subset:
  $$w_c = \frac{N_{\text{train}}}{K \cdot N_{c,\text{train}}}$$
  - Non-Demented: $0.0717$
  - Very Mild Demented: $0.1024$
  - Mild Demented: $0.2562$
  - Moderate Demented: $3.5696$ (penalized $\approx 50\times$ more heavily to prevent minority class collapse)
- **Focal Loss Support:** Configured in `src/training/loss.py` ($\gamma=2.0$) as an alternative for hard-example focusing.
- **Fair Evaluation:** Models will be benchmarked on Balanced Accuracy, Macro F1-Score, and Matthews Correlation Coefficient (MCC), preventing standard accuracy distortion.

---

## 10. Data Leakage Prevention

| Check | Result | Verification Detail |
| :--- | :---: | :--- |
| **Train vs. Validation Hash Collision** | **0** | No SHA-256 hashes shared between Train and Validation |
| **Train vs. Test Hash Collision** | **0** | No SHA-256 hashes shared between Train and Test |
| **Validation vs. Test Hash Collision** | **0** | No SHA-256 hashes shared between Validation and Test |
| **Cross-Split Filepath Collision** | **0** | Manifests contain strictly disjoint sets of relative paths |
| **Augmentation Containment** | **PASSED** | Transforms verified: Validation difference across reads is $0.00000000$ |
| **Archive Duplication Separation** | **PASSED** | Only unique 6,400 scans indexed; duplicate test folder discarded |

---

## 11. Reproducibility Settings
- **Fixed Random Seed:** `42` across random generators and split partitioning.
- **Split Manifests:** Fixed CSV files stored in `reports/splits/` (`train.csv`, `validation.csv`, `test.csv`).
- **Configuration Master:** Settings documented in `experiment_config.json`.
- **Worker Configuration:** Set to single-process `num_workers=0` for deterministic Windows execution.

---

## 12. Limitations & Ethical Disclaimers
1. **Subject-Level Leakage Notice:**
   > "Subject-level leakage cannot be verified because patient/subject identifiers are not available."
   While slice-level separation is cryptographically guaranteed (0 duplicate hashes across splits), if the original dataset creator sampled adjacent 2D axial slices from the same subject, slices from one patient may appear in both train and test.
2. **2D Slice Scope:** The system analyzes 2D axial image slices, not full volumetric 3D MRI series.
3. **Academic Prototype:** This pipeline is engineered strictly for academic research and evaluation; it is not a clinical diagnostic system.
