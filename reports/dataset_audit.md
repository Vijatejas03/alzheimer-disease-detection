# Dataset Audit

## Dataset Source
- **Origin / Source Platform:** Kaggle (`tourist55/alzheimers-dataset-4-class-of-images`)
- **Original Archive Filename:** `archive.zip`
- **Location on Disk:** `C:\Users\vijay\Downloads\archive.zip` (copied into `data\raw\archive.zip`)
- **Provenance / Medical Origin:** The evaluated dataset is the downloaded four-class Alzheimer's MRI image collection used in this study. Dataset provenance and subject-level metadata are limited. External clinical and scanner provenance cannot be independently verified.

---

## ZIP Structure
Inspection of `archive.zip` revealed 12,800 entries organized into two predefined top-level directories (`train` and `test`), each containing the four class subfolders:

```
archive.zip
└── Alzheimer_s Dataset/
    ├── test/
    │   ├── MildDemented/           (896 JPEG images)
    │   ├── ModerateDemented/       (64 JPEG images)
    │   ├── NonDemented/            (3,200 JPEG images)
    │   └── VeryMildDemented/       (2,240 JPEG images)
    └── train/
        ├── MildDemented/           (896 JPEG images)
        ├── ModerateDemented/       (64 JPEG images)
        ├── NonDemented/            (3,200 JPEG images)
        └── VeryMildDemented/       (2,240 JPEG images)
```

### Critical Finding on ZIP Internal Duplication:
A complete SHA-256 cryptographic hash verification across all 12,800 files established that **every single image in `Alzheimer_s Dataset/test/` is an exact, byte-for-byte duplicate of the corresponding image in `Alzheimer_s Dataset/train/`** (6,400 common relative filenames, 6,400 identical SHA-256 matches, 100% duplicate rate).

To prevent complete train/test data leakage, the duplicate `test` directory was disregarded, and the single unique set of **6,400 authentic MRI scans** was extracted into the canonical project directory:
```
data\dataset\
├── Non_Demented\           (3,200 images)
├── Very_Mild_Demented\      (2,240 images)
├── Mild_Demented\           (896 images)
└── Moderate_Demented\       (64 images)
```

---

## Class Distribution

| Class | Actual Image Count | Class Proportion (%) |
| :--- | ---: | ---: |
| **Non-Demented** | 3,200 | 50.00% |
| **Very Mild Demented** | 2,240 | 35.00% |
| **Mild Demented** | 896 | 14.00% |
| **Moderate Demented** | 64 | 1.00% |
| **Total** | **6,400** | **100.00%** |

*(Counts are 100% physically verified on extracted image files in `data\dataset\` and logged in `reports\dataset_manifest.csv`).*

---

## Image Properties
- **Dimensions:** Exactly $128 \times 128$ pixels for 100% of images (min: 128, max: 128, mean: 128.0, standard deviation: 0.0).
- **Color Channels:** 1 channel (Grayscale, mode `'L'`).
- **File Extensions:** 100% `.jpg` (JPEG standard encoding).
- **File Size Distribution:**
  - Minimum file size: 3,343 bytes (~3.26 KB)
  - Maximum file size: 4,947 bytes (~4.83 KB)
  - Mean file size: 4,392.8 bytes (~4.29 KB)
  - Median file size: 4,461.0 bytes (~4.36 KB)
  - Total dataset storage: 28,114,136 bytes (~26.81 MB)

---

## Data Quality
- **Corrupted / Unreadable Images:** **0** (All 6,400 images decoded successfully with PIL verification).
- **Duplicate Images (Within Extracted Dataset):** **0** (All 6,400 files possess distinct, unique SHA-256 cryptographic hashes).
- **Duplicate Images (Within Raw Archive):** **6,400** duplicate pairs caused by the raw archive's redundant `train/` and `test/` folders.
- **Suspicious Files:** None (no hidden desktop files, thumbs.db, or non-image assets).
- **Empty Folders:** None (all 4 class folders are properly populated).
- **Pre-existing Augmentations:** Visual and histogram inspection indicates standard axial slice skull-stripping/centering. Slices appear naturally acquired, but lack rotation/affine artifacts.

---

## Subject-Level Information
- **Subject / Patient Identifiers:** Not available in image metadata, EXIF tags, or filenames (filenames follow synthetic sequential conventions such as `nonDem0.jpg`, `verymild0.jpg`, `mild_10.jpg`, `moderate_10.jpg`).
- **Data Leakage Assessment:**
  > "Subject-level leakage cannot be verified because patient/subject identifiers are not available."

---

## Class Imbalance
The dataset exhibits significant natural multi-class imbalance:
- **Non-Demented (Majority Class):** 50.00% (3,200 samples)
- **Very Mild Demented:** 35.00% (2,240 samples)
- **Mild Demented:** 14.00% (896 samples)
- **Moderate Demented (Extreme Minority Class):** 1.00% (64 samples)
- **Imbalance Ratio ($\text{Max} / \text{Min}$):** $3200 / 64 = \mathbf{50.0:1}$

This extreme 50:1 imbalance confirms the necessity of:
1. **Class-Weighted Cross-Entropy Loss** ($w_c = \frac{N}{K \cdot N_c}$).
2. **Stratified Splitting** across train, validation, and test sets.
3. **Imbalance-robust evaluation metrics** (Macro F1-Score, Balanced Accuracy, Specificity, and Matthews Correlation Coefficient).

---

## Dataset Assessment
- **Compatibility:** **ACCEPTED.** The downloaded dataset matches our required four-class Alzheimer’s MRI project specifications.
- **Class Alignment:** Perfectly maps to our four canonical targets: *Non-Demented*, *Very Mild Demented*, *Mild Demented*, and *Moderate Demented*.
- **Integrity:** Every slice is readable, standardized in resolution ($128 \times 128$), and completely free from corruptions.

---

## Recommended Next Step
Do **NOT** train models yet.

The next stage is:
> **"dataset split + preprocessing + controlled augmentation + model training"**
