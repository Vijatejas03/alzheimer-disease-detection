# Dataset Split Summary

## Split Strategy
- **Training Set:** 70% (4,480 images)
- **Validation Set:** 15% (960 images)
- **Test Set:** 15% (960 images)
- **Splitting Method:** Deterministic Stratified Split across all 4 dementia classes
- **Random Seed:** 42 (fixed for complete reproducibility)
- **Split Execution:** Performed prior to any data augmentation or model input

## Overall Distribution

| Class | Total Images | Percentage |
| :--- | ---: | ---: |
| **Non-Demented** | 3,200 | 50.00% |
| **Very Mild Demented** | 2,240 | 35.00% |
| **Mild Demented** | 896 | 14.00% |
| **Moderate Demented** | 64 | 1.00% |
| **Total** | **6,400** | **100.00%** |

## Train Distribution (70%)

| Class | Train Images | Split Proportion |
| :--- | ---: | ---: |
| Non-Demented | 2,240 | 50.00% |
| Very Mild Demented | 1,568 | 35.00% |
| Mild Demented | 627 | 14.00% |
| Moderate Demented | 45 | 1.00% |
| **Total Train** | **4,480** | **100.00%** |

## Validation Distribution (15%)

| Class | Validation Images | Split Proportion |
| :--- | ---: | ---: |
| Non-Demented | 480 | 50.00% |
| Very Mild Demented | 336 | 35.00% |
| Mild Demented | 134 | 13.96% |
| Moderate Demented | 10 | 1.04% |
| **Total Validation** | **960** | **100.00%** |

## Test Distribution (15%)

| Class | Test Images | Split Proportion |
| :--- | ---: | ---: |
| Non-Demented | 480 | 50.00% |
| Very Mild Demented | 336 | 35.00% |
| Mild Demented | 135 | 14.06% |
| Moderate Demented | 9 | 0.94% |
| **Total Test** | **960** | **100.00%** |

## Leakage Checks

| Check | Result | Details |
| :--- | :---: | :--- |
| **Train vs. Validation Hash Collision** | **0** | No SHA-256 hashes shared between Train and Validation |
| **Train vs. Test Hash Collision** | **0** | No SHA-256 hashes shared between Train and Test |
| **Validation vs. Test Hash Collision** | **0** | No SHA-256 hashes shared between Validation and Test |
| **Filename / Filepath Collision** | **0** | Every file path belongs strictly to exactly one split |
| **Augmentation Leakage Prevention** | **PASSED** | Validation and Test sets are strictly held out from data augmentation |
| **Archive Duplication Leakage** | **PASSED** | The redundant duplicate `test` directory from `archive.zip` was eliminated |

### Important Scientific Notice:
Subject-level leakage cannot be verified because patient/subject identifiers are not available in the public Kaggle dataset. Slice-level stratified independence has been rigorously enforced.
