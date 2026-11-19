# Data Card: Dry Bean Dataset (as used in this project)

## Source

- **Dataset:** Dry Bean, UCI Machine Learning Repository, dataset ID 602, DOI 10.24432/C50S4B.
- **Original paper:** Koklu, M. and Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507. DOI 10.1016/j.compag.2020.105507.
- **License:** CC BY 4.0 (see the badge in the README).
- **Collection:** Images of dry beans were captured and segmented, and 16 geometric features were computed from each bean's shape.

## Contents

- 13,611 raw observations, 16 numerical predictors, 7 classes.
- Classes: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA.
- Predictors: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4.
- Column names are kept as published, including `AspectRation` and lowercase `roundness`.

## Raw file integrity

- No missing or nonfinite values.
- No feature-identical rows with conflicting labels.
- 68 exact duplicate rows, all in class HOROZ.
- The raw file is checked by hash in `tests/test_data_integrity.py`.

## Cleaning

- Keep the first occurrence of each exact duplicate. This leaves 13,543 distinct rows.
- Deduplication runs before splitting, so identical observations cannot appear in both training and evaluation data. HOROZ drops from 1,928 to 1,860 rows.
- No outlier removal, skewness transformation, PCA or feature deletion is applied. All 16 features are kept in the main pipeline.

## Split

- Test: 20% of the cleaned data, stratified, random state 171.
- Validation: 25% of the remainder, stratified, random state 172.
- Result: 8,125 training, 2,709 validation, 2,709 test observations.
- The same split is used for every experimental condition.

| Class | Cleaned | Train | Validation | Test |
|---|---:|---:|---:|---:|
| BARBUNYA | 1,322 | 793 | 264 | 265 |
| BOMBAY | 522 | 313 | 105 | 104 |
| CALI | 1,630 | 978 | 326 | 326 |
| DERMASON | 3,546 | 2,127 | 710 | 709 |
| HOROZ | 1,860 | 1,116 | 372 | 372 |
| SEKER | 2,027 | 1,216 | 405 | 406 |
| SIRA | 2,636 | 1,582 | 527 | 527 |
| **Total** | **13,543** | **8,125** | **2,709** | **2,709** |

## Known properties relevant to modeling

- **Class imbalance:** DERMASON has 2,127 training rows and BOMBAY has 313, a ratio of 6.8. Accuracy is therefore dominated by the large classes, so Macro-F1 is the primary metric.
- **Collinearity:** Several predictors are nearly collinear. Area and ConvexArea have Pearson r = 0.9999, Compactness and ShapeFactor3 r = 0.9987, and Perimeter and EquivDiameter r = 0.9914. The 16 columns carry fewer than 16 independent pieces of information, so individual LR coefficients should not be read causally.
- **Scale:** Feature standard deviations differ by several orders of magnitude. LR and the MLP use a StandardScaler fitted on training inputs only. RF uses raw values.
- **Exploratory analysis** uses only the training partition.

## Label randomization (study manipulation)

- Only training labels are changed. Validation and test labels are never changed.
- With probability p, a training label is replaced by a uniform draw from all seven classes. The draw can return the original class, so the expected fraction of changed labels is 6p/7.
- The same random draws are reused across models for a given seed.

## Intended use

Research and teaching on classification, label noise and evaluation methodology. The data describe geometric measurements of bean images from one collection. They should not be used to make claims about other bean varieties, growing conditions or imaging setups.

## Limitations

- One dataset, one fixed split per seed, three seeds per condition.
- Duplicates were removed, so results are not directly comparable to the cross-validated accuracy in the original paper.
- The features are strongly collinear, so feature-level conclusions are limited.
