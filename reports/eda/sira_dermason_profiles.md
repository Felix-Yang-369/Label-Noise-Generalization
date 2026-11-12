# SIRA and DERMASON on the main training split

SIRA is the hardest class in the main experiment. With clean labels its test F1 is 0.869 for LR, 0.870 for RF and 0.852 for MLP. Its most common mistake is DERMASON: pooled over the three seeds, LR labels 8.0% of SIRA test beans as DERMASON and 6.1% of DERMASON beans as SIRA. We compared the two classes feature by feature on the 8,125 training rows (1,582 SIRA, 2,127 DERMASON). No validation or test rows were used.

## Quartiles

| Feature | SIRA median (Q1–Q3) | DERMASON median (Q1–Q3) | IQR overlap |
|---|---|---|---:|
| Area | 44,568 (41,685–47,709) | 31,882 (28,537–35,451) | 0 |
| MajorAxisLength | 298.0 (285.6–313.1) | 245.9 (231.4–260.8) | 0 |
| ShapeFactor2 | 0.00168 (0.00154–0.00181) | 0.00214 (0.00198–0.00235) | 0 |
| roundness | 0.887 (0.872–0.900) | 0.913 (0.897–0.925) | 0.06 |
| Compactness | 0.799 (0.782–0.814) | 0.820 (0.803–0.838) | 0.21 |
| AspectRation | 1.562 (1.502–1.627) | 1.484 (1.422–1.548) | 0.22 |
| Extent | 0.752 (0.710–0.789) | 0.755 (0.723–0.785) | 0.79 |
| Solidity | 0.9885 (0.9868–0.9898) | 0.9887 (0.9873–0.9899) | 0.81 |

IQR overlap is the length the two interquartile ranges share divided by the length they cover together: 0 means the middle halves of the two classes do not touch, and 1 means they are identical. The [full table](../../results/exploratory/main_train_sira_dermason.csv) has all 16 features.

## What it shows

- **Size separates the two classes most clearly.** For all six size features (Area, Perimeter, both axis lengths, ConvexArea and EquivDiameter) and for ShapeFactor1 and ShapeFactor2, the middle halves do not overlap. A typical SIRA bean has about 40% more area than a typical DERMASON bean.
- **The overlap is in the tails.** Only 1.6% of SIRA beans are smaller than DERMASON's upper quartile of Area, and 1.1% of DERMASON beans are larger than SIRA's lower quartile. On ShapeFactor2 these shares are 7.2% and 9.5%. The confused beans come from these tails, which quartiles alone cannot describe.
- **Shape features add little here.** SIRA is slightly more elongated (higher AspectRation and Eccentricity, lower roundness), but for AspectRation, Eccentricity, Compactness, ShapeFactor3 and ShapeFactor4 the middle halves overlap by about 20%. Roundness separates better (6% overlap), yet 31% of SIRA beans are rounder than DERMASON's lower quartile and 29% of DERMASON beans are less round than SIRA's upper quartile. Extent and Solidity are almost the same for both classes.

This matches the [feature selection notes](feature_selection_notes.md): the six highest-ranked features all measure size. It describes the training data only; it does not show which features the models actually rely on.

## Comparison with the notebook split

The same quartiles on the notebook's 9,480-row training split are in [notebook_train_sira_dermason.csv](../../results/exploratory/notebook_train_sira_dermason.csv). All 16 medians differ by less than 1% between the two splits, so the picture does not depend on which split is used.

## Rerun

From the repository root:

```bash
python -m exploratory.feature_profiles --split main > results/exploratory/main_train_sira_dermason.csv
python -m exploratory.feature_profiles --split notebook > results/exploratory/notebook_train_sira_dermason.csv
```
