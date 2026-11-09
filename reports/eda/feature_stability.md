# Feature selection across the two training splits

Would the 10-feature subset from our earlier analysis change on the notebook split? We applied the same rule to each training set: rank all 16 features by ANOVA F-score, then skip a feature if its absolute Pearson correlation with an already selected feature exceeds the cutoff.

| Correlation cutoff | Main training set (8,125) | Notebook training set (9,480) | Features in both subsets |
|---|---:|---:|---:|
| 0.90 | 8 | 8 | 8 |
| 0.95 | 10 | 10 | 10 |
| 0.98 | 12 | 12 | 12 |

All 16 features had the same rank on both splits. At 0.95, both kept Area, MajorAxisLength, ShapeFactor2, ShapeFactor1, AspectRation, Eccentricity, roundness, ShapeFactor4, Solidity and Extent. Changing the cutoff affected which measurements survived. For example, Eccentricity survived at 0.95 but dropped out at 0.98. The algorithm checks each feature against those it has already kept, and that list changes with the cutoff.

The two training sets share 5,687 beans, so this comparison covers two overlapping samples from the same dataset. It describes the feature selection rule; it does not estimate how a reduced model would score. The notebook's model-selection results are separate.

Run `python -m exploratory.feature_stability` from the repository root to regenerate the [feature-by-feature results](../../results/exploratory/feature_stability.csv).
