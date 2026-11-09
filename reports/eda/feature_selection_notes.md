# Feature selection on the Dry Bean data

Area and ConvexArea have a training correlation of 0.99994. With measurements this similar, we wanted to see how many features we could drop before validation performance changed.

We ranked the features by their ANOVA F-scores on the training rows. For each model, we tried the top 4 through 16 features in steps of two. We also tried walking down the same ranking and skipping a feature if its absolute correlation with one we had kept exceeded 0.95. That left 10 features. We used clean labels, seed 17 and the settings from the main experiment.

## Validation macro-F1

| Features | LR | RF | MLP |
|---|---:|---:|---:|
| Top 4 | 0.7975 | 0.8181 | 0.8292 |
| Top 6 | 0.9109 | 0.9118 | 0.9214 |
| Top 8 | 0.9138 | 0.9100 | 0.9270 |
| Top 10 | 0.9138 | 0.9079 | 0.9303 |
| Top 12 | 0.9151 | 0.9047 | 0.9292 |
| Top 14 | 0.9292 | 0.9180 | 0.9340 |
| All 16 | 0.9328 | 0.9241 | 0.9338 |
| Correlation-pruned (10) | 0.9340 | 0.9294 | 0.9297 |

The six highest-ranked features all measure size. For the random forest, adding the next four features brought the score from 0.9118 to 0.9079. At the same feature count, pruning highly correlated measurements gave the forest 0.9294. That set kept shape measurements such as eccentricity and roundness.

Compared with all 16 features, the pruned set scored slightly higher for LR and RF and slightly lower for MLP. These are small differences on one validation split. The model settings were originally tuned with all 16 features, which we kept for the label-noise experiment.

The [recorded scores](../../results/exploratory/feature_selection.csv) and [ANOVA ranking](../../results/exploratory/feature_anova_scores.csv) come from a run with a newer scikit-learn version than the project's pinned 1.7.2. To rerun the comparison from the repository root:

```bash
python -m exploratory.feature_selection
```
