# Test Macro-F1 at 75% label redraw

At each seed, logistic regression had the smallest drop in test Macro-F1 when the training-label redraw probability increased from 0 to 0.75. The table shows the clean-label baseline score minus the score at 0.75 for the same model and seed. Both scores use the original test labels.

| Model | Seed 17 | Seed 29 | Seed 43 | Mean drop |
|---|---:|---:|---:|---:|
| Logistic regression | 0.0284 | 0.0337 | 0.0372 | 0.0331 |
| Random forest | 0.2950 | 0.3120 | 0.2694 | 0.2921 |
| MLP | 0.3367 | 0.3932 | 0.3171 | 0.3490 |

The MLP lost the most at every seed. Its largest drop was 0.3932 at seed 29. All three seeds used the same 2,709 test beans, so this comparison covers one data split.

Source: [run-level results at 8d2653d](https://github.com/Felix-Yang-369/Label-Noise-Generalization/blob/8d2653dfaafb571139511d7c488bb48aef8509d2/results/test_metrics.csv).

To reproduce the table from a repository checkout:

    python3 reports/experiments/check_noise_results.py results
