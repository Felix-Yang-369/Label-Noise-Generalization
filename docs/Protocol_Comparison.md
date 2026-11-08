# Protocol comparison: current `main` experiment vs. the selected Group 8 notebook

This note compares the recorded experiment on `main` (`src/`, `config/`, `results/`) with the
team-selected notebook `integration/elsey/ECS171_Group8_Project.ipynb`. It is documentation only:
it changes no scientific source, configuration, result, figure, paper text or deployed model.

Sources: `results/model_noise_summary.csv` for `main`; the notebook's own saved cell outputs for
the notebook. The notebook's 45-run experiment has **not** been re-run under this repository's
locked environment and integrity checks.

## Shared design

- UCI Dry Bean, 68 exact duplicates removed, 13,543 rows, 16 features, 7 classes.
- Logistic regression, random forest (200 trees) and an MLP with hidden layers (128, 64).
- Replacement probability p ∈ {0, 0.25, 0.5, 0.75, 1}; seeds 17, 29, 43; 45 main fits.
- Each training label is replaced with probability p by a uniform draw over all seven classes,
  which can return the original class. Validation and test labels stay clean.
- Hyperparameters are chosen on clean-label validation Macro-F1.

## Differences

| Setting | `main` | Notebook |
|---|---|---|
| Train / validation / test | 8,125 / 2,709 / 2,709 (seeds 171, 172) | 9,480 / 2,031 / 2,032 (seed 17) |
| Feature selection | none | `SelectKBest` with k ∈ {10, 16}; k = 16 selected (no reduction) |
| LR | C = 1, `max_iter` 2000 | C = 10, `max_iter` 800 |
| RF depth | grid {8, unlimited}; **unlimited** selected | grid {12, unlimited}; **12** selected |
| MLP training budget | up to 500 epochs, tol 1e-5, 20-epoch patience | **fixed 180 epochs** |
| MLP alpha | 0.01 | 0.01 |
| Tuning candidates | 10 | 18 |
| Convergence warnings | recorded (14 main runs) | suppressed |
| Test set | sealed until all choices are frozen | scored in every run (not used for selection) |
| EDA scope | training split only | full cleaned data (class plot uses pre-deduplication data) |
| Data and environment | hashed raw files, locked dependencies | downloaded at run time, no hash or version lock |

## Results (mean over three seeds)

| Model | p | Test Macro-F1, `main` / notebook | Accuracy on noisy training labels, `main` / notebook |
|---|---:|---|---|
| LR | 0 | 0.940 / 0.942 | 0.925 / 0.926 |
| LR | 0.75 | 0.907 / 0.908 | 0.327 / 0.329 |
| LR | 1 | 0.114 / 0.098 | 0.160 / 0.160 |
| RF | 0 | 0.936 / 0.933 | 1.000 / 0.985 |
| RF | 0.75 | 0.644 / 0.849 | 1.000 / 0.584 |
| RF | 1 | 0.131 / 0.122 | 1.000 / 0.846 |
| MLP | 0 | 0.935 / 0.944 | 0.971 / 0.950 |
| MLP | 0.75 | 0.586 / 0.796 | 0.437 / 0.364 |
| MLP | 1 | 0.131 / 0.126 | 0.381 / 0.275 |

## Reading the comparison

- Clean-label performance agrees within about 0.01 for every model, so the split difference has
  little effect on baseline accuracy.
- Logistic regression is almost identical in both experiments.
- The large differences at p = 0.75 and p = 1 follow the capacity settings. The unlimited-depth
  forest fits every randomized training label; the depth-12 forest fits about 85% of them at
  p = 1. The fixed 180-epoch MLP acts like early stopping. Both constrained models keep higher
  clean-test Macro-F1 at p = 0.75.
- Both experiments support the same qualitative claim: fitting corrupted training labels does not
  imply generalization. They differ on how completely each model memorizes.

## Demo-bundle reproduction check

`integration/elsey/export_demo_models.py` rebuilds the six seed-17 demo pipelines (p = 0 and
p = 1) with the notebook's split, hyperparameters and label-resampling code. Run with the
locked dependencies under Python 3.13, its training-label accuracies match the notebook's printed
seed-17 values to within 0.001 (LR 0.926 / 0.159, RF 0.985 / 0.841, MLP 0.951 / 0.282). The
MLP clean validation Macro-F1 matches the notebook's tuning table exactly (0.9393). The RF
values differ slightly in the third decimal place, likely because the notebook used a different
environment. All 180-epoch MLP fits emit convergence warnings.

## Open decisions (not changed here)

- Whether the notebook protocol replaces the current scientific protocol, results and paper.
- Whether the current results stay in the paper as an unlimited-capacity comparison.
- Whether the public demo switches to the notebook models (`ECS171_EXPERIMENT=elsey`). In that
  mode the page still states 2,709 held-out test rows; the notebook split has 2,032.
