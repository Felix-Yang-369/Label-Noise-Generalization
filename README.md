# Label Noise & Generalization

**Memorization and Generalization under Training-Label Randomization on Tabular Data**

**ECS 171 · UC Davis · Group 8**

A controlled study of how logistic regression (LR), random forests (RF), and a multilayer perceptron (MLP) respond when training labels are progressively randomized. The experiment uses the public UCI Dry Bean dataset, retains clean validation and test labels, and separates fitting corrupted training targets from predicting true labels on unseen observations.

The required research experiment is complete: **10 clean-label tuning fits, 45 main fits, and evaluation on one fixed clean holdout**. This repository contains the scientific source, locked dependencies, public raw data, recorded results, figures, tests, and an inference application with a verified public deployment. Generated model files and row-level predictions are excluded from Git and can be regenerated.

## Live Demo

**[Open the online Demo](https://ecs171-label-demo.vercel.app)** — no installation or sign-in required. Choose LR, RF, or MLP, load a validation example, and compare clean versus randomized-label predictions. Probabilities are uncalibrated model outputs.

**Deployment status (October 18, 2026):** Vercel production is live on the existing free Hobby plan. Anonymous HTTP checks compared all six saved models on seven validation examples (42 comparisons); classes and metadata matched, and the maximum probability difference from offline inference was `2.78e-17` (tolerance `1e-12`). Browser prediction and model switching were checked. No paid integrations or plan upgrade were enabled. Availability remains subject to Hobby usage limits. See [deployment verification](docs/Deployment_Verification.json).

## Main result

| Model | Clean-label test Macro-F1, mean ± sample SD | Fully randomized-label training accuracy, mean | Fully randomized-label test Macro-F1, mean |
|---|---:|---:|---:|
| Logistic regression | 0.9404 ± 0.0000 | 0.1602 | 0.1144 |
| Random forest | 0.9361 ± 0.0012 | 1.0000 | 0.1314 |
| MLP | 0.9346 ± 0.0040 | 0.3806 | 0.1309 |

Each condition has three seeds on the **same data split**. These standard deviations describe seed variation, not uncertainty across independent datasets or splits. The random forest interpolates the fully randomized training labels while clean-test performance collapses. The tested MLP does **not** interpolate those labels under its fixed architecture, regularization, optimizer, and iteration budget; 14 main runs recorded convergence warnings. The results do not establish universal model rankings or statistical significance.

![Clean-test performance as training labels are randomized](paper/figures/test_performance.png)

## English IEEE report

Read the [compiled report](paper/ECS171_Final_Report.pdf), edit the [LaTeX source](paper/main.tex), or inspect the [paper package guide](paper/README.md). The report has ten body pages plus one reference page, six main figures, five tables, and seven primary references. Its final body page includes this repository's URL, roadmap, delivery status, and contribution record. It is a review version dated October 18, 2026; public deployment is verified; team acceptance, backup recording, rehearsal, and course submission remain pending.

The paper's vector figures and compact recorded source copies are tracked. After installing the locked plotting dependencies, `python paper/tools/build_figures.py --recorded-sources` regenerates these historical figures without model fitting. This is separate from running the scientific experiment again.

The parameter `p` is the probability of replacing a label with a uniform draw from all seven classes, including its original class. Its expected actual change rate is `6p/7`; `p=1` does not mean every label changes.

## Inspect the recorded experiment

No Python environment is required to read these tracked files:

- [Run-level metrics: 45 rows](results/test_metrics.csv)
- [Condition summaries: 15 model/noise groups](results/model_noise_summary.csv)
- [Ten tuning candidates](results/tuning.csv)
- [Model overview](reports/experiments/overview.csv)
- [Result figures and original analysis notes](reports/experiments/)
- [Training-only EDA](reports/eda/)
- [Frozen protocol](config/protocol.json) and [main run matrix](config/main_matrix.json)
- [Final evaluation receipt](results/FINAL_EVALUATION.json)
- [Original implementation verification](reports/Implementation_Verification.md) and [web verification](reports/Web_Verification.md)

Some historical analysis and execution notes are in Chinese. Numerical column names, source code, configuration keys, and this reproduction guide are in English. Configuration fields such as `training_completed: false` and matrix `status: planned` preserve the **pre-execution specification**; actual completion is established by the recorded results and evaluation receipt.

## Reproduce safely

**Use a fresh sibling run directory.** A Git clone includes the original frozen configuration and result receipts but omits the original models, processed splits, noise files, and row-level predictions. Running `all` directly inside that evidence checkout would combine incomplete historical assets with a new execution and may correctly fail an integrity check.

Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) to create an input-only run directory and execute the pipeline without altering the recorded experiment. The documented path uses Windows, PowerShell, and Python 3.12; Python 3.12.14 and scikit-learn 1.7.2 were used for the original run. A fresh run records its own interpreter/environment signature, timestamps, hashes, models, and receipt.

**Fresh local reproduction verified on October 18, 2026:** a new directory and newly installed environment completed all 55 research fits and 47 tests. Every compared non-runtime numerical field in the tuning/main tables matched the original exactly; all six regenerated web models matched offline probabilities in real HTTP checks. The evidence is in [docs/Fresh_Reproduction_Verification.json](docs/Fresh_Reproduction_Verification.json). This is a same-machine reproduction, with its own metadata and timing, not a cross-platform claim.

The repository-root helper performs that entire fresh-run workflow, including environment installation, all stages, and verification:

```powershell
.\Reproduce.ps1
```

Use `-DestinationPath 'D:\Experiments\ecs171-new-run'` to choose a new destination, `-PythonExecutable 'C:\path\to\python.exe'` for an explicit Python 3.12 interpreter, or `-PrepareOnly` to copy inputs without installing packages or running fits. Existing destinations are rejected.

After creating that clean run directory and its virtual environment, the main commands are:

```powershell
.\Run_Project.ps1 -Stage all
.\Run_Project.ps1 -Stage verify
.\Run_Demo.ps1
```

The full pipeline includes four short smoke fits in addition to the 55 research fits. Smoke fits are excluded from research results. No GPU, cloud compute, paid API, or external dataset download is required after cloning; locked Python dependencies still need to be installed.

## Experimental design

- **Data:** 13,611 raw observations; 68 exact duplicates removed before splitting; 13,543 retained observations; 16 numerical features; 7 classes. Original source names `AspectRation` and `roundness` are preserved. `row_id` is provenance, never a predictor.
- **Fixed stratified split:** 8,125 training, 2,709 validation, and 2,709 test observations; split seeds 171 and 172.
- **Preprocessing:** train-fitted `StandardScaler` inside the LR and MLP pipelines; unscaled numerical inputs for RF. EDA uses training rows only.
- **Tuning:** ten candidates trained on clean labels; selection uses validation Macro-F1 and predeclared complexity tie rules within 0.001. Selected settings are LR `C=1`, RF `max_depth=None` / `min_samples_leaf=1` with 200 trees, and MLP `alpha=0.01` / hidden layers `(128, 64)`.
- **Main matrix:** three models × `p ∈ {0, .25, .5, .75, 1}` × seeds `{17, 29, 43}` = 45 fresh fits. Parameters remain fixed across noise levels.
- **Noise control:** within each seed, the same uniform draws are reused across noise levels; each label vector is shared across models and remains fixed within a fit. Validation and test labels remain clean.
- **Metrics:** primary clean-test Macro-F1; secondary accuracy, balanced accuracy, training accuracy against both clean and fitted targets, convergence information, and fitting time. Results report means and sample SD over three seeds.
- **Holdout:** tuning and the complete main matrix were frozen before final evaluation. Tests check that a missing run or an incomplete/substituted matrix cannot open the final evaluation.
- **Compute:** sequential fits, BLAS limited to one thread during fitting, RF configured for four workers. Recorded research fitting time is about 463 seconds on the original machine; this excludes setup, plotting, inference, smoke fits, and other overhead.

The test set has already been evaluated. Any later modeling motivated by these results must be identified as exploratory rather than presented as part of the original confirmatory sequence. The optional 18-fit capacity study described in the protocol has not been run and is absent from the reported results.

## Repository map

| Path | Role | In Git? |
|---|---|---|
| `src/core.py` | Data integrity, splitting, noise, models, metrics, cache checks | Yes |
| `src/pipeline.py` | Staged tuning, freezing, main training, final evaluation | Yes |
| `src/visuals.py` | Training-only EDA and result figures | Yes |
| `config/` | Protocol, matrix, original selected parameters, frozen evidence | Yes |
| `data/raw/` | Unmodified public archive and spreadsheet | Yes |
| `results/` | Original metrics, class reports, final evaluation receipt | Yes, except row-level test predictions |
| `reports/` | Figures, EDA, verification, and demonstration evidence | Yes |
| `app/` | Flask application and browser interface | Yes |
| `tests/` | Scientific contract, execution, acceptance, and HTTP tests | Yes |
| `artifacts/` | Generated models, metadata, traces, train/validation predictions | No |
| `data/processed/`, `data/noise/` | Generated splits and perturbed targets | No |
| `results/test_predictions/` | Generated row-level holdout predictions/probabilities | No |
| `.venv/`, `.tmp/` | Local environment and temporary files | No |

The original artifact manifest lists 397 generated files. A clone can inspect that inventory but cannot validate missing model/prediction files until a complete matching artifact set is supplied. A fresh execution produces new evidence; its timestamp-dependent hashes are not expected to match the original receipt.

## Online and local inference demonstration

The primary shared Demo is **[https://ecs171-label-demo.vercel.app](https://ecs171-label-demo.vercel.app)**. The local launcher below remains a development and presentation fallback.

After reproducing the experiment, run `Run_Demo.ps1` from the fresh run directory and open `http://127.0.0.1:5172`. Stop with Ctrl+C; use `-Port 5173` if necessary.

The app loads six saved seed-17 pipelines: LR, RF, and MLP trained at `p=0` and `p=1`. It validates the environment, scientific source, frozen inputs, feature/class order, and model checksums before loading. Requests perform inference only. The interface offers 16 numeric inputs, seven validation examples chosen by row order independently of prediction quality, and paired clean/random-label predictions.

Displayed probabilities are uncalibrated model outputs. Editing a validation example clears its inherited true-label claim. The local launcher binds to `127.0.0.1` with Flask debug disabled. The public deployment uses the same six saved models and preprocessing; requests perform inference only. Team acceptance, backup recording, and presentation rehearsal remain separate tasks. See the [five-minute demo runbook](reports/Demo_Runbook.md).

## Provenance, attribution, and licensing

The implementation and execution records explicitly identify Codex assistance under the user's direction. Planned teammate roles are not evidence of completed individual contributions. Report authorship and contribution statements must reflect work actually completed and confirmed by the group.

The **Dry Bean dataset** is distributed by the UCI Machine Learning Repository under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The tracked raw ZIP and XLSX are unmodified; this project removes exact duplicates and creates derived training/validation/test partitions during preprocessing. Cite: *Dry Bean* [Dataset]. (2020). UCI Machine Learning Repository. [doi:10.24432/C50S4B](https://doi.org/10.24432/C50S4B). See the [official dataset page](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).

No additional software license is asserted for project-authored code in this repository. The dataset license is not a license for the project's source code.

## References

1. C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals, “Understanding Deep Learning Requires Rethinking Generalization,” ICLR, 2017. [arXiv:1611.03530v2](https://arxiv.org/abs/1611.03530v2).
2. *Dry Bean* [Dataset], UCI Machine Learning Repository, 2020. [doi:10.24432/C50S4B](https://doi.org/10.24432/C50S4B).
3. [scikit-learn 1.7 documentation](https://scikit-learn.org/1.7/).
