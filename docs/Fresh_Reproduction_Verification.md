# Fresh-workspace reproduction verification

Verified on 2026-10-18 using the repository-root `Reproduce.ps1` helper. The original scientific implementation and recorded experiment were left unchanged.

## Execution

The helper created a previously absent workspace at `D:\Projects\ECS 171\work\final_report\fresh_reproduction`, copied only the documented inputs, created a new virtual environment with Python 3.12.14, and installed all 32 locked packages. The original freeze, selected parameters, environment record, processed data, models, predictions, and results were not copied.

The new workspace completed:

- Raw ZIP/XLSX verification, 68-duplicate removal, and fixed 8,125 / 2,709 / 2,709 splits.
- Training-only EDA and all four excluded diagnostic smoke fits.
- All 10 clean-label tuning fits and the same selected LR/RF/MLP parameters.
- A new freeze before all 45 main fits and the fresh final holdout evaluation.
- Generated figures, overview tables, analysis files, and complete final evaluation receipt.
- **47 tests passed**, with the same one known warning from a synthetic single-class metric fixture; test time was **8.17 seconds**.

The helper returned exit code 0 and wrote `REPRODUCTION_SETUP.json` with status `full pipeline and verification completed` and `copied_original_results: false`.

## Original versus fresh results

An independent comparison script completed **197 checks**, all passing:

| Comparison | Result |
|---|---|
| Main metric table | Same 45 run IDs; all non-runtime fields match. The 600 numerical comparisons have maximum absolute difference **0**. |
| Tuning table | Same 10 candidates; all non-runtime fields match. The 20 numerical comparisons have maximum absolute difference **0**. |
| Selected parameters | LR `C=1`; RF `max_depth=None`, `min_samples_leaf=1`; MLP `alpha=0.01`, all identical. |
| Processed/noise CSVs | Byte-identical to the original derived data. |
| Scientific source | `core.py` and `pipeline.py` retain their exact original SHA-256 values. |
| Fresh receipt | All **95** result-file hashes verify against the new receipt. |
| Convergence records | **14** main runs recorded warnings, matching the original experiment. |

The comparison used an absolute numerical tolerance of `1e-12`; actual differences in the compared metrics were zero. Runtime was intentionally excluded from equality checks. Fresh research fitting time was **380.1094909001258 seconds** (tuning 50.03888700000243; main 330.0706039001234), versus 462.835090000066 seconds recorded for the original execution. These are timed fitting calls, not total elapsed execution time.

Fresh timestamps, environment paths, metadata, and receipt hashes differ by design. Fresh receipt SHA-256: `1a59cea4e7232aafcd6a774fcf3300d39088697998700c6d34b92b9a83311299`. Fresh freeze fingerprint: `446f5c5e89f0b8b1cac8be3a98d589e3ec26635ade2ca42955c2228323ef6428`.

## Real HTTP smoke check

A new process served the **freshly reproduced six model pipelines** on `127.0.0.1:5174`. A free-port check preceded startup. The live HTTP health endpoint reported six models; bootstrap returned 16 inputs and seven validation examples. Paired predictions for all three models were compared against independently loaded fresh pipelines, and all six probability vectors and predicted classes matched exactly.

The server was stopped after verification. This check validates real listener startup and inference; it is not a new browser visual inspection or a public deployment.

## Interpretation and evidence

This is a technical reproduction using the same dataset, split, seeds, conditions, source, and locked package versions. It does not add independent statistical replications, unseen holdout data, or new model-selection evidence. The original final report should retain the original research results and runtime; the fresh execution demonstrates that the documented repository workflow recreates them.

Detailed evidence in this folder:

- `Fresh_Reproduction.log`: environment installation, all stages, and test output.
- `Fresh_Reproduction_Verification.json`: 197 checks, table comparisons, receipt verification, and HTTP prediction checks.
- `Fresh_Comparison_and_HTTP.log`: compact independent verification output.

The complete generated reproduction assets remain in the separate reproduction workspace; no original experimental asset was modified.
