# Integration of the selected Group 8 notebook experiment

The team has selected the experiment in `integration/elsey/ECS171_Group8_Project.ipynb` as the final scientific specification. The current `src/`, `config/`, `results/`, `paper/`, and deployed `app/` still describe a different historical protocol, and **must not be presented as results of this notebook** until migration and revalidation finish.

## Selected notebook protocol

- Deduplicated UCI Dry Bean data: 13,543 rows.
- Fixed stratified train/validation/test: 9,480 / 2,031 / 2,032 (roughly 70/15/15; random state 17).
- LR, RF, MLP; 18 clean-label tuning candidates; selected `k=16` for all three models.
- Selected LR `C=10`, RF 200 trees / depth 12 / leaf 1, MLP `(128, 64)` / `alpha=0.01` / learning rate 0.001 / 180 iterations.
- Five replacement probabilities and seeds 17, 29, 43: 45 main fits. Validation and test labels remain clean.
- At `p=1`, mean RF accuracy on resampled training labels is approximately 0.8461; clean-test macro-F1 approximately 0.1222.

## Required migration before declaring the existing application current

1. Agree on the exact notebook snapshot and freeze its environment and raw-data hashes.
2. Reproduce the selected notebook cleanly and export run-level results, summary metrics, and figures.
3. Update the scientific source/configuration, split manifest, selected parameters, and all associated checksums together. Preserve the old version in Git history.
4. Generate the six model pipelines used by the current paired clean/random-label web interface from the selected protocol, or adjust the interface to explicitly support a reduced model set.
5. Rebuild and verify the web application's model metadata, validation examples, expected feature/class order, and online/offline prediction comparisons.
6. Update the README, paper, scientific figure assets, tests, and public deployment only after the preceding steps pass.
7. Review and confirm each person's actual contributions before using completed-work attribution.

The historical deployed Demo and existing final paper are **not yet validated** against the selected notebook. Keep this change on an integration branch until scientific and web acceptance tests pass.

## New optional demo adapter (work in progress)

`integration/elsey/export_demo_models.py` reconstructs the notebook's exact
70/15/15 split, selected LR/RF/MLP hyperparameters, seed 17, and the notebook's
label-resampling procedure. It exports the six clean (`p=0`) and fully random
(`p=1`) model pipelines required by the existing compare-mode UI, as well as
validation examples and model checksums. **It never reads test labels during
model selection or demo prediction.**

From the repository root, with the dependencies installed:

```bash
python integration/elsey/export_demo_models.py
ECS171_EXPERIMENT=elsey python -m app
```

This uses the new `app/elsey_service.py` through the existing Flask routes and
front-end. Without `ECS171_EXPERIMENT=elsey`, the legacy demo remains selected.
Switching the *public deployment* requires updating its deployment environment
and supplying the new model bundle; this has not been done automatically.
