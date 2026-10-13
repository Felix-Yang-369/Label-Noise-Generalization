# Execution ledger - plan: ../11_Proposal_and_Master_Plan/02_Master_Plan.txt

Spec: ../11_Proposal_and_Master_Plan/experiment_protocol.json. Scope: environment, data preparation, training-only EDA, ten tuning fits, 45 main fits, frozen final evaluation, reproducible local results. Optional capacity study and web deployment are outside this requested phase.

- [x] Environment
- [x] Tests / data preparation / EDA
- [x] Noise generation / smoke checks / ten candidate fits
- [x] Freeze protocol and selected parameters / 45 main runs
- [x] Final holdout evaluation / plots / report / verification

Pre-flight: raw XLSX -> cleaning -> fixed split -> EDA/models: feature spelling AspectRation preserved, original Excel row_id is excluded from features; duplicate removal precedes split.
Pre-flight: noise -> tuning/main -> evaluation: only training labels change; validation and test labels remain original; selected_params immutable before main; test metrics withheld until all main fits complete.
Pre-flight: training -> results/inference: save complete Pipelines, class order and every row-level prediction.

Ruling: use a dedicated 12_Implementation folder rather than Git-specific worktree helpers, because this task is a projectless deliverable folder with no accessible repository. No commits, push or external submission. Code SHA-256 and local ledger replace commit provenance; cost if wrong: later Git history must start from the delivered sources.
Environment: bundled Python is 3.12.14, matching the planned Python version. Initial venv ensurepip failed because sandbox temporary directory is unwritable; use a workspace-local temporary directory without modifying system Python.

Environment complete: Python 3.12.14; pinned dependencies installed; pip check clean. Workspace temp directory resolves sandbox temp permission issue.
Core contracts RED -> GREEN: 10 failed before implementation -> 10 passed. Execution contracts RED -> GREEN: 2 failed -> 12 passed. EDA contract RED -> GREEN: 1 failed -> 13 passed. One upstream warning is confined to the single-class metric test; seven-class experiments do not use that fixture.
Data preparation complete: 68 duplicates removed before split; train=8125, validation=2709, test=2709. Raw ZIP CRC/SHA and XLSX member verified; row IDs persisted. Fifteen deterministic training-label files generated.
Ruling: limit BLAS to one thread for all fits, keep RF n_jobs=4, and execute fits sequentially; this prevents tiny MLP matrix operations from incurring excessive thread overhead. Cost if wrong: runtime measurements describe this hardware/thread configuration, not all CPUs.

Independent review: three Important findings, no Critical leakage issue. All three reproduced (3 failed, 13 passed) and fixed (16 passed): live environment comparison on every scientific context; canonical full-matrix final gate; independent completed metadata SHA seal. No minor findings.
Ruling: preserve first-pass tuning and code under artifacts/pre_review + initial_code_version and config/history, and repeat tuning after provenance fixes. They are excluded from the current 55 research fits; cost is about one minute of extra local computation, avoiding mixed code fingerprints. No holdout opened before this correction.
Final review unjudged items (report rendering, completed provenance, README) will be checked locally once results exist; they are required deliverables, not deferred scope.

Final acceptance complete: 17 passed; actual original-data row comparison, independent 45-run metrics, six model reloads, mean/SD, all hashes and stage-time gate verified. Four result plots and three EDA plots visually checked. Read-only reviewer findings all fixed; no deferred minors. Optional capacity and web/deployment outside this phase.
