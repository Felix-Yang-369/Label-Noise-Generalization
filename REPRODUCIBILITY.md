# Reproducing the Dry Bean label-randomization experiment

This guide separates the original recorded experiment from a new execution. It preserves the Git checkout as evidence and generates all new outputs in a fresh sibling directory.

## 1. Requirements and scope

The original execution used Windows 11, Python 3.12.14, scikit-learn 1.7.2, and the package versions in `requirements.lock.txt`. The commands below use Windows PowerShell. Python 3.12 and Git must already be installed. Only the new run's local environment and files are created; no cloud service, GPU, or paid API is used.

The raw UCI archive and spreadsheet are already tracked. Dependency installation requires internet access unless wheels are cached. On October 18, 2026, the helper completed a new-directory, new-environment local reproduction: all 55 research fits, 47 tests, and six-model HTTP/offline checks passed. All compared non-runtime numerical fields matched the original exactly. See `docs/Fresh_Reproduction_Verification.json`; no cross-platform reproduction is claimed.

## 2. Preserve the evidence checkout

The recommended entry point, from the repository root, is:

```powershell
.\Reproduce.ps1
```

It validates Python 3.12 and the raw archive checksum, creates a new sibling directory, copies only the required inputs, creates the virtual environment, installs locked packages, executes all stages, and runs verification. It records the setup and completion in the new run's `REPRODUCTION_SETUP.json`. `-DestinationPath` selects a new destination; `-PythonExecutable` selects an explicit interpreter; `-PrepareOnly` creates the input-only workspace without installing dependencies or running fits. The helper never deletes or overwrites an existing run.

The following manual steps are equivalent and make the process inspectable.

Open PowerShell in the cloned repository. Choose a new sibling directory. The following block refuses to reuse an existing destination and copies only the inputs required by the implementation:

```powershell
$evidenceRoot = (Get-Location).Path
$newRun = Join-Path (Split-Path -Parent $evidenceRoot) 'ecs171-reproduction'
if (Test-Path -LiteralPath $newRun) {
    throw 'Destination already exists. Choose a new run directory.'
}
New-Item -ItemType Directory -Path $newRun | Out-Null

foreach ($folder in @('src', 'tests', 'app')) {
    Copy-Item -LiteralPath (Join-Path $evidenceRoot $folder) -Destination $newRun -Recurse
}
foreach ($name in @('Run_Project.ps1', 'Run_Demo.ps1', 'pytest.ini',
                    'requirements.txt', 'requirements.lock.txt')) {
    Copy-Item -LiteralPath (Join-Path $evidenceRoot $name) -Destination $newRun
}
foreach ($folder in @('config', 'data\raw', 'logs')) {
    New-Item -ItemType Directory -Path (Join-Path $newRun $folder) -Force | Out-Null
}
foreach ($name in @('protocol.json', 'main_matrix.json')) {
    Copy-Item -LiteralPath (Join-Path $evidenceRoot "config\$name") `
        -Destination (Join-Path $newRun 'config')
}
foreach ($name in @('UCI_Dry_Bean_Original.zip', 'Dry_Bean_Dataset.xlsx')) {
    Copy-Item -LiteralPath (Join-Path $evidenceRoot "data\raw\$name") `
        -Destination (Join-Path $newRun 'data\raw')
}
Set-Location -LiteralPath $newRun
```

The `logs` directory is required before preparation because environment capture writes `logs/sklearn_versions.txt`. Do not copy original `environment.json`, `config/FROZEN.json`, `config/selected_params.json`, `data/processed`, `data/noise`, `artifacts`, `results`, or generated reports into the new run. A new execution must establish its own internally consistent evidence.

The scientific code is copied byte-for-byte. The raw archive hash is checked against the protocol; the spreadsheet is checked against the verified archive member. No sibling course-material folder is required when both tracked raw inputs are present.

## 3. Create the environment

From the new run directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

Use the virtual environment's Python directly. Avoid installing unrelated packages into this environment after preparation: the pipeline compares `pip freeze` and the Python version to the saved environment signature before resuming. Python patch/build strings, platform details, and timestamps are captured for provenance. A fresh run with a different compatible interpreter establishes a new signature rather than reusing the original frozen metadata.

## 4. Execute the stages

```powershell
.\Run_Project.ps1 -Stage prepare
.\Run_Project.ps1 -Stage eda
.\Run_Project.ps1 -Stage smoke
.\Run_Project.ps1 -Stage tune
.\Run_Project.ps1 -Stage train
.\Run_Project.ps1 -Stage evaluate
.\Run_Project.ps1 -Stage report
.\Run_Project.ps1 -Stage verify
```

Alternatively, `Run_Project.ps1 -Stage all` executes preparation through report generation, followed separately by `-Stage verify`. The launcher stops at a failed stage and places temporary files inside the run's `.tmp` directory.

| Stage | Expected output and integrity condition |
|---|---|
| `prepare` | Captures the live environment; verifies raw ZIP/XLSX; removes 68 exact duplicates; writes fixed stratified splits and 15 noise files. |
| `eda` | Produces descriptive statistics and figures from training rows only. |
| `smoke` | Four short diagnostic fits; all are excluded from research summaries. |
| `tune` | Ten clean-label candidate fits; selects using validation Macro-F1 and declared tie rules; writes selected parameters and freezes the complete matrix. |
| `train` | Executes all 45 declared main fits; saves complete preprocessing/model pipelines, two training-target metrics, validation predictions, traces, warnings, and hashes. |
| `evaluate` | Requires the complete canonical matrix, matching frozen inputs, and validated cached artifacts before reading the clean holdout for metrics; writes 45 test rows, 15 groups, per-class reports, predictions, and a final receipt. |
| `report` | Regenerates experiment plots, HTML/Markdown analysis, and overview CSV. |
| `verify` | Runs the Python test suite, including independent metric calculations and checks of saved model inference. |

There are **55 research fits** plus four diagnostic smoke fits. The optional 18-fit capacity study is specified in the protocol but is not implemented as an executed stage or included in the reported experiment.

## 5. Check completion

Successful evaluation creates `results/FINAL_EVALUATION.json` with `run_count: 45`, `test_rows: 2709`, and hashes for the generated result files. Its fingerprint connects the evaluation to that run's frozen inputs. Check the following:

- Raw observations: 13,611; after deduplication: 13,543.
- Split sizes: training 8,125; validation 2,709; test 2,709.
- Tuning table: 10 candidates.
- Test metric table: 45 unique run IDs.
- Model/noise summary: 15 conditions.
- All 45 main model directories contain complete metadata, a metadata seal, model, train/validation predictions, and training trace.
- Full verification completes after artifacts have been generated. Original verification recorded 47 passing tests and one known warning from a synthetic single-class metric fixture; a fresh reproduction must be judged by its own test output.

The original selected parameters and rounded results are in the README. Identical numerical outputs are useful to compare, but machine-dependent numerical kernels can affect iterative training. Fitting times are machine dependent. Timestamps, environment paths, metadata fingerprints, and new model bytes are not expected to reproduce the original checksums.

`reports/Experiment_Artifact_Manifest.json` in the evidence checkout is an inventory of the **original** 397 generated experiment files. It is not copied into the new run and is not a checksum target for newly trained assets.

## 6. Launch the inference demonstration

After training and verification:

```powershell
.\Run_Demo.ps1
```

Open `http://127.0.0.1:5172`. The app loads the six generated seed-17 pipelines for LR/RF/MLP and `p=0/1`. It must find the new run's processed train/validation data, environment record, frozen inputs, complete metadata, and corresponding models. It performs no fitting during HTTP requests.

If the port is occupied, use `Run_Demo.ps1 -Port 5173`. Stop with Ctrl+C. Flask debug and automatic reloading are disabled. The launcher binds to localhost; it does not create a public deployment.

## 7. Resuming and interpreting failures

Resume a partially finished **same** run by rerunning its failed stage after fixing the external cause. A cached fit is reused only if its expected fingerprint, metadata seal, and artifact hashes match. The code deliberately rejects incompatible environment signatures, changed scientific source, changed data/noise, missing fits, and altered result files.

Do not disable these protections, overwrite a receipt, or remove original evidence to make a run continue. For a changed environment or scientific configuration, create another fresh run directory. Keep the original results separate for comparison.

The final holdout is already revealed in the recorded study. The evaluation gate enforces this pipeline's sequence; it is not an access-control system that prevents a person from inspecting public labels. Modeling decisions made after examining the recorded holdout should be disclosed as exploratory.

## 8. Files deliberately absent from Git

`.venv`, temporary files, trained model files, processed/noise CSVs, and row-level test predictions are ignored. Their absence keeps the code repository small and avoids committing redundant generated assets. The original ZIP and XLSX are retained because preparation requires both verified raw files and because the public dataset permits redistribution with attribution.

The project does not download model objects. Load only model artifacts from a trusted execution or validated project package; serialized Python model files are executable objects when deserialized. For this workflow, generate them locally through the documented pipeline.
