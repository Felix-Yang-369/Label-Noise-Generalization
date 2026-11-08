# ECS171 live demo runbook

Primary demo: <https://ecs171-label-demo.vercel.app> (no setup or sign-in). Open it before presenting. The local launcher below is the fallback; keep `reports/demo/01_desktop_comparison.png` ready in case neither is reachable.

## Start and check

For the local fallback, open a terminal in a reproduced run directory, run `./Run_Demo.ps1` (or `python -m app --port 5172` on macOS/Linux), then open <http://127.0.0.1:5172>. Keep the server terminal running. Use `-Port 5173` if 5172 is occupied. This command starts the local fallback; the primary public deployment is linked above.

1. Confirm the page loads 16 numeric fields and the validation-example selector.
2. Load the default BARBUNYA example (validation row 2032), select LR, choose Compare both, and click Predict & compare.
3. The existing LR predicts SIRA under clean labels and CALI under random labels on this example. It is a deliberately deterministic source-order choice, not selected for correctness. Explain that even a strong aggregate model can misclassify one bean.
4. Switch to Random Forest and run the same input. Show its training fit for random labels versus clean-label validation F1; distinguish memorization from generalization.
5. Switch to MLP and show the recorded iterations and stop reason in Model details. Do not interpret reaching an iteration budget as proof that neural networks cannot memorize labels.
6. Optionally choose another validation class and click Load example. Clear Area once to demonstrate validation, then reload the example to restore all values.

## Five-minute structure

| Time | Evidence / action |
|---|---|
| 0:00–0:40 | Assigned paper and the question: fitting arbitrary labels versus predicting clean labels |
| 0:40–1:30 | UCI source, 16 features, deduplication, fixed train/validation/test split and train-only scaling |
| 1:30–3:00 | Actual 45-run comparison; test Macro-F1 means/SD; RF random-label memorization and its weak clean-test score |
| 3:00–4:30 | Live input → saved Pipeline → real output, same-input clean/random comparison |
| 4:30–5:00 | One dataset, three noise/model seeds on one fixed split, uncalibrated probabilities, MLP budget limits |

Open `reports/experiments/index.html` within this repository for the offline research report. Screenshots in `reports/demo` document actual execution; they are visual evidence, not a substitute for live inference.

## Recover

- If the page cannot connect, restart `Run_Demo.ps1` and refresh. No Internet connection is needed after setup.
- If asset/environment verification fails, use the locked environment and the original completed assets; do not bypass checks or deserialize third-party joblib files.
- A Git clone intentionally excludes generated models. Reproduce a fresh experiment in a separate output copy using README instructions before starting its demo.
- Stop using Ctrl+C. Do not run this development server as a public production service.

## Interpretation

p is the probability of redrawing a training label uniformly, allowing the original class. p=1 is not a guarantee that every label changes. Training fit uses each run's actual fit labels. Validation Macro-F1 uses clean validation labels. Probability bars are raw model outputs, not calibrated confidence. Generalization claims use all 2,709 held-out test rows, not selected demonstrations.

Implementation references: [Flask quickstart](https://flask.palletsprojects.com/en/stable/quickstart/) and [scikit-learn 1.7 model persistence](https://scikit-learn.org/1.7/model_persistence.html).
