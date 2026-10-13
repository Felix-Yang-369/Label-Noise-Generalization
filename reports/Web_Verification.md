# Real inference demo verification

Scope: section 09 of the approved master plan. Existing research fits, scientific source hashes and the final holdout receipt were preserved. No retraining or new test-based model selection occurred.

## Evidence

- Local Git baseline: `6fcccff`, 124 files. Implementation commit: `6b7ce37`. Dedicated repository, branch `codex/ecs171`; no remote configured.
- Fresh full Python suite before review: **47 passed**, one previously documented sklearn warning from a single-class synthetic metric fixture (`logs/web_before_review_tests.log`).
- Six actual HTTP prediction/probability outputs checked against independently loaded saved Pipelines. Feature dictionary order deliberately reversed; output classes matched, probabilities agreed with absolute tolerance 1e-12. No fabricated output or HTTP training path.
- Rejected missing/extra features, wrong model/condition, arrays instead of objects, malformed JSON, None, empty strings, booleans, NaN/Inf, nonnumeric strings, and extreme magnitudes. No prediction returned for invalid input.
- Outside-range finite values are passed without clipping and generate training-range warnings. Ranges use train only. Seven examples use each class's smallest validation row_id, independent of model performance.
- Startup validates actual dependencies, frozen context, scientific source bytes, input checksums and sealed model metadata/artifacts before loading joblib. Deliberately corrupt model refused startup.
- Chrome live interaction: LR/RF/MLP each produced both clean/random saved-model outputs; single-random display worked; empty Area returned a visible error with zero result cards; Area=-1 displayed its range warning. Editing an example removed its source-class claim.
- Slow-request UI issue reproduced before repair with 2000 ms emulated latency. After repair, input/model/example controls were disabled while pending and restored after completion. Network throttling was removed after the test.
- Desktop full-page screenshot inspected. At requested 390×844 viewport, content/client widths both 375 px, showing no horizontal overflow; readable upper output-region screenshot inspected. Full-page and lower-region mobile captures timed out. Mobile visual inspection is partial; do not claim complete mobile screenshot acceptance. Viewport override was reset. No Chrome console errors observed.
- The sandbox prevented local networking (WinError 10013); live browser verification used an ordinary local process listening only on 127.0.0.1:5172. This is a development/local service.

Evidence files are under `reports/demo/`; `tests/browser_checks.mjs` records the browser slow-request regression scenario. Its equivalent actions were executed directly through the documented browser API because the runtime disallowed loading the local module. Python tests run through `Run_Project.ps1 -Stage verify`.

## Boundary

Public deployment, GitHub push, formal course submission, outbound messages, backup video and optional capacity experiments were not performed. The localhost service is not evidence of public deployment. A Git clone does not include ignored generated model assets; README gives fresh-output reproduction instructions.

## Independent review and completion

Fresh reviewer inspected baseline `6fcccff` through feature commit `6b7ce37`, requirements, tests, source and saved evidence. No Critical or Important defects found; ready for local-phase completion. Its independent rerun passed 47 tests with the known synthetic warning, and all 397 experiment-artifact hashes matched. The initial bare pytest invocation encountered sandbox temp-directory setup errors; an accessible workspace basetemp resolved those errors. The project's normal wrapper already sets its temp paths appropriately.

One minor runbook path typo in the feature commit had already been corrected during packaging; the final commit includes that correction. No deferred code defects remain. Reviewer independently inspected the saved screenshots/JSON but did not repeat live browser actions.

Rulings on declined scope: public deployment/production-server hardening remains outside this local phase; video/formal presentation readiness remains future work; capacity experiments remain optional. These exclusions are documented, and local-phase completion does not imply full course submission readiness.

The actual local service was stopped and restarted through the launcher. The same request returned exactly identical classes, all probabilities and metadata, recorded in `reports/demo/restart_consistency.json`. Final Chrome reload and prediction confirmed the expected LR clean/random outputs; a repeated full-page screenshot attempt timed out, so the earlier successfully saved desktop comparison is retained rather than claiming a new capture.

Final full-suite log: `logs/web_final_tests.log`. Local Git retains the baseline, feature and verification commits; portable history record/backup is in the parent project folder. Source manifest and the original final-evaluation receipt remain unchanged.
