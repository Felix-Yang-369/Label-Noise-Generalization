# Execution ledger — plan: reports/Web_Phase_Plan.md

Pre-flight: saved Pipeline + model classes → HTTP prediction API; use all 16 raw inputs in fixed protocol order and retain the existing Pipeline preprocessing.
Pre-flight: metadata/model checksums + frozen context → demo loading; verify before deserialization, allow only six predeclared local model paths.
Pre-flight: API probabilities/class ordering → HTML bars; deliver name/probability pairs rather than assume array positions.

Ruling: implementation and inference remain inside this dedicated repository; no worktree copy is needed because there is no shared project checkout and the existing large artifacts must remain available locally.

- [x] Local Git baseline and artifact manifest — baseline 6fcccff; 124 tracked files; 397 generated artifact checksums. Fresh baseline suite 17 passed, one known synthetic warning.
- [x] Real inference API, validation and integrity tests — RED missing application; GREEN full suite 47 passed, one known synthetic warning. Actual six model outputs match independent offline predictions/probabilities within 1e-12.
- [x] Browser UI, launch script and actual interaction checks — three models × both label conditions, single random condition, missing Area, range notice and custom-input source-label removal verified in Chrome. Desktop screenshot inspected. At a 390×844 viewport, page content width equals client width (375 px): no horizontal overflow; readable top-of-output screenshot inspected. Full-page/lower-region mobile screenshots timed out; recorded scope is partial, not full mobile visual acceptance. No browser console errors observed.
- [x] Fresh review — 47 tests independently passed; all 397 artifact hashes matched; no Critical/Important defects. One documentation-path minor already corrected and included in final bookkeeping. Reviewer inspected saved browser evidence but did not repeat interactions.
- [x] Final verification and local commit — fresh full suite 47 passed, one known synthetic warning; source/holdout integrity preserved. Local baseline, feature and verification commits retained; Git fsck and portable bundle verification performed during final history readback.

Browser finding: a pending request left model/input controls editable. Real Chrome check with 2000 ms emulated latency showed `modelEnabled=true` and `areaEnabled=true` (RED). Lock all input controls for the duration of the request to prevent results under a changed selection. Restore controls on both success and failure; verify GREEN in the same browser.

Browser GREEN: model, Area and Load example disabled during the pending request; controls restored after completion. Evidence: `reports/demo/slow_request_check.json`.

Ruling: retain the user-authorized local Git history on `codex/ecs171` — this dedicated repository has no separate base branch or remote, so there is no branch integration to perform.

Final review rulings: public deployment/production hardening, recorded video/presentation readiness and optional capacity work are explicitly future scope — local inference is complete; these are not claimed as delivered. Cost if omitted from later course delivery: presentation/deployment requirements may remain unmet, so the project status tracks them.

Actual launcher restart: identical classes, seven-class probabilities and metadata on the same input, verified over local HTTP and saved in `reports/demo/restart_consistency.json`. No retraining or scientific code changes.
