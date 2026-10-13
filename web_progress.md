# Execution ledger — plan: reports/Web_Phase_Plan.md

Pre-flight: saved Pipeline + model classes → HTTP prediction API; use all 16 raw inputs in fixed protocol order and retain the existing Pipeline preprocessing.
Pre-flight: metadata/model checksums + frozen context → demo loading; verify before deserialization, allow only six predeclared local model paths.
Pre-flight: API probabilities/class ordering → HTML bars; deliver name/probability pairs rather than assume array positions.

Ruling: implementation and inference remain inside this dedicated repository; no worktree copy is needed because there is no shared project checkout and the existing large artifacts must remain available locally.

- [ ] Local Git baseline and artifact manifest
- [ ] Real inference API, validation and integrity tests
- [ ] Browser UI, launch script and actual interaction checks
- [ ] Fresh review, verification report and final local commit
