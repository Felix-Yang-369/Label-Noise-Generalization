# Real inference demo phase

Authority: `../11_Proposal_and_Master_Plan/02_Master_Plan.txt`, section 09.

1. Preserve the completed scientific pipeline and frozen evaluation; establish a local Git baseline containing source, protocol, raw data, summary results, figures and artifact checksums.
2. Load the six existing seed-17 Pipelines (LR/RF/MLP, p=0/1), after verifying their metadata, model checksums, frozen protocol and runtime compatibility. Never fit in HTTP requests.
3. Expose a Flask JSON API with strict 16-feature validation, ordered class probabilities, training-range warnings and per-run information. Provide deterministic validation examples, chosen without looking at predictions.
4. Build a responsive local HTML demo: editable numeric features, model selection, label-condition selection and a side-by-side clean/random-label comparison. Clearly distinguish a single example from aggregate evidence; probabilities are uncalibrated.
5. Test actual HTTP outputs against independently loaded saved Pipelines; reject malformed, missing, non-finite and extreme numeric inputs; fail closed on corrupt artifacts. Verify desktop/mobile interaction in a real browser and preserve evidence. Finish with one fresh-context review and local Git commits.

Ruling: the next required stage is the real inference webpage — the master plan explicitly puts webpage before report and optional capacity extension. No new training or test-based model selection is needed.

Ruling: use a dedicated repository and `codex/ecs171` branch in `12_Implementation` — an ancestor repository spans the Windows user directory; keep all staging and commits confined to this project.

Ruling: preserve existing scientific source bytes and hashes — new inference modules live in `app/`; existing 45-run results are historical evidence, not inputs to a new tuning decision.

Public deployment, GitHub push, course submission and outbound messages remain separate future actions. Localhost is a local demo, not a completed public deployment. Optional capacity fits are still optional.
