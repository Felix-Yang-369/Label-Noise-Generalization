# Inference demo verification

The demo serves six saved seed-17 pipelines (LR, RF, MLP × `p=0`, `p=1`). It performs inference only: no fitting, no test-set access, and no model selection happens through the web interface. Research source hashes and the final holdout receipt were left unchanged when the demo was added.

## Checks

- **Test suite.** 47 tests passed, with one known scikit-learn warning from a synthetic single-class metric fixture.
- **Agreement with offline inference.** HTTP predictions for all six models were compared with independently loaded pipelines. Feature order in the request was deliberately reversed; predicted classes matched and probabilities agreed within an absolute tolerance of `1e-12`.
- **Input validation.** Missing or extra features, unknown model/condition names, arrays instead of objects, malformed JSON, `null`, empty strings, booleans, NaN/Inf, non-numeric strings, and extreme magnitudes are rejected without returning a prediction.
- **Out-of-range inputs.** Finite values outside the training range are passed through unclipped and produce a visible range notice. Ranges come from training rows only.
- **Examples.** Each of the seven examples is the validation row with the smallest `row_id` in its class, chosen independently of model accuracy. Editing an example removes its true-label tag.
- **Start-up integrity.** The server checks dependency versions, frozen inputs, scientific source bytes, and model checksums before loading any joblib file; a deliberately corrupted model is refused.
- **Browser interaction.** In Chrome, each model produced clean and random-label outputs; an empty `Area` field produced an error with no result cards; `Area=-1` produced the range notice.
- **Slow responses.** With 2000 ms of simulated latency, inputs and model controls stay disabled until the response arrives, so results are never shown under a changed selection (`tests/browser_checks.mjs`).
- **Layout.** At a 390×844 viewport the page has no horizontal overflow; the desktop comparison and the upper mobile view were inspected (`reports/demo/`).
- **Restart consistency.** After stopping and restarting the server, the same request returned identical classes, probabilities, and metadata (`reports/demo/restart_consistency.json`).

## Public deployment

The same six models are deployed at <https://ecs171-label-demo.vercel.app>. Anonymous production requests on all seven validation examples (42 model/example pairs) matched offline classes and metadata, with a maximum probability difference of `2.78e-17` (`docs/Deployment_Verification.json`).

## Limits

Displayed probabilities are raw model outputs, not calibrated confidence. The local launcher binds to `127.0.0.1` with Flask debug disabled and is meant as a development and presentation fallback, not a production server.
