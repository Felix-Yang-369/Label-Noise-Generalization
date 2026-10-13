"use strict";
const byId = id => document.getElementById(id);
let setup, activeExample = null;
function message(text, error = false) {
  const node = byId("message"); node.textContent = text; node.hidden = !text;
  node.className = "message" + (error ? " error" : "");
}
function clearResult() { byId("results").replaceChildren(); message(""); }
function setBusy(busy) {
  document.querySelectorAll(".inputs input, .inputs select, .inputs button").forEach(node => { node.disabled = busy; });
}
function loadExample() {
  clearResult(); activeExample = setup.examples[Number(byId("example").value)];
  for (const field of setup.features) byId("field-" + field.name).value = activeExample.features[field.name];
  byId("example-info").textContent = `Validation row ${activeExample.row_id} · source class ${activeExample.class}. Chosen by row order, not prediction quality.`;
}
function element(tag, cls, text) {
  const node = document.createElement(tag); node.className = cls || "";
  if (text !== undefined) node.textContent = text; return node;
}
function resultCard(result) {
  const card = element("article", "result-card " + result.condition);
  card.append(element("p", "eyebrow", result.condition === "clean" ? "CLEAN LABELS · P = 0" : "RANDOM LABELS · P = 1"));
  card.append(element("h3", "prediction-class", result.class));
  const stats = element("div", "result-stats");
  stats.append(element("p", "", `Training fit ${(100 * result.training_label_accuracy).toFixed(1)}%`),
    element("p", "", `Validation F1 ${result.validation_macro_f1.toFixed(4)}`)); card.append(stats);
  const bars = element("div", "bars");
  for (const item of result.probabilities) {
    const row = element("div", "prob-row");
    const label = element("div", "prob-label");
    label.append(element("span", "", item.class), element("b", "", (item.probability * 100).toFixed(1) + "%"));
    const track = element("div", "track"), fill = element("span", "fill");
    fill.style.width = (item.probability * 100) + "%"; track.append(fill); row.append(label, track); bars.append(row);
  }
  card.append(bars, element("p", "muted run-id", result.run_id));
  const details = element("details", "model-details"), summary = element("summary", "", "Model details");
  details.append(summary, element("p", "", "Parameters: " + JSON.stringify(result.params)),
    element("p", "", "Iterations: " + JSON.stringify(result.iterations) + " · " + result.stop_reason),
    element("p", "hash", "Model SHA-256: " + result.model_sha256)); card.append(details); return card;
}
async function predict(event) {
  event.preventDefault(); clearResult(); const features = {};
  for (const field of setup.features) {
    const input = byId("field-" + field.name), text = input.value.trim();
    input.removeAttribute("aria-invalid");
    if (text === "" || !Number.isFinite(Number(text)) || Math.abs(Number(text)) > 1e12) {
      input.setAttribute("aria-invalid", "true"); input.focus();
      message(`${field.name}: enter a finite number with magnitude at most 1e12.`, true); return;
    }
    features[field.name] = Number(text);
  }
  const button = byId("predict"); setBusy(true); button.textContent = "Running saved models…";
  try {
    const response = await fetch("/api/predict", {method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({model:byId("model").value, condition:byId("condition").value, features})});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || "Prediction failed.");
    const output = byId("results"), cards = element("div", "result-grid");
    for (const result of data.predictions) cards.append(resultCard(result));
    output.append(cards, element("p", "muted footnote", data.probability_note));
    if (data.range_warnings.length) message(data.range_warnings.join("\n"));
    else message("Prediction complete · saved models · unchanged raw feature values.");
  } catch (error) { message(error.message + " No prediction is displayed.", true); }
  finally { setBusy(false); button.replaceChildren(document.createTextNode("Predict & compare "), element("span", "", "→")); }
}
async function initialize() {
  try {
    const response = await fetch("/api/bootstrap"); if (!response.ok) throw new Error("Model service unavailable.");
    setup = await response.json(); const examples = byId("example"); examples.replaceChildren();
    setup.examples.forEach((item, index) => { const option = element("option", "", item.class); option.value = index; examples.append(option); });
    const inputs = byId("features");
    for (const field of setup.features) {
      const label = element("label", "feature-label", field.name === "AspectRation" ? "Aspect ratio" : field.name);
      label.htmlFor = "field-" + field.name;
      const input = element("input"); input.id = "field-" + field.name; input.name = field.name;
      input.type = "number"; input.step = "any"; input.required = true;
      const hint = element("small", "", `Train: ${field.min.toPrecision(4)}–${field.max.toPrecision(4)}`);
      label.append(input, hint); inputs.append(label);
      input.addEventListener("input", () => {
        activeExample = null; clearResult(); input.removeAttribute("aria-invalid");
        byId("example-info").textContent = "Custom measurements · source class no longer applies to the edited input.";
      });
    }
    examples.disabled = false; byId("load-example").disabled = false; byId("predict").disabled = false;
    byId("load-example").addEventListener("click", loadExample);
    for (const id of ["model", "condition", "example"]) byId(id).addEventListener("change", clearResult);
    byId("prediction-form").addEventListener("submit", predict); loadExample();
  } catch (error) { byId("example-info").textContent = "Unable to initialize saved models."; message(error.message, true); }
}
initialize();
