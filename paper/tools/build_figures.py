"""Rebuild paper vector figures and chart-ready data without fitting any model.

Run from any directory with the project's existing Python environment:
    .venv/Scripts/python.exe paper/tools/build_figures.py
For a clean clone containing the recorded paper source data:
    python paper/tools/build_figures.py --recorded-sources
All source paths are relative to the implementation repository root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
FIG = PAPER / "figures"
GEN = PAPER / "generated"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--recorded-sources", action="store_true",
    help="Read all 33 logical inputs exclusively from paper/generated/source_data; validate every recorded SHA256 before writing any output.",
)
args = parser.parse_args()
REQUIRED_SOURCES = [
    "results/test_metrics.csv", "results/model_noise_summary.csv",
    "data/processed/audit.json", "data/processed/split_manifest.json",
    "reports/eda/eda_metadata.json", "reports/eda/class_counts.csv",
    "reports/eda/feature_statistics.csv", "reports/eda/feature_correlation.csv",
    *[f"results/class_reports/{model}_p{p:03d}_s{seed}.json"
      for model in ["LR", "RF", "MLP"] for p in [0, 100] for seed in [17, 29, 43]],
    *[f"artifacts/runs/MLP_p{p:03d}_s17/training_trace.json" for p in [0, 25, 50, 75, 100]],
    "results/tuning.csv", "results/baselines.json",
]
recorded_manifest = None
validated_recorded_hashes = {}
if args.recorded_sources:
    manifest_path = GEN / "provenance.json"
    try:
        recorded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_sources = recorded_manifest["sources"]
        if set(recorded_sources) != set(REQUIRED_SOURCES):
            raise ValueError("provenance source set must match all 33 required logical inputs")
        # Complete this loop before making directories, copying sources, or saving figures.
        for rel in REQUIRED_SOURCES:
            raw = (GEN / "source_data" / rel).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != recorded_sources[rel]["sha256"]:
                raise ValueError(f"SHA256 mismatch for recorded source: {rel}")
            if len(raw) != recorded_sources[rel]["bytes"]:
                raise ValueError(f"byte-count mismatch for recorded source: {rel}")
            validated_recorded_hashes[rel] = digest
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"Recorded-source validation failed before output writes: {exc}\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

FIG.mkdir(exist_ok=True, parents=True)
GEN.mkdir(exist_ok=True, parents=True)

MODELS = ["LR", "RF", "MLP"]
PS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [17, 29, 43]
MODEL_COLORS = {"LR": "#245B85", "RF": "#B36D16", "MLP": "#805A8D"}
MODEL_MARKERS = {"LR": "o", "RF": "s", "MLP": "^"}
MODEL_DASHES = {"LR": "-", "RF": "--", "MLP": "-."}
ABBREV = ["Area", "Perim", "Major", "Minor", "Aspect", "Ecc", "Convex", "Diam", "Extent", "Solid", "Round", "Compact", "SF1", "SF2", "SF3", "SF4"]
CLASS_SHORT = ["BAR", "BOM", "CAL", "DER", "HOR", "SEK", "SIR"]
sources: dict[str, dict] = {}
figure_metadata: dict[str, dict] = {}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#656B70",
    "axes.labelcolor": "#22272B", "text.color": "#22272B",
    "xtick.color": "#30363B", "ytick.color": "#30363B",
    "axes.linewidth": .65, "lines.linewidth": 1.5,
    "grid.color": "#DBDFE2", "grid.linewidth": .5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def register(rel: str) -> Path:
    if rel not in REQUIRED_SOURCES:
        raise ValueError(f"Undeclared logical source: {rel}")
    path = (GEN / "source_data" / rel) if args.recorded_sources else (ROOT / rel)
    raw = path.read_bytes()
    exact_copy = GEN / "source_data" / rel
    digest = hashlib.sha256(raw).hexdigest()
    if args.recorded_sources:
        if digest != validated_recorded_hashes[rel]:
            raise ValueError(f"Recorded source changed after preflight validation: {rel}")
        sources[rel] = recorded_manifest["sources"][rel].copy()
    else:
        exact_copy.parent.mkdir(parents=True, exist_ok=True)
        exact_copy.write_bytes(raw)
        sources[rel] = {"sha256": digest, "bytes": len(raw), "exact_copy": f"generated/source_data/{rel}"}
    return path


def read_json(rel: str):
    return json.loads(register(rel).read_text(encoding="utf-8"))


def read_csv(rel: str, **kwargs):
    return pd.read_csv(register(rel), **kwargs)


def csv(df: pd.DataFrame, name: str, index: bool = False):
    df.to_csv(GEN / name, index=index, float_format="%.15g", lineterminator="\n")


def write_json(data, name: str):
    (GEN / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save(fig, name: str, caption: str, size: tuple[float, float]):
    # Avoid tight bbox resizing: PDF dimensions match intended publication widths.
    fig.savefig(FIG / f"{name}.pdf", metadata={"Creator": "ECS171 paper/tools/build_figures.py", "CreationDate": None, "ModDate": None})
    fig.savefig(FIG / f"{name}.png", dpi=220)
    plt.close(fig)
    figure_metadata[name] = {"width_inches": size[0], "height_inches": size[1], "caption_facts": caption, "pdf": f"figures/{name}.pdf", "preview": f"figures/{name}.png"}


def rate_axis(ax, ylabel=None):
    ax.set_ylim(0, 1.035)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
    ax.set_xticks(PS, ["0", ".25", ".50", ".75", "1"])
    ax.set_xlim(-.025, 1.025)
    ax.set_xlabel("Replacement probability $p$")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", zorder=0)


metrics = read_csv("results/test_metrics.csv")
assert len(metrics) == 45 and metrics["run_id"].is_unique
assert set(metrics["model"]) == set(MODELS)
assert sorted(metrics["seed"].unique().tolist()) == SEEDS
assert (metrics.groupby(["model", "p"]).size() == 3).all()
numeric = ["changed_fraction", "selected_fraction", "train_acc_noisy", "train_acc_clean", "validation_macro_f1", "test_macro_f1", "test_balanced_accuracy", "test_accuracy", "clean_target_gap", "fit_seconds", "warning_count"]
summary = metrics.groupby(["model", "p"])[numeric].agg(["mean", "std"])
summary.columns = [f"{c}_{a}" for c, a in summary.columns]
summary = summary.reset_index()
assert np.isfinite(summary.select_dtypes("number")).all().all()
existing_summary = read_csv("results/model_noise_summary.csv")
for col in ["test_macro_f1_mean", "test_macro_f1_std", "test_balanced_accuracy_mean", "test_balanced_accuracy_std", "train_acc_noisy_mean", "train_acc_clean_mean"]:
    a = summary.sort_values(["model", "p"])[col].to_numpy()
    b = existing_summary.sort_values(["model", "p"])[col].to_numpy()
    assert np.allclose(a, b, atol=1e-13), col
csv(metrics, "main_metrics_45.csv")
csv(summary, "metric_summary_15.csv")
audit = read_json("data/processed/audit.json")
split = read_json("data/processed/split_manifest.json")
eda_meta = read_json("reports/eda/eda_metadata.json")
counts = read_csv("reports/eda/class_counts.csv")
stats = read_csv("reports/eda/feature_statistics.csv", index_col=0)
corr = read_csv("reports/eda/feature_correlation.csv", index_col=0)
features, classes = split["features"], split["classes"]
assert list(stats.index) == features and list(corr.index) == features
assert counts["train_count"].sum() == split["counts"]["train"] == 8125
assert audit["original_rows"] - audit["exact_duplicate_rows"] == audit["cleaned_rows"] == 13543
assert sum(split["counts"].values()) == audit["cleaned_rows"]
csv(counts, "training_class_counts.csv")
csv(stats, "training_feature_statistics.csv", index=True)
csv(corr, "training_feature_correlation.csv", index=True)
mapping = pd.DataFrame({"abbreviation": ABBREV, "source_feature": features})
csv(mapping, "feature_abbreviations.csv")
write_json({"original_rows": audit["original_rows"], "duplicates_removed": audit["exact_duplicate_rows"], "cleaned_rows": audit["cleaned_rows"], "split_counts": split["counts"], "split_seeds": split["seeds"], "classes": classes, "class_abbreviations": dict(zip(CLASS_SHORT, classes)), "scope": "EDA uses training split only"}, "study_design.json")

# 1. Compact honest workflow: data partition, target ownership, and sweep budget.
size = (7.1, 2.65)
fig, ax = plt.subplots(figsize=size)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(x, y, w, h, text, fc="#F3F6F8", ec="#879AA7", fs=8):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=.018", linewidth=.8, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, text, va="center", ha="center", fontsize=fs, linespacing=1.35)


def arrow(start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": "#6F7B84", "lw": .8, "mutation_scale": 9})


box(.015, .735, .20, .19, "Raw Dry Bean\n13,611 rows · 16 features")
box(.28, .735, .20, .19, "Exact deduplication\n68 rows removed")
box(.545, .735, .20, .19, "Clean dataset\n13,543 rows · 7 classes")
box(.805, .735, .18, .19, "Stratified split\n60% / 20% / 20%")
arrow((.222, .83), (.267, .83)); arrow((.487, .83), (.532, .83)); arrow((.752, .83), (.792, .83))
for x, txt in [(.015, "TRAIN · 8,125\nTrain-fitted preprocessing\nFixed noisy labels for 45 main fits"), (.355, "VALIDATION · 2,709\nClean labels\n10 tuning fits; then freeze choices"), (.695, "TEST · 2,709\nClean labels\nEvaluate after all 45 fits finish")]:
    box(x, .265, .29, .31, txt, fc="#EFF4F7" if x == .015 else "#F8F8F7", fs=8)
ax.plot([.895, .895, .16], [.723, .65, .65], color="#87949D", lw=.8)
for x in [.16, .50, .84]:
    arrow((x, .65), (x, .588))
ax.text(.5, .09, "LR / RF / MLP  ×  $p$ ∈ {0, .25, .50, .75, 1}  ×  seeds {17, 29, 43}  =  45 main fits", ha="center", fontsize=8.3)
fig.subplots_adjust(left=.015, right=.985, bottom=.02, top=.98)
save(fig, "study_flow", "68 exact duplicates removed before a fixed stratified 8,125/2,709/2,709 split. Only training labels are randomized. Ten clean-label tuning fits precede 45 main fits; clean-test evaluation follows completion of the whole sweep.", size)

# 2. Training-only class distribution and raw feature scales.
size = (7.1, 3.25)
fig, axes = plt.subplots(1, 2, figsize=size, gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
order = counts.sort_values("train_count")
bars = ax.barh(order["Class"], order["train_count"], color="#517D99", height=.64)
ax.bar_label(bars, labels=[f"{n:,}" for n in order["train_count"]], padding=3, fontsize=8)
ax.set_xlim(0, 2450); ax.set_xlabel("Training samples"); ax.grid(axis="x"); ax.set_axisbelow(True)
ax.set_title("(a) Training class counts", loc="left")
ax = axes[1]
y = np.arange(16)
ax.scatter(stats["std"], y, s=22, color="#805A8D", marker="D", zorder=3)
ax.set_yticks(y, ABBREV); ax.invert_yaxis(); ax.set_xscale("log")
ax.set_xlabel("Raw feature standard deviation (log scale)"); ax.grid(axis="x", which="major")
ax.set_title("(b) Feature scales", loc="left")
fig.subplots_adjust(left=.115, right=.98, bottom=.17, top=.90, wspace=.52)
save(fig, "eda_class_scale", "Training split only (n=8,125). Class counts range from 313 BOMBAY to 2,127 DERMASON. Raw feature standard deviations span several orders of magnitude in their original mixed units; LR/MLP use a training-fitted StandardScaler. Feature abbreviations are recorded in generated/feature_abbreviations.csv.", size)

# 3. Full-width correlation map with explicitly preserved feature ordering.
size = (7.1, 5.7)
fig, ax = plt.subplots(figsize=size)
im = ax.pcolormesh(np.arange(17)-.5, np.arange(17)-.5, corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r", rasterized=False, edgecolors="white", linewidth=.35)
ax.invert_yaxis(); ax.set_aspect("equal")
ax.set_xticks(np.arange(16), ABBREV, rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(np.arange(16), ABBREV)
ax.set_xticks(np.arange(-.5, 16, 1), minor=True); ax.set_yticks(np.arange(-.5, 16, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=.35); ax.tick_params(which="minor", bottom=False, left=False)
for spine in ax.spines.values(): spine.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=.034, pad=.025)
cb.solids.set_rasterized(False)
cb.set_label("Pearson correlation (training split)")
fig.subplots_adjust(left=.14, right=.89, bottom=.15, top=.985)
save(fig, "eda_correlation", "Pearson correlations of all 16 original features in the training split (n=8,125). Complete matrix and abbreviation-to-feature mapping are included as generated CSVs. High correlation among size descriptors is descriptive; all original features were retained.", size)

# 4. Primary and class-balanced evaluation with sample SD, not confidence bands.
size = (7.1, 2.9)
fig, axes = plt.subplots(1, 2, figsize=size)
for ax, metric, name in zip(axes, ["test_macro_f1", "test_balanced_accuracy"], ["(a) Test Macro-F1", "(b) Test balanced accuracy"]):
    for model in MODELS:
        g = summary.loc[summary["model"] == model].sort_values("p")
        ax.errorbar(g["p"], g[metric+"_mean"], yerr=g[metric+"_std"], color=MODEL_COLORS[model], marker=MODEL_MARKERS[model], linestyle=MODEL_DASHES[model], markersize=4, capsize=2.5, elinewidth=.85, label=model, zorder=3)
    rate_axis(ax, "Score"); ax.set_title(name, loc="left")
axes[0].legend(loc="lower left", frameon=False, ncol=3, handlelength=1.7, columnspacing=.8)
fig.subplots_adjust(left=.10, right=.985, bottom=.19, top=.9, wspace=.30)
save(fig, "test_performance", "Means and sample standard deviations (ddof=1) across three seeds on one fixed split; bars are not confidence intervals. Same trained conditions and clean test labels are used in both panels. No Macro-F1 chance baseline is drawn.", size)

# 5. Separate fitting target from original-label training and test targets.
size = (7.1, 2.9)
fig, axes = plt.subplots(1, 3, figsize=size, sharey=True)
targets = [("train_acc_noisy", "Noisy-label train", "#A76E19", "s", "--"), ("train_acc_clean", "Original-label train", "#58766A", "^", "-."), ("test_accuracy", "Clean-label test", "#245B85", "o", "-")]
for ax, model in zip(axes, MODELS):
    g = summary.loc[summary["model"] == model].sort_values("p")
    for col, name, color, marker, dash in targets:
        ax.errorbar(g["p"], g[col+"_mean"], yerr=g[col+"_std"], color=color, marker=marker, linestyle=dash, markersize=3.7, capsize=2, elinewidth=.75, label=name, zorder=3)
    rate_axis(ax); ax.set_title(model, loc="left")
axes[0].set_ylabel("Accuracy")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, 1.01), ncol=3, frameon=False, columnspacing=1.2)
fig.subplots_adjust(left=.10, right=.985, bottom=.2, top=.83, wspace=.17)
save(fig, "fitting_targets", "Accuracy against three explicitly distinct targets. Points are three-seed means and bars sample SD. Noisy-label training and clean-label testing are not the same target, so their difference is not called the conventional generalization gap. RF fits noisy training labels perfectly at every p; finite-budget MLP does not at p=1.", size)

# 6. Endpoint confusion matrices, pooled only for visualization of row proportions.
endpoint_reports = {}
conf_rows, class_rows = [], []
for model in MODELS:
    for p in [0, 1]:
        reports = []
        for seed in SEEDS:
            rel = f"results/class_reports/{model}_p{int(p*100):03d}_s{seed}.json"
            report = read_json(rel)
            assert report["classes"] == classes
            matrix = np.asarray(report["confusion_matrix"], dtype=int)
            assert matrix.shape == (7, 7) and matrix.sum() == 2709
            reports.append(matrix)
            for label in classes:
                item = report["classification"][label]
                class_rows.append({"model": model, "p": p, "seed": seed, "class": label, **item})
        pooled = sum(reports)
        norm = pooled / pooled.sum(axis=1, keepdims=True)
        endpoint_reports[(model, p)] = (pooled, norm)
        for i, truth in enumerate(classes):
            for j, pred in enumerate(classes):
                conf_rows.append({"model": model, "p": p, "true_class": truth, "predicted_class": pred, "pooled_count_3_seeds": int(pooled[i, j]), "row_normalized_fraction": float(norm[i, j]), "unique_test_rows": 2709, "predictions_per_row": 3})
csv(pd.DataFrame(conf_rows), "confusion_endpoints_pooled.csv")
class_df = pd.DataFrame(class_rows)
csv(class_df, "per_class_f1_endpoint_runs.csv")
size = (7.1, 4.85)
fig, axes = plt.subplots(2, 3, figsize=size)
for row, p in enumerate([0, 1]):
    for col, model in enumerate(MODELS):
        ax = axes[row, col]
        _, arr = endpoint_reports[(model, p)]
        im = ax.pcolormesh(np.arange(8)-.5, np.arange(8)-.5, arr, vmin=0, vmax=1, cmap="Blues", rasterized=False)
        ax.invert_yaxis(); ax.set_aspect("equal")
        for i in range(7):
            for j in range(7):
                value = arr[i, j]
                ax.text(j, i, f"{value*100:.0f}", ha="center", va="center", fontsize=7.2, color="white" if value > .55 else "#26343D")
        ax.set_xticks(range(7), CLASS_SHORT, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(7), CLASS_SHORT)
        ax.tick_params(length=0, labelsize=7.5)
        ax.set_title(f"{model} · $p={p}$", fontsize=8.5, pad=3)
        for spine in ax.spines.values(): spine.set_visible(False)
        if col == 0: ax.set_ylabel("True class", fontsize=8)
        if row == 1: ax.set_xlabel("Predicted class", fontsize=8)
fig.subplots_adjust(left=.075, right=.88, bottom=.13, top=.955, wspace=.26, hspace=.41)
cax = fig.add_axes([.907, .20, .016, .64])
cb = fig.colorbar(im, cax=cax); cb.solids.set_rasterized(False); cb.ax.yaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0)); cb.ax.tick_params(labelsize=7.5)
save(fig, "confusion_endpoints", "Endpoint confusion matrices: columns are models; rows p=0 and p=1. Counts from all three seeds are pooled and each true-class row is normalized. Cells show row percentages. The same 2,709 unique test rows are predicted three times, giving 8,127 prediction records per matrix, not 8,127 independent observations. BAR/BOM/CAL/DER/HOR/SEK/SIR map to the seven full class names in study_design.json.", size)

# 7. MLP trajectories and conditional fit-cost/convergence diagnostics.
trace_rows = []
trace_store = {}
for p in PS:
    trace = read_json(f"artifacts/runs/MLP_p{int(p*100):03d}_s17/training_trace.json")
    trace_store[p] = trace["loss_curve"]
    for epoch, loss in enumerate(trace["loss_curve"], 1):
        trace_rows.append({"p": p, "seed": 17, "epoch": epoch, "training_objective": loss})
csv(pd.DataFrame(trace_rows), "mlp_loss_seed17.csv")
mlp = metrics.loc[metrics["model"] == "MLP"].copy()
mlp["iterations_numeric"] = pd.to_numeric(mlp["iterations"])
mlp_diagnostics = mlp.groupby("p").agg(fit_seconds_mean=("fit_seconds", "mean"), fit_seconds_std=("fit_seconds", "std"), iterations_mean=("iterations_numeric", "mean"), iterations_min=("iterations_numeric", "min"), iterations_max=("iterations_numeric", "max"), warning_runs=("warning_count", lambda s: int((s > 0).sum())), warning_count=("warning_count", "sum")).reset_index()
csv(mlp_diagnostics, "mlp_optimization_summary.csv")
size = (7.1, 3.05)
fig, axes = plt.subplots(1, 2, figsize=size)
noise_colors = ["#245B85", "#447B91", "#77855C", "#B38324", "#805A8D"]
noise_dashes = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
for p, color, dash in zip(PS, noise_colors, noise_dashes):
    vals = trace_store[p]
    axes[0].plot(range(1, len(vals)+1), vals, color=color, linestyle=dash, linewidth=1.25, label=f"$p={p:g}$")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training objective")
axes[0].set_xlim(0, 510); axes[0].set_ylim(0, 2.04); axes[0].grid(axis="y")
axes[0].set_title("(a) MLP loss, seed 17", loc="left")
axes[0].legend(frameon=False, ncol=2, loc="center", bbox_to_anchor=(.63, .27), handlelength=2, columnspacing=.8)
g = mlp_diagnostics
bars = axes[1].bar(np.arange(5), g["fit_seconds_mean"], yerr=g["fit_seconds_std"], color="#A6B7C1", edgecolor="#45657A", linewidth=.7, capsize=3, width=.60)
for bar, n in zip(bars, g["warning_runs"]):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2.6, f"{n}/3", ha="center", va="bottom", fontsize=8)
axes[1].set_xticks(np.arange(5), ["0", ".25", ".50", ".75", "1"])
axes[1].set_xlabel("Replacement probability $p$")
axes[1].set_ylabel("Fit time (seconds)"); axes[1].set_ylim(0, max(g["fit_seconds_mean"] + g["fit_seconds_std"])+7)
axes[1].grid(axis="y"); axes[1].set_axisbelow(True)
axes[1].set_title("(b) MLP fit time; warning runs / 3", loc="left")
fig.subplots_adjust(left=.10, right=.985, bottom=.19, top=.90, wspace=.34)
save(fig, "mlp_optimization", "Left: stored loss curves for fixed seed 17 (the first protocol seed) at five p values, with different noisy-label objectives; their absolute levels are not comparable estimates of clean predictive quality. Right: three-seed mean fit seconds with sample-SD bars; labels count runs emitting convergence warnings. 14/15 MLP main fits reach max_iter=500; one clean-label fit stops at 430 epochs. Times describe this local execution, not a hardware-independent benchmark.", size)

# 8. Optional readable per-class endpoint view; report may omit if redundant.
class_summary = class_df.groupby(["model", "p", "class"])["f1-score"].agg(["mean", "std"]).reset_index()
csv(class_summary, "per_class_f1_endpoint_summary.csv")
size = (7.1, 3.0)
fig, axes = plt.subplots(1, 3, figsize=size, sharey=True)
for ax, model in zip(axes, MODELS):
    for p, marker, fill, offset, label in [(0, "o", MODEL_COLORS[model], -.10, "$p=0$"), (1, "s", "white", .10, "$p=1$")]:
        g = class_summary.loc[(class_summary["model"] == model) & (class_summary["p"] == p)].set_index("class").loc[classes]
        ax.errorbar(g["mean"], np.arange(7)+offset, xerr=g["std"], linestyle="none", marker=marker, markerfacecolor=fill, markeredgecolor=MODEL_COLORS[model], ecolor=MODEL_COLORS[model], color=MODEL_COLORS[model], markersize=4.5, capsize=2, elinewidth=.8, label=label)
    ax.set_yticks(range(7), classes); ax.invert_yaxis(); ax.set_xlim(-.025, 1.04)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.xaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Class F1"); ax.grid(axis="x"); ax.set_title(model, loc="left")
axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(1.65, 1.25), ncol=2)
fig.subplots_adjust(left=.13, right=.985, bottom=.19, top=.83, wspace=.17)
save(fig, "per_class_f1", "Per-class F1 at clean and fully randomized endpoints; dots are means and horizontal bars sample SD across three seeds on the same test rows. These per-class means are computed before averaging, whereas pooled confusion-derived F1 would be a different aggregation.", size)

# Copy exact table sources, export audit and reproducibility metadata.
tuning = read_csv("results/tuning.csv")
csv(tuning, "tuning_candidates_10.csv")
write_json(read_json("results/baselines.json"), "baselines.json")
write_json(figure_metadata, "figure_metadata.json")
assert set(sources) == set(REQUIRED_SOURCES)
write_json({
    "builder": "paper/tools/build_figures.py",
    "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_mode": "recorded_sources" if args.recorded_sources else "original_local_artifacts",
    "input_basis": "All 33 exact recorded copies validated against existing provenance SHA256 before any output write; no local-artifact fallback." if args.recorded_sources else "All 33 original local artifacts; exact byte copies recorded under generated/source_data.",
    "sources": sources,
    "transformations": {
        "metric_summary_15.csv": "Group 45 main-run rows by model,p; arithmetic mean and pandas sample std ddof=1; cross-checked against saved model_noise_summary.csv at atol 1e-13.",
        "confusion_endpoints_pooled.csv": "Sum 3 saved integer confusion matrices per endpoint/model, then divide by pooled true-class row totals. Each source matrix has 2,709 rows; same unique rows repeated at three seeds.",
        "per_class_f1_endpoint_summary.csv": "Mean and sample SD across three per-run class F1 values; not F1 recomputed after pooling predictions.",
        "mlp_loss_seed17.csv": "Flatten stored training_trace loss_curve, epoch starting at 1, fixed seed 17 (the first protocol seed).",
        "mlp_optimization_summary.csv": "Aggregate 15 saved MLP records by p; mean/sample-SD seconds; count runs with warning_count>0; extract integer iteration extrema.",
        "EDA": "Existing training-only class/statistics/correlation tables; no validation/test observations used in EDA.",
    },
    "coverage": {"main_run_rows": len(metrics), "conditions": len(summary), "seeds_per_condition": 3, "endpoint_class_report_files": 18, "confusion_cells": len(conf_rows), "unique_test_rows": 2709, "EDA_training_rows": 8125, "model_fits_performed_by_builder": 0},
    "limits": ["One fixed split; three seeds are not independent test datasets.", "Intervals display sample SD, not CI.", "Complete vector PDFs; PNGs are previews.", "No model loading, fitting, tuning, or source result mutations performed."],
    "outputs": {name: {"pdf_sha256": hashlib.sha256((FIG / f"{name}.pdf").read_bytes()).hexdigest(), "png_sha256": hashlib.sha256((FIG / f"{name}.png").read_bytes()).hexdigest()} for name in figure_metadata},
}, "provenance.json")
print(json.dumps({"figures": len(figure_metadata), "main_rows": len(metrics), "condition_rows": len(summary), "source_files": len(sources), "input_mode": "recorded_sources" if args.recorded_sources else "original_local_artifacts", "output_directory": str(FIG), "MLP_warning_runs": int((mlp["warning_count"]>0).sum())}, indent=2))
