"""Compare feature subsets on the clean validation split (seed 17)."""
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score

from src.core import prepare_frame, make_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "exploratory"

K_VALUES = [4, 6, 8, 10, 12, 14, 16]
CORR_CUTOFF = 0.95
SEED = 17


def load_data():
    protocol = json.loads((ROOT / "config" / "protocol.json").read_text(encoding="utf-8"))
    selected = json.loads((ROOT / "config" / "selected_params.json").read_text(encoding="utf-8"))
    features = protocol["dataset"]["features"]
    classes = protocol["dataset"]["classes"]

    raw = pd.read_excel(ROOT / "data" / "raw" / "Dry_Bean_Dataset.xlsx")
    _, splits, _ = prepare_frame(raw, features, classes)
    train, val = splits["train"], splits["validation"]

    # sanity check: same split sizes as the report
    assert len(train) == 8125 and len(val) == 2709
    return protocol, selected, features, classes, train, val


def rank_features(train, features):
    f_scores, p_values = f_classif(train[features], train["Class"])
    table = pd.DataFrame({"feature": features, "f_score": f_scores, "p_value": p_values})
    table = table.sort_values("f_score", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def correlation_pruned(train, ranking, cutoff):
    # go through features from best to worst F-score and skip any feature
    # that is too correlated with one we already kept
    corr = train[ranking["feature"]].corr().abs()
    kept = []
    for feat in ranking["feature"]:
        if all(corr.loc[feat, k] <= cutoff for k in kept):
            kept.append(feat)
    return kept


def fit_and_score(model_name, protocol, selected, cols, train, val, classes):
    params = selected["selected"][model_name]["params"]
    model = make_model(model_name, protocol["models"][model_name], params, SEED)

    y_train = train["Class"].map({c: i for i, c in enumerate(classes)})
    y_val = val["Class"].map({c: i for i, c in enumerate(classes)})

    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)  # newer sklearn warns about penalty="l2"
        model.fit(train[cols], y_train)
    seconds = time.perf_counter() - start

    pred = model.predict(val[cols])
    f1 = f1_score(y_val, pred, labels=range(len(classes)), average="macro", zero_division=0)
    return f1, seconds


def main():
    protocol, selected, features, classes, train, val = load_data()

    ranking = rank_features(train, features)
    pruned = correlation_pruned(train, ranking, CORR_CUTOFF)
    print("ANOVA ranking (training rows only):")
    print(ranking[["rank", "feature", "f_score"]].to_string(index=False))
    print(f"\nCorrelation-pruned set (|r| <= {CORR_CUTOFF}), {len(pruned)} features:")
    print(", ".join(pruned))

    feature_sets = [("anova_top_k", k, list(ranking["feature"][:k])) for k in K_VALUES]
    feature_sets.append(("corr_pruned", len(pruned), pruned))

    rows = []
    for model_name in ["LR", "RF", "MLP"]:
        for method, k, cols in feature_sets:
            f1, seconds = fit_and_score(model_name, protocol, selected, cols, train, val, classes)
            rows.append({"model": model_name, "method": method, "n_features": k,
                         "val_macro_f1": f1, "fit_seconds": seconds,
                         "features": ";".join(cols)})
            print(f"{model_name:3s} {method:12s} k={k:2d}  val Macro-F1 = {f1:.4f}")

    results = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUT / "feature_anova_scores.csv", index=False)
    results.to_csv(OUT / "feature_selection.csv", index=False)


if __name__ == "__main__":
    main()
