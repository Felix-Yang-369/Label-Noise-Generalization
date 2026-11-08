"""Export the six demo models using the selected Group 8 notebook protocol.

Run from the repository root:
    python integration/elsey/export_demo_models.py

This intentionally writes to a separate directory. It does not overwrite the
legacy frozen experiment, results, or existing deployed model artifacts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/Dry_Bean_Dataset.xlsx"
DEST = ROOT / "integration/elsey/demo_artifacts"
CONFIGS = {
    "LR": {"k": 16, "C": 10},
    "RF": {"k": 16, "depth": 12, "leaf": 1},
    "MLP": {"k": 16, "alpha": 0.01, "lr": 0.001},
}


def model_for(name: str, seed: int = 17) -> Pipeline:
    cfg = CONFIGS[name]
    steps = [] if name == "RF" else [("scaler", StandardScaler())]
    if name == "LR":
        estimator = LogisticRegression(C=cfg["C"], max_iter=800, random_state=seed)
    elif name == "RF":
        estimator = RandomForestClassifier(n_estimators=200, max_depth=cfg["depth"],
                                           min_samples_leaf=cfg["leaf"], random_state=seed,
                                           n_jobs=-1)
    elif name == "MLP":
        estimator = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=cfg["alpha"],
                                  learning_rate_init=cfg["lr"], max_iter=180,
                                  early_stopping=False, random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline(steps + [("model", estimator)])


def randomized_labels(y, probability: float, seed: int, classes):
    """Exact same RNG order and sampling logic as the selected notebook."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(y).copy()
    mask = rng.random(len(labels)) < probability
    labels[mask] = rng.choice(classes, size=int(mask.sum()), replace=True)
    return labels


def export():
    df = pd.read_excel(RAW).drop_duplicates().reset_index(drop=True)
    features = [column for column in df if column != "Class"]
    if df.shape != (13543, 17) or len(features) != 16:
        raise ValueError("Unexpected UCI Dry Bean dataset; refusing to export")
    x, y = df[features], df["Class"]
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=17)
    x_val, _x_test, y_val, _y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=17)
    if (len(x_train), len(x_val), len(_x_test)) != (9480, 2031, 2032):
        raise ValueError("Notebook's train/validation/test split did not match")
    classes = np.sort(y.unique())
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "integration/elsey/ECS171_Group8_Project.ipynb",
        "raw_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        "features": features, "classes": classes.tolist(),
        "split_sizes": {"train": len(x_train), "validation": len(x_val), "test": len(_x_test)},
        "seed": 17, "model_configs": CONFIGS, "models": {},
        "examples": [],
        "ranges": {f: [float(x_train[f].min()), float(x_train[f].max())] for f in features},
    }
    for label in classes:
        row = x_val.loc[y_val == label].iloc[0]
        manifest["examples"].append({"class": label, "source": "validation",
                                     "features": {f: float(row[f]) for f in features}})
    for name in CONFIGS:
        for condition, p in (("clean", 0.0), ("random", 1.0)):
            noisy = randomized_labels(y_train, p, 17, classes)
            estimator = model_for(name)
            estimator.fit(x_train, noisy)
            predictions_train = estimator.predict(x_train)
            predictions_val = estimator.predict(x_val)
            filename = f"{name}_{condition}.joblib"
            target = DEST / filename
            joblib.dump(estimator, target, compress=3)
            manifest["models"][f"{name}_{condition}"] = {
                "file": filename, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "train_noisy_accuracy": float(accuracy_score(noisy, predictions_train)),
                "validation_macro_f1": float(f1_score(y_val, predictions_val, average="macro")),
                "params": CONFIGS[name], "noise_p": p,
            }
            print(name, condition, manifest["models"][f"{name}_{condition}"])
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Saved:", DEST)


if __name__ == "__main__":
    export()
