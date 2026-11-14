"""Independent data, split, and EDA checks that run on a fresh clone.

These tests read only tracked files (the raw UCI archive, recorded manifests,
EDA tables, and result CSVs). They do not need trained models, processed
splits, or noise files, and they never fit a model.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/generated/source_data/data/processed"
PROTOCOL = json.loads((ROOT / "config/protocol.json").read_text(encoding="utf-8"))
FEATURES = PROTOCOL["dataset"]["features"]
CLASSES = PROTOCOL["dataset"]["classes"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def manifest():
    return json.loads((SOURCE / "split_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def raw():
    frame = pd.read_excel(ROOT / "data/raw/Dry_Bean_Dataset.xlsx")
    # Recorded row identity: original Excel row number, first data row = 2.
    frame.insert(0, "row_id", np.arange(len(frame)) + 2)
    return frame


@pytest.fixture(scope="module")
def splits(raw):
    cleaned = raw.drop_duplicates(subset=FEATURES + ["Class"], keep="first")
    remaining, test = train_test_split(
        cleaned, test_size=0.20, stratify=cleaned["Class"], random_state=171
    )
    train, validation = train_test_split(
        remaining, test_size=0.25, stratify=remaining["Class"], random_state=172
    )
    return {"train": train, "validation": validation, "test": test}


def test_raw_files_match_recorded_hashes(manifest):
    assert sha256(ROOT / "data/raw/Dry_Bean_Dataset.xlsx") == manifest["raw_xlsx_sha256"]
    assert sha256(ROOT / "data/raw/UCI_Dry_Bean_Original.zip") == manifest["raw_archive_sha256"]


def test_audit_counts_recomputed_from_raw(raw):
    audit = json.loads((SOURCE / "audit.json").read_text(encoding="utf-8"))
    duplicated = raw.duplicated(subset=FEATURES + ["Class"], keep="first")
    assert list(raw.columns[1:]) == FEATURES + ["Class"]
    assert len(raw) == audit["original_rows"] == 13611
    assert int(raw[FEATURES].isna().sum().sum()) == audit["missing_cells"] == 0
    assert np.isfinite(raw[FEATURES].to_numpy(dtype=float)).all()
    assert int(duplicated.sum()) == audit["exact_duplicate_rows"] == 68
    # The report states that every removed duplicate belongs to HOROZ.
    assert set(raw.loc[duplicated, "Class"]) == {"HOROZ"}
    # No identical feature vector carries two different labels.
    assert (raw.groupby(FEATURES)["Class"].nunique() == 1).all()
    cleaned_counts = raw.loc[~duplicated, "Class"].value_counts().to_dict()
    assert cleaned_counts == audit["class_counts_cleaned"]
    assert sum(cleaned_counts.values()) == audit["cleaned_rows"] == 13543


def test_split_reproduces_recorded_row_ids(splits, manifest):
    for name, frame in splits.items():
        assert len(frame) == manifest["counts"][name]
        assert sorted(frame["row_id"]) == sorted(manifest["rows"][name]), name


def test_splits_are_disjoint_and_cover_cleaned_data(splits):
    ids = {name: set(frame["row_id"]) for name, frame in splits.items()}
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]
    assert sum(map(len, ids.values())) == 13543
    # Stronger than row ids: no identical feature vector crosses partitions.
    vectors = {
        name: set(map(tuple, frame[FEATURES].to_numpy())) for name, frame in splits.items()
    }
    assert not vectors["train"] & vectors["test"]
    assert not vectors["train"] & vectors["validation"]
    assert not vectors["validation"] & vectors["test"]
    for frame in splits.values():
        assert set(frame["Class"]) == set(CLASSES)


def test_eda_correlations_are_training_only(splits):
    recorded = pd.read_csv(ROOT / "reports/eda/feature_correlation.csv", index_col=0)
    recomputed = splits["train"][FEATURES].corr()
    pd.testing.assert_frame_equal(
        recorded.loc[FEATURES, FEATURES], recomputed, check_exact=False, atol=1e-9
    )
    # Values quoted in the report (Section III-B).
    assert recomputed.loc["Area", "ConvexArea"] == pytest.approx(0.999940, abs=5e-7)
    assert recomputed.loc["Compactness", "ShapeFactor3"] == pytest.approx(0.998683, abs=5e-7)
    assert recomputed.loc["Perimeter", "EquivDiameter"] == pytest.approx(0.991365, abs=5e-7)
    # Training-only correlations differ from all-data correlations, so this
    # also guards against EDA being computed on validation or test rows.
    everything = pd.concat(splits.values())[FEATURES].corr()
    assert not np.allclose(recorded.to_numpy(), everything.to_numpy(), atol=1e-12)


def test_majority_baseline_recomputed(splits):
    recorded = json.loads((ROOT / "results/baselines.json").read_text(encoding="utf-8"))
    majority = splits["train"]["Class"].value_counts().idxmax()
    assert majority == recorded["majority_class"] == "DERMASON"
    truth = splits["test"]["Class"].to_numpy()
    hit = truth == majority
    accuracy = hit.mean()
    recall = [hit[truth == c].mean() for c in CLASSES]
    precision = hit.mean()  # every prediction is the majority class
    f1_majority = 2 * precision * 1.0 / (precision + 1.0)
    macro_f1 = f1_majority / len(CLASSES)  # other six classes have F1 = 0
    expected = recorded["clean_training_majority_test"]
    assert accuracy == pytest.approx(expected["accuracy"])
    assert np.mean(recall) == pytest.approx(expected["balanced_accuracy"])
    assert macro_f1 == pytest.approx(expected["macro_f1"])


def test_realized_noise_rates_follow_six_sevenths_rule():
    runs = pd.read_csv(ROOT / "results/test_metrics.csv")
    n_train = 8125
    for p, group in runs.groupby("p"):
        per_seed = group.groupby("seed")[["selected_fraction", "changed_fraction"]].first()
        expected_changed = p * 6 / 7
        if p == 0:
            assert (per_seed == 0).all().all()
            continue
        # Each seed's rate is a binomial proportion over 8,125 training labels;
        # allow 4 standard errors so the check flags mistakes, not chance.
        tol_sel = 4 * np.sqrt(p * (1 - p) / n_train) + 1e-12
        tol_chg = 4 * np.sqrt(expected_changed * (1 - expected_changed) / n_train)
        assert (per_seed["selected_fraction"] - p).abs().max() <= max(tol_sel, 1e-12), p
        assert (per_seed["changed_fraction"] - expected_changed).abs().max() <= tol_chg, p
        # Redraws can return the original class, so changed <= selected.
        assert (per_seed["changed_fraction"] <= per_seed["selected_fraction"]).all()
