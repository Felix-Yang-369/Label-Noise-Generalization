from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def test_reported_training_class_counts():
    beans = pd.read_excel(ROOT / "data/raw/Dry_Bean_Dataset.xlsx").drop_duplicates()
    remaining, _ = train_test_split(
        beans, test_size=0.20, stratify=beans["Class"], random_state=171
    )
    train, _ = train_test_split(
        remaining, test_size=0.25, stratify=remaining["Class"], random_state=172
    )

    counts = train["Class"].value_counts().sort_index()
    reported = pd.read_csv(ROOT / "reports/eda/class_counts.csv", index_col="Class")
    assert len(counts) == 7
    assert counts.sum() == 8125
    pd.testing.assert_series_equal(
        reported["train_count"].sort_index(), counts, check_names=False
    )
