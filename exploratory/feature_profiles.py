"""Print SIRA and DERMASON feature quartiles for the Group 8 notebook's training split."""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA = Path(__file__).resolve().parents[1] / "data/raw/Dry_Bean_Dataset.xlsx"


def feature_profiles(workbook):
    beans = pd.read_excel(workbook).drop_duplicates().reset_index(drop=True)
    train, _ = train_test_split(
        beans, test_size=0.30, stratify=beans["Class"], random_state=17
    )
    rows = []
    for feature in train.columns.drop("Class"):
        row = {"feature": feature}
        for name in ("SIRA", "DERMASON"):
            values = train.loc[train["Class"] == name, feature]
            q1, median, q3 = values.quantile([0.25, 0.50, 0.75])
            row.update({
                f"{name}_n": len(values),
                f"{name}_q1": q1,
                f"{name}_median": median,
                f"{name}_q3": q3,
            })
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RAW_DATA)
    args = parser.parse_args()
    feature_profiles(args.data).to_csv(sys.stdout, index=False, float_format="%.9g")
