"""Print SIRA and DERMASON feature quartiles on a project training split.

--split main      the main experiment's training split (8,125 rows; seeds 171 and 172)
--split notebook  the Group 8 notebook's training split (9,480 rows; random state 17)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.core import prepare_frame

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data/raw/Dry_Bean_Dataset.xlsx"
CLASSES = ("SIRA", "DERMASON")


def main_train(workbook):
    dataset = json.loads((ROOT / "config/protocol.json").read_text(encoding="utf-8"))["dataset"]
    _, splits, _ = prepare_frame(pd.read_excel(workbook), dataset["features"], dataset["classes"])
    return splits["train"].drop(columns="row_id")


def notebook_train(workbook):
    beans = pd.read_excel(workbook).drop_duplicates().reset_index(drop=True)
    train, _ = train_test_split(
        beans, test_size=0.30, stratify=beans["Class"], random_state=17
    )
    return train


def profiles(train):
    rows = []
    for feature in train.columns.drop("Class"):
        row = {"feature": feature}
        for name in CLASSES:
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


def feature_profiles(workbook, split="notebook"):
    train = {"main": main_train, "notebook": notebook_train}[split](workbook)
    return profiles(train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=RAW_DATA)
    parser.add_argument("--split", choices=["main", "notebook"], default="notebook")
    args = parser.parse_args()
    feature_profiles(args.data, args.split).to_csv(sys.stdout, index=False, float_format="%.9g")
