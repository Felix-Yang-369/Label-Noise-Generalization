"""Compare feature rankings and correlation pruning on the two project training splits."""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from exploratory.feature_selection import correlation_pruned, rank_features
from src.core import prepare_frame

ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = (0.90, 0.95, 0.98)


def main():
    raw = pd.read_excel(ROOT / "data/raw/Dry_Bean_Dataset.xlsx")
    config = json.loads((ROOT / "config/protocol.json").read_text())["dataset"]
    features = config["features"]
    _, splits, _ = prepare_frame(raw, features, config["classes"])
    main_train = splits["train"]
    beans = raw.drop_duplicates()
    notebook_train, _ = train_test_split(
        beans, test_size=0.30, random_state=17, stratify=beans["Class"]
    )
    shared = len(set(main_train["row_id"]) & set(notebook_train.index + 2))
    print(f"Training sizes: {len(main_train)} and {len(notebook_train)}; shared beans: {shared}")
    comparison = pd.DataFrame(index=pd.Index(features, name="feature"))
    for name, train in (("main", main_train), ("notebook", notebook_train)):
        ranking = rank_features(train, features)
        comparison[f"{name}_rank"] = ranking.set_index("feature")["rank"]
        for cutoff in CUTOFFS:
            kept = correlation_pruned(train, ranking, cutoff)
            comparison[f"{name}_kept_{cutoff:.2f}"] = comparison.index.isin(kept)
    comparison = comparison.sort_values("main_rank")
    for cutoff in CUTOFFS:
        a = comparison[f"main_kept_{cutoff:.2f}"]
        b = comparison[f"notebook_kept_{cutoff:.2f}"]
        print(f"{cutoff:.2f}: {a.sum()} main, {b.sum()} notebook, {(a & b).sum()} in both")
    comparison.to_csv(ROOT / "results/exploratory/feature_stability.csv")


if __name__ == "__main__":
    main()
