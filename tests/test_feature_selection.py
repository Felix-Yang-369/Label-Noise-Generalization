import numpy as np
import pandas as pd

from exploratory.feature_selection import correlation_pruned, rank_features


def toy_data():
    rng = np.random.default_rng(0)
    labels = np.repeat(["X", "Y", "Z"], 50)
    a = np.repeat([0.0, 5.0, 10.0], 50) + rng.normal(0, 1, 150)
    return pd.DataFrame({
        "a": a,
        "a_copy": 2 * a + rng.normal(0, 0.01, 150),
        "noise": rng.normal(0, 1, 150),
        "Class": labels,
    })


def test_rank_puts_informative_features_first():
    ranking = rank_features(toy_data(), ["noise", "a", "a_copy"])
    assert list(ranking["rank"]) == [1, 2, 3]
    assert ranking["feature"].iloc[-1] == "noise"
    assert ranking["f_score"].is_monotonic_decreasing


def test_pruning_drops_near_duplicate_but_keeps_the_rest():
    data = toy_data()
    ranking = rank_features(data, ["a", "a_copy", "noise"])
    kept = correlation_pruned(data, ranking, cutoff=0.95)
    assert len({"a", "a_copy"} & set(kept)) == 1
    assert kept[0] == ranking["feature"].iloc[0]
    assert "noise" in kept


def test_pruning_with_cutoff_one_keeps_everything():
    data = toy_data()
    ranking = rank_features(data, ["a", "a_copy", "noise"])
    assert sorted(correlation_pruned(data, ranking, cutoff=1.0)) == ["a", "a_copy", "noise"]
