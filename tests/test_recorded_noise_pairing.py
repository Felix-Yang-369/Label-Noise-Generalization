from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_noise_rates_match_across_models():
    runs = pd.read_csv(ROOT / "results/test_metrics.csv")
    assert len(runs) == 45
    assert set(runs["seed"]) == {17, 29, 43}
    assert set(runs["p"]) == {0, 0.25, 0.5, 0.75, 1}

    for condition, group in runs.groupby(["seed", "p"], dropna=False):
        assert sorted(group["model"]) == ["LR", "MLP", "RF"], condition
        for column in ("selected_fraction", "changed_fraction"):
            rates = group[column]
            assert rates.between(0, 1).all(), (condition, column)
            assert rates.nunique() == 1, (condition, column)
