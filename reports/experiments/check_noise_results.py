"""Audit Group 8's published CSVs. Usage: python3 check_noise_results.py path/to/results"""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev

MODELS = ("LR", "RF", "MLP")
LEVELS = (0, .25, .5, .75, 1)
SEEDS = {17, 29, 43}
FIELDS = ("test_macro_f1", "test_accuracy", "test_balanced_accuracy",
          "train_acc_noisy", "train_acc_clean", "changed_fraction")


def read_rows(folder, filename):
    with (folder / filename).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def check(condition, description):
    if not condition:
        raise ValueError(description)


def audit(folder):
    runs = read_rows(folder, "test_metrics.csv")
    summaries = read_rows(folder, "model_noise_summary.csv")
    check(len(runs) == 45 and len(summaries) == 15, "Unexpected table size")
    for model in MODELS:
        for level in LEVELS:
            def matches(row):
                return row["model"] == model and float(row["p"]) == level
            group = [row for row in runs if matches(row)]
            summary = [row for row in summaries if matches(row)]
            check(len(group) == 3 and {int(r["seed"]) for r in group} == SEEDS,
                  f"Incomplete or duplicate runs: {model}, p={level}")
            check(len(summary) == 1, f"Missing or repeated summary: {model}, p={level}")
            for field in FIELDS:
                values = [float(row[field]) for row in group]
                for suffix, stat in (("mean", mean), ("std", stdev)):
                    recorded = float(summary[0][f"{field}_{suffix}"])
                    check(math.isclose(stat(values), recorded, rel_tol=0, abs_tol=1e-12),
                          f"Summary mismatch: {model}, p={level}, {field}_{suffix}")
    print("Verified 45 run identities and 180 summary statistics across 15 conditions.")
    for model in MODELS:
        def score(level, seed):
            return next(float(r["test_macro_f1"]) for r in runs
                        if r["model"] == model and float(r["p"]) == level
                        and int(r["seed"]) == seed)
        drops = [score(0, seed) - score(.75, seed) for seed in sorted(SEEDS)]
        print(f"{model}: clean-test Macro-F1 decrease from p=0 to p=.75 "
              f"by seed (17, 29, 43): " + ", ".join(f"{v:.4f}" for v in drops)
              + f"; mean {mean(drops):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Directory containing the two CSV files")
    audit(parser.parse_args().results)
