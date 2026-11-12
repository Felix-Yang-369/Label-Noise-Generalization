import pandas as pd
import pytest

from exploratory.feature_profiles import RAW_DATA, main_train, notebook_train, profiles


def test_profiles_report_quartiles_per_class():
    train = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 7.0],
        "Class": ["SIRA"] * 5 + ["DERMASON"] * 5 + ["BOMBAY"],
    })
    row = profiles(train).iloc[0]
    assert row["feature"] == "a"
    assert (row["SIRA_n"], row["DERMASON_n"]) == (5, 5)
    assert (row["SIRA_q1"], row["SIRA_median"], row["SIRA_q3"]) == (2.0, 3.0, 4.0)
    assert (row["DERMASON_q1"], row["DERMASON_median"], row["DERMASON_q3"]) == (20.0, 30.0, 40.0)


@pytest.mark.parametrize("split, rows, sira, dermason", [
    (main_train, 8125, 1582, 2127),
    (notebook_train, 9480, 1845, 2482),
])
def test_training_splits_match_the_project(split, rows, sira, dermason):
    train = split(RAW_DATA)
    assert len(train) == rows
    assert list(train.columns)[-1] == "Class" and train.shape[1] == 17
    counts = train["Class"].value_counts()
    assert (counts["SIRA"], counts["DERMASON"]) == (sira, dermason)
