"""Check the algebraic definitions of the Dry Bean shape features.

Several of the 16 features are exact functions of Area, Perimeter, and the two
axis lengths. These identities explain why the EDA finds near-perfect
correlations (for example, ShapeFactor3 is exactly Compactness squared).
ShapeFactor2 is not checked: its released values do not reduce to a simple
formula of the other columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def beans():
    return pd.read_excel(ROOT / "data/raw/Dry_Bean_Dataset.xlsx")


IDENTITIES = {
    "AspectRation": lambda d: d.MajorAxisLength / d.MinorAxisLength,
    "Eccentricity": lambda d: np.sqrt(1 - (d.MinorAxisLength / d.MajorAxisLength) ** 2),
    "EquivDiameter": lambda d: np.sqrt(4 * d.Area / np.pi),
    "Solidity": lambda d: d.Area / d.ConvexArea,
    "roundness": lambda d: 4 * np.pi * d.Area / d.Perimeter**2,
    "Compactness": lambda d: d.EquivDiameter / d.MajorAxisLength,
    "ShapeFactor1": lambda d: d.MajorAxisLength / d.Area,
    "ShapeFactor3": lambda d: 4 * d.Area / (np.pi * d.MajorAxisLength**2),
    "ShapeFactor4": lambda d: d.Area / (np.pi * d.MajorAxisLength * d.MinorAxisLength / 4),
}


@pytest.mark.parametrize("feature", sorted(IDENTITIES))
def test_feature_matches_its_definition(beans, feature):
    np.testing.assert_allclose(beans[feature], IDENTITIES[feature](beans), rtol=1e-12)


def test_shapefactor3_is_compactness_squared(beans):
    np.testing.assert_allclose(beans.ShapeFactor3, beans.Compactness**2, rtol=1e-12)


def test_area_features_are_integer_pixel_counts(beans):
    for column in ("Area", "ConvexArea"):
        assert pd.api.types.is_integer_dtype(beans[column])
        assert (beans[column] > 0).all()
    assert (beans.ConvexArea >= beans.Area).all()
