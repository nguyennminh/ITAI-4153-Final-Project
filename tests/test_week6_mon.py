"""Week 6 Monday: the population stability index, and the drift table built on it.

# DO NOT EDIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring import category_psi, drift_table, psi

RNG = np.random.default_rng(0)
REF = pd.Series(RNG.normal(0, 1, 5000))
PSI_SMALL = 0.05  # two samples of the same distribution, 1,000 rows: noise only


# DO NOT EDIT
def test_psi_known_value() -> None:
    reference = np.array([0.0, 0.0, 1.0, 1.0])
    current = np.array([0.0, 1.0, 1.0, 1.0])
    expected = (0.25 - 0.5) * np.log(0.25 / 0.5) + (0.75 - 0.5) * np.log(0.75 / 0.5)
    value = psi(reference, current, edges=np.array([0.5]))
    assert isinstance(value, float)
    assert value == pytest.approx(expected, rel=1e-3)


# DO NOT EDIT
def test_psi_same_distribution_is_small() -> None:
    assert psi(REF, REF) == pytest.approx(0.0, abs=1e-9)
    assert psi(REF, pd.Series(RNG.normal(0, 1, 1000))) < PSI_SMALL


# DO NOT EDIT
def test_psi_grows_with_the_shift() -> None:
    small = psi(REF, REF + 0.1)
    large = psi(REF, REF + 1.0)
    assert 0 <= small < 0.1 < 0.25 < large


# DO NOT EDIT
def test_psi_counts_missing_values() -> None:
    current = REF.copy()
    current[RNG.random(len(current)) < 0.5] = np.nan
    assert psi(REF, current) > 0.25, "values going missing is drift too"


# DO NOT EDIT
def test_psi_uses_reference_bins() -> None:
    current = pd.Series(RNG.normal(3, 1, 1000))
    assert psi(REF, current) > psi(current, current) + 1


# DO NOT EDIT
def test_category_psi() -> None:
    ref = pd.Series(["F"] * 650 + ["M"] * 350)
    assert category_psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert category_psi(ref, pd.Series(["F"] * 350 + ["M"] * 650)) > 0.25


# DO NOT EDIT
def test_drift_table() -> None:
    reference = pd.DataFrame({"income": REF * 1000 + 50_000, "score": REF,
                              "gender": ["F", "M"] * 2500})
    current = reference.sample(1000, random_state=1).reset_index(drop=True)
    current["income"] *= 0.7
    table = drift_table(reference, current, categorical=["gender"])
    assert table.iloc[0]["feature"] == "income" and table.iloc[0]["level"] == "major"
    assert table.set_index("feature").loc["score", "level"] == "none"
    assert {"psi", "ks_stat", "ks_pvalue", "missing_ref", "missing_cur"} <= set(table.columns)
