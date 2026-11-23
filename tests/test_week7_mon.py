"""Week 7 Monday: real performance on known outcomes, and the promote-or-keep decision.

# DO NOT EDIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.champion import decide
from src.outcomes import PERFORMANCE_KEYS, performance
from src.retrain import recent_holdout

COSTS = {"approve_defaulter": 10, "reject_good": 1}
# 10 applicants: 3 defaulted. Threshold 0.5: scores >= 0.5 are rejected.
Y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
P = np.array([0.9, 0.6, 0.2, 0.7, 0.4, 0.3, 0.2, 0.1, 0.1, 0.05])


# DO NOT EDIT
def test_performance_counts_and_rates() -> None:
    p = performance(Y, P, 0.5, COSTS)
    assert set(p) == set(PERFORMANCE_KEYS)
    assert (p["n"], p["n_default"]) == (10, 3)
    assert p["default_rate"] == pytest.approx(0.3)
    assert p["mean_score"] == pytest.approx(P.mean())
    # Rejected: 0.9, 0.6 (defaulters) and 0.7 (a good payer). Approved: the other 7.
    assert p["approval_rate"] == pytest.approx(0.7)
    assert p["bad_rate_approved"] == pytest.approx(1 / 7)


# DO NOT EDIT
def test_performance_cost_and_auc() -> None:
    p = performance(Y, P, 0.5, COSTS)
    # One approved defaulter (cost 10) and one rejected good payer (cost 1), over 10.
    assert p["cost"] == pytest.approx(1.1)
    assert p["auc"] == pytest.approx(16.5 / 21)
    assert all(isinstance(v, (int, float)) for v in p.values())


# DO NOT EDIT
def test_performance_when_nobody_is_approved() -> None:
    p = performance(Y, P, 0.0, COSTS)
    assert p["approval_rate"] == 0.0
    assert np.isnan(p["bad_rate_approved"]), "nobody approved: the bad rate is unknown (NaN)"


def _comparison(champ_cost: float, chall_cost: float) -> dict:
    base = {"auc": 0.75, "n": 500, "n_default": 60}
    return {"validation": {"champion": {**base, "cost": 0.6}, "challenger": {**base, "cost": 0.6}},
            "recent": {"champion": {**base, "cost": champ_cost},
                       "challenger": {**base, "cost": chall_cost}}}


# DO NOT EDIT
def test_decide_keeps_the_champion_when_the_gate_fails() -> None:
    result = decide(_comparison(0.8, 0.5), ["validation AUC 0.690 is below the floor 0.7"],
                    {"max_cost_increase": 0.0})
    assert result["decision"] == "keep"
    assert result["reasons"] == ["gate: validation AUC 0.690 is below the floor 0.7"]


# DO NOT EDIT
def test_decide_keeps_the_champion_when_the_challenger_costs_more() -> None:
    result = decide(_comparison(0.722, 0.764), [], {"max_cost_increase": 0.0})
    assert result["decision"] == "keep"
    text = " ".join(result["reasons"])
    assert "0.764" in text and "0.722" in text, "give both recent costs"


# DO NOT EDIT
def test_decide_promotes_a_cheaper_challenger() -> None:
    result = decide(_comparison(0.722, 0.700), [], {"max_cost_increase": 0.0})
    assert result["decision"] == "promote"
    assert "0.700" in " ".join(result["reasons"])


# DO NOT EDIT
def test_decide_respects_the_tolerance() -> None:
    assert decide(_comparison(0.70, 0.71), [], {"max_cost_increase": 0.02})["decision"] == \
        "promote"
    assert decide(_comparison(0.70, 0.73), [], {"max_cost_increase": 0.02})["decision"] == "keep"


# DO NOT EDIT
def test_recent_holdout_is_fixed() -> None:
    ids = pd.Series(range(100000, 102000))
    a, b = recent_holdout(ids, 0.3, 2027), recent_holdout(ids[::-1].reset_index(drop=True),
                                                         0.3, 2027)
    assert 0.25 < a.mean() < 0.35
    assert set(ids[a]) == set(ids[::-1].reset_index(drop=True)[b]), "same ids, same slice"
