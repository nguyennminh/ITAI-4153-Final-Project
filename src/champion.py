"""Champion or challenger? Compare them, run your quality gate, decide.

Run it with ``python run.py champion`` after ``python run.py retrain``. It:

1. scores the champion (alias ``staging``, the model your service runs) and the challenger
   (alias ``challenger``) on two FIXED slices that neither trained on:

       validation   the 1,000 validation applicants (the slice every week has used)
       recent       the labeled production applicants kept aside by ``retrain``

   with your ``performance`` (src/outcomes.py), each model at its own threshold, plus a
   paired 95% bootstrap interval of the difference (challenger minus champion);
2. runs the model tests (Week 4) on the challenger;
3. applies YOUR Week 4 quality gate (``gate_failures`` in scripts/quality_gate.py, with
   config/gate.yaml) to the challenger, with the champion as the model it may not fall
   behind;
4. calls YOUR ``decide`` with the comparison, the gate's failures, and the rule in
   config/retrain.yaml;
5. writes reports/retrain/comparison.json, comparison.md and decision.json.

It changes nothing: promoting is a separate step (``python run.py promote``), so a person
reads the decision before it happens.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.quality_gate import gate_failures
from src.outcomes import performance
from src.retrain import (
    CHALLENGER,
    OUT,
    champion_record,
    load_retrain_config,
    retrain_data,
)
from src.uncertainty import bootstrap_auc_diff_ci

GATE_CONFIG = Path("config/gate.yaml")
SLICES = ("validation", "recent")


# TODO(student)
def decide(comparison: dict[str, Any], failures: list[str],
           rule: dict[str, Any]) -> dict[str, Any]:
    """Promote the challenger, or keep the champion. Always say why.

    Args:
        comparison: for each slice ("validation", "recent"): ``{"champion": {...},
            "challenger": {...}}``, each a ``performance`` dict (keys: auc, cost, ...).
        failures: what your quality gate says is wrong with the challenger (empty = passes).
        rule: config/retrain.yaml's ``decision`` section, with ``max_cost_increase``.

    Returns:
        ``{"decision": "promote" or "keep", "reasons": [sentences]}``, where:

        1. If the gate failed: "keep", and the reasons are the gate's failures, each
           starting with "gate: ".
        2. Else, if the challenger's expected cost on the RECENT slice is above the
           champion's plus ``max_cost_increase``: "keep", with a reason that gives both
           costs (3 decimals).
        3. Else: "promote", with a reason that gives both recent costs.

    The recent slice decides because it looks like the applicants the model will see next;
    the validation slice is already guarded by the gate.
    """
    raise NotImplementedError("TODO(student): promote or keep, with the reasons")


# PROVIDED
def cost_diff_ci(y: np.ndarray, new: np.ndarray, t_new: float, old: np.ndarray, t_old: float,
                 costs: dict[str, float], n_boot: int, seed: int) -> tuple[float, float]:
    """95% paired bootstrap interval of cost(new) - cost(old) per applicant."""
    rng = np.random.default_rng(seed)
    w_bad, w_good = costs["approve_defaulter"], costs["reject_good"]
    cost_new = np.where(y == 1, w_bad * (new < t_new), w_good * (new >= t_new))
    cost_old = np.where(y == 1, w_bad * (old < t_old), w_good * (old >= t_old))
    diffs = [(cost_new[i] - cost_old[i]).mean()
             for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


# PROVIDED
def load_model(record: dict[str, Any], config: dict[str, Any]) -> Any:
    import mlflow.sklearn

    from src.tracking import setup_mlflow

    setup_mlflow(config)
    return mlflow.sklearn.load_model(record["registered"]["uri"])


# PROVIDED
def compare(config: dict[str, Any], champion: dict[str, Any], challenger: dict[str, Any],
            champ_model: Any, chall_model: Any) -> dict[str, Any]:
    """Both models on both fixed slices, with paired intervals of the differences."""
    rd = retrain_data(config, champion["features"])
    week3, seed = config["week3"], config["seed"]
    costs = week3["final"]["costs"]
    slices = {"validation": (rd.data.X_valid, rd.data.y_valid.to_numpy()),
              "recent": (rd.recent[champion["features"]], rd.recent["TARGET"].to_numpy())}
    t_old, t_new = float(champion["threshold"]), float(challenger["threshold"])
    out: dict[str, Any] = {}
    for name, (X, y) in slices.items():
        old = champ_model.predict_proba(X)[:, 1]
        new = chall_model.predict_proba(X)[:, 1]
        out[name] = {
            "champion": performance(y, old, t_old, costs),
            "challenger": performance(y, new, t_new, costs),
            "auc_diff_ci": list(bootstrap_auc_diff_ci(y, new, old, week3["n_bootstrap"], seed)),
            "cost_diff_ci": list(cost_diff_ci(y, new, t_new, old, t_old, costs,
                                              week3["n_bootstrap"], seed)),
        }
    return out


# PROVIDED
def model_tests_on(record_path: Path, root: Path) -> dict[str, int]:
    """Run tests/model on the model described by ``record_path``; count the outcomes."""
    env = {**os.environ, "MODEL_UNDER_TEST": str(record_path),
           "MLFLOW_DISABLE_AGENT_HINT": "1"}
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/model", "-q", "--tb=no",
                             "-p", "no:cacheprovider"], cwd=root, env=env,
                            capture_output=True, text=True)
    last = (result.stdout.strip().splitlines() or [""])[-1]
    counts = {"passed": 0, "failed": 0, "xfail": 0, "skipped": 0}
    for n, word in re.findall(r"(\d+) (passed|failed|xfailed|skipped|error)", last):
        key = {"xfailed": "xfail", "error": "failed"}.get(word, word)
        counts[key] += int(n)
    return counts


# PROVIDED
def gate_challenger(config: dict[str, Any], comparison: dict[str, Any],
                    tests: dict[str, int]) -> list[str]:
    """Your Week 4 gate on the challenger; the champion's validation AUC is the baseline."""
    thresholds = yaml.safe_load((config["root"] / GATE_CONFIG).read_text()) or {}
    valid = comparison["validation"]
    metrics = {"valid_auc": valid["challenger"]["auc"],
               "cost_at_threshold": valid["challenger"]["cost"], "model_tests": tests}
    return gate_failures(metrics, valid["champion"]["auc"], thresholds)


# PROVIDED
def render(comparison: dict[str, Any], decision: dict[str, Any]) -> str:
    lines = ["# Champion or challenger", "",
             f"Champion version {decision['champion_version']} (threshold "
             f"{decision['champion_threshold']:.3f}), challenger version "
             f"{decision['challenger_version']} (threshold "
             f"{decision['challenger_threshold']:.3f}). Each model at its own threshold. "
             "Intervals: 95% paired bootstrap of challenger minus champion. Written by "
             "python run.py champion.", ""]
    for name in SLICES:
        c = comparison[name]
        a, b = c["champion"], c["challenger"]
        lines += [f"## The {name} slice ({a['n']} applicants, {a['n_default']} defaults)", "",
                  "| | AUC | Cost per applicant | Approval rate | Bad rate among approved |",
                  "|---|---|---|---|---|",
                  f"| champion | {a['auc']:.3f} | {a['cost']:.3f} | {a['approval_rate']:.1%} | "
                  f"{a['bad_rate_approved']:.1%} |",
                  f"| challenger | {b['auc']:.3f} | {b['cost']:.3f} | "
                  f"{b['approval_rate']:.1%} | {b['bad_rate_approved']:.1%} |",
                  f"| difference [95% interval] | {b['auc'] - a['auc']:+.3f} "
                  f"[{c['auc_diff_ci'][0]:+.3f}, {c['auc_diff_ci'][1]:+.3f}] | "
                  f"{b['cost'] - a['cost']:+.3f} [{c['cost_diff_ci'][0]:+.3f}, "
                  f"{c['cost_diff_ci'][1]:+.3f}] | | |", ""]
    gate = decision["gate_failures"]
    lines += ["## Quality gate on the challenger", "",
              "PASSED" if not gate else "FAILED: " + "; ".join(gate),
              f"(model tests: {decision['model_tests']})", "",
              f"## Decision: {decision['decision'].upper()}", ""]
    lines += [f"- {reason}" for reason in decision["reasons"]]
    return "\n".join(lines) + "\n"


# PROVIDED
def main() -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    config = load_retrain_config()
    root = config["root"]
    if not (root / CHALLENGER).exists():
        raise SystemExit(f"{CHALLENGER} not found: python run.py retrain first")
    champion, challenger = champion_record(config), json.loads((root / CHALLENGER).read_text())
    print(f"Champion version {champion['registered']['version']} vs challenger version "
          f"{challenger['registered']['version']} ...")
    comparison = compare(config, champion, challenger, load_model(champion, config),
                         load_model(challenger, config))
    print("Running the model tests on the challenger ...")
    tests = model_tests_on(root / CHALLENGER, root)
    failures = gate_challenger(config, comparison, tests)
    verdict = decide(comparison, failures, config["retrain"]["decision"])
    decision = {
        **verdict, "gate_failures": failures, "model_tests": tests,
        "champion_version": str(champion["registered"]["version"]),
        "challenger_version": str(challenger["registered"]["version"]),
        "champion_threshold": float(champion["threshold"]),
        "challenger_threshold": float(challenger["threshold"]),
        "rule": config["retrain"]["decision"],
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out = root / OUT
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2, default=float) + "\n")
    (out / "decision.json").write_text(json.dumps(decision, indent=2, default=float) + "\n")
    (out / "comparison.md").write_text(render(comparison, decision))
    print((out / "comparison.md").read_text())
    print(f"Wrote {OUT}/comparison.json, comparison.md, decision.json")
    print("Next: write docs/decision_record.md; then python run.py promote, or keep the "
          "champion (nothing to do).")


if __name__ == "__main__":
    main()
