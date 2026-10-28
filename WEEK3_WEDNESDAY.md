# Week 3: Modeling and experiment tracking (Wednesday)

<!-- PROVIDED -->

On Monday you compared models one setting at a time. Today you search the settings together,
turn scores into **decisions**, and, by Sunday, register the one model that Weeks 4-7 will
test, deploy, and monitor:

```
python run.py tune        Optuna: 15 trials per run, each a child run in MLflow
python run.py threshold   the threshold that costs the lender least
python run.py final       calibrate, check, compare with the tree, and register
```

## Tuning with Optuna

Monday's best variations each changed one setting, and none of them helped by more than the
noise; combining several did. With 6 settings to tune, a grid of 5 values each is 15,625
models. **Optuna** instead proposes one set of settings per **trial**, scores it, and uses
every trial so far to propose the next one near the good ones (its TPE sampler). About 30
trials find settings as good as a huge grid.

- Your `objective` scores a trial by its **mean CV AUC on the train split**, as on Monday.
  Never by validation AUC: 30 trials picked on 81 validation defaults fit the validation set.
- Each `python run.py tune` adds 15 trials to the **same saved study** (`optuna.db`), so run
  it once in class and once at home: the second run continues where the first stopped.
- In MLflow, each `tune` is one parent run (`optuna_hgb-tuning`) with a child run per trial.
- `reports/optuna_best.json` holds the best trial so far. `config/final.yaml` uses it
  (`params: best_trial`) for everything after tuning.

## From a score to a decision: the threshold

The service must answer approve or reject: **reject when score ≥ threshold**. Your Week 1
costs:

| | Actually repays | Actually defaults |
|---|---|---|
| **Approve** | 0 | **10** (lost principal) |
| **Reject** | **1** (lost interest) | 0 |

`expected_cost` averages these over applicants for one threshold; `cost_curve` does it for
every threshold from 0.01 to 0.60; `choose_threshold` picks the cheapest. A missed default
costs 10 times a lost customer, so the cheapest threshold is far below 0.5. If the scores
are true probabilities, theory says reject when p ≥ 1 / (1 + 10) ≈ 0.09.

**Choose the threshold on out-of-fold scores of the train split** (the `oof` array your
`cross_validate_auc` already returns), never on the validation split. Choosing it there
fits it to those 1,000 applicants, and the validation cost then looks better than it will
be on real applicants.

## Calibration

A model can rank applicants well and still give probabilities that are too high or too
low. Undersampling, for example, trains on a 25% default rate, so its scores come out about
twice too high. The threshold above assumes the scores are real probabilities, so
`python run.py final` wraps the model in **calibration**: a small monotone map from score to
probability, learned on out-of-fold scores (`isotonic` = a flexible step function,
`sigmoid` = Platt scaling, a smooth curve). The reliability plot shows the result: a
calibrated model's points lie on the diagonal. Calibration barely changes the AUC; it changes
what the scores mean. Report whether it helped **your** model, honestly.

## Registering the model

`python run.py final` logs the run `final_model` and **registers** it in the MLflow Model
Registry as `credit-risk`, with the alias `staging` pointing at the new version, and the
threshold stored on the version. Week 5 loads `models:/credit-risk@staging`; shipping a
better model later just means registering a new version and moving the alias.

## The files today

| File | Status | What it is |
|---|---|---|
| `config/final.yaml` | PROVIDED | costs, threshold grid, the model, Optuna, calibration, registry |
| `src/tune.py` | MIXED | **`suggest_params`**, **`objective`** (yours); the study runner (provided) |
| `src/cost.py` | MIXED | **`expected_cost`**, **`cost_curve`**, **`choose_threshold`** (yours); the `threshold` command |
| `src/calibration.py` | MIXED | **`calibrate`** (yours); reliability table, Brier score |
| `src/final.py` | MIXED | **`register_model`** (yours); the whole `final` command |
| `src/plots.py` | PROVIDED | cost curve, reliability, AUC-versus-features plots |
| `src/experiment.py` | PROVIDED (replaced) | now also reads final.yaml and names the report files |
| `src/tracking.py` | PROVIDED (replaced) | also hides MLflow's warnings when registering a model |
| `scripts/check_week3.py` | DO NOT EDIT (replaced) | `--stage wed` for today, no flag for the submission |
| `tests/test_week3_wed.py` | DO NOT EDIT | 13 tests, one or more for each of your functions |
| `docs/modeling_report_template.md` | PROVIDED | copy to `docs/modeling_report.md` |

## Files the commands write (commit all of them)

| File | Written by |
|---|---|
| `reports/optuna_best.json` | `tune` |
| `reports/threshold.json`, `reports/figures/cost_curve.png` | `threshold` |
| `reports/final_model.json`, `reports/figures/reliability.png` | `final` |
| `reports/mlflow_runs.csv` | `runs` |

## Rules today

- Tune and choose the threshold on the **train split** only. The validation split is scored
  once, by `python run.py final`.
- The final model trains on the real default rate: no `undersample` in final.yaml's model.
- Do not edit `scripts/check_week3.py` or the tests. Do not commit `mlflow.db`, `optuna.db`,
  `mlruns/`, or `data/`.
