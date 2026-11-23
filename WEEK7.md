# Week 7: Retrain and final demo (Monday)

<!-- PROVIDED -->

Last week you watched your live model and found that batch 2 looked different. You could
not tell whether its decisions were still right: a default shows up months after the loan.
Today the outcomes arrive. You measure what really happened, train a challenger, compare it
with the champion, run your own quality gate, and decide: **promote or keep**. On Wednesday
you automate the whole loop on Google Cloud and demo it.

```
live log (Week 6) + true outcomes ──▶ outcomes: how did the champion REALLY do?
                                                │
train + labeled production ──▶ retrain ──▶ challenger (MLflow, alias challenger)
                                                │
       validation + recent slice ──▶ champion: compare, YOUR gate, YOUR decide
                                                │
                                   promote ──or── keep      (rollback = the same move back)
```

## 1. The truth arrives

`python run.py outcomes` joins the predictions your **live** service logged in Week 6
(`data/logs/predictions.json`) with the true outcomes of batches 1 and 2
(`data/production/labels.parquet`: you may open it now). For each batch, and for the
validation set: AUC, expected cost per applicant, approval rate, the default rate against the
mean score, and the **bad rate among approved** applicants: the loans that go wrong. You
write `performance`. Then compare with what the monitoring said last week.

## 2. The challenger

`python run.py retrain` trains a new model with **exactly the champion's recipe** (model,
settings, calibration, threshold rule) on more data: the 4,000 training applicants plus the
labeled production applicants. A fixed 30% of those are kept aside: the **recent slice**.
The challenger is logged to MLflow and registered as a new version with the alias
`challenger`. It does not serve anything yet.

## 3. Champion or challenger?

`python run.py champion` scores both models on two fixed slices that neither trained on:

| Slice | What it stands for |
|---|---|
| validation (1,000) | the applicants every week has used; your gate's slice |
| recent (about 600) | applicants like the ones the model will see next |

Each model uses its own threshold. Every difference comes with a paired 95% interval. Then:

- the **model tests** (Week 4) run on the challenger;
- **your quality gate** (`gate_failures`, config/gate.yaml) judges the challenger, with the
  champion as the model it may not fall behind;
- **your** `decide` turns all of it into "promote" or "keep", with the reasons.

It writes `reports/retrain/comparison.md` and `decision.json`, and changes nothing.

## 4. Promote, keep, roll back

| You decide | Laptop | Cloud (docs/cloud_shell_week7.md) |
|---|---|---|
| promote | `python run.py promote` (alias + export) | upload `models/` to `models/v2/`, point the service at it |
| keep | nothing | nothing (but practise the rollback) |
| roll back | `python run.py rollback --version 1` | point the service at `models/v1/` |

The champion is always the alias `staging` (since Week 3) and its record
`reports/champion.json`. `promote` refuses when your decision says "keep", unless you pass
`--override "reason"`: overruling your own rule is allowed, but it is written down.

From today every version lives in its own bucket folder (`models/v1/`, `models/v2/`) and the
service loads the folder named in `MODEL_BUCKET`. Switching versions is one command and needs
no new image.

## Commands

```
python run.py test tests/test_week7_mon.py   # 8 tests (no data needed)
python run.py outcomes                       # reports/retrain/outcomes.md
python run.py retrain                        # the challenger, registered in MLflow
python run.py champion                       # compare, gate, decide
python run.py promote                        # only if you decide to (or rollback --version 1)
python run.py evaluate                       # the zip added code under src/
python run.py check --stage mon              # the Monday checkpoint
```

## The files today

| File | Status | What it is |
|---|---|---|
| `src/outcomes.py` | MIXED | **`performance`** (yours); joining the log with the outcomes (provided) |
| `src/retrain.py` | PROVIDED | `python run.py retrain` |
| `src/champion.py` | MIXED | **`decide`** (yours); the comparison, model tests, gate (provided) |
| `src/promote.py` | PROVIDED | `promote` and `rollback` |
| `config/retrain.yaml` | PROVIDED | labeled batches, the recent slice, your decision rule's tolerance |
| `src/export_model.py`, `src/model_io.py` | PROVIDED (replaced) | export and the model tests follow the current champion |
| `tests/test_week7_mon.py` | DO NOT EDIT | 8 tests |
| `scripts/check_week7.py` | DO NOT EDIT | the checkpoint checks |
| `docs/decision_record_template.md` | PROVIDED | copy to `docs/decision_record.md` |
| `docs/cloud_shell_week7.md` | PROVIDED | version folders, promote and roll back in the cloud |
| `run.py` | PROVIDED (replaced) | new commands |

## Rules this week

- Work on a branch, merge through pull requests; `python run.py evaluate` before you commit.
- Never commit `data/`, `models/`, `mlflow.db`, or any key. The bucket stays private.
- Do not edit `scripts/check_week7.py` or the tests.
