# Week 4: Testing and CI (Monday)

<!-- PROVIDED -->

Last week you registered a model (`models:/credit-risk@staging`). Next week it goes live
behind an API. Before anyone depends on it, you protect it with **tests**: small programs
that check, every time anyone changes anything, that the code, the data, and the model
still behave the way the lender needs. On Wednesday those tests start running automatically
on every pull request.

## Three kinds of tests

| Kind | Folder | Checks | Needs the data? | Runs in CI (Wednesday)? |
|---|---|---|---|---|
| **Unit** | `tests/unit/` | one function, on a tiny hand-made table | no | yes |
| **Data** | `tests/data/` | the real feature table: schema, value ranges, missing values | yes | skipped |
| **Model** | `tests/model/` | the registered model's behavior, through `predict_proba` | yes | skipped |

Data and model tests need the Kaggle data, which never goes to GitHub, so they are skipped
where it is missing (a teammate's fresh clone, or the CI server). That is why the unit tests
matter so much: they are the tests that run everywhere.

### Unit tests: properties, not numbers

Week 2's tests checked that each function gives the right numbers on one small table. Today
you test **properties** that must hold for any data: one applicant's history never changes
another's features; shuffling the rows changes nothing; nobody disappears in a join; no
ratio is ever infinite. A property test catches bugs nobody thought to write a number for.

### Data tests: the assumptions the model makes

The model was trained on data that looked a certain way. If a new sample, a changed
aggregation, or a teammate's "small fix" changes that, the model still runs, and quietly
gets worse. Choose each limit by **looking at the data first**, and write down why:
a limit that is too loose catches nothing; one that is too tight fails on good data.

### Model tests: behavior the lender needs

A model can have a good AUC and still behave in ways the lender cannot accept. Three kinds:

- **Minimum quality** (provided): the AUC on the validation split is above a floor.
- **Invariance**: changing something that must not matter (the weekday someone applied on,
  or their gender) must not change the score.
- **Directional**: changing something in a known direction (more income, everything else
  equal) must not move the risk the wrong way.

Every model test calls the model only through `score(model, X)`, which is `predict_proba`.
The tests never look inside the model, so they keep working whatever model you register.

## When a test fails

**Do not change the test to make it pass.** A failing model test is a finding about your
model. Find out why (which feature, how many applicants, by how much), and write it in
`docs/test_findings.md`. Then decide with your team:

- **fix it**: change the model or the data, retrain, and the test passes; or
- **accept it for now**: keep the test and mark it as a known failure,
  `@pytest.mark.xfail(strict=True, reason="see docs/test_findings.md")`, with the reason
  and what would make you fix it. `strict=True` means the test turns red again if it ever
  starts passing, so the mark cannot be forgotten.

## Commands

```
python run.py test tests/unit       # only the unit tests (fast, no data needed)
python run.py test tests/data       # only the data tests
python run.py test tests/model      # only the model tests (about 5 seconds)
python run.py test                  # everything, Weeks 1-4
python run.py testreport            # all three kinds, written to reports/test_report.md
python run.py check --stage mon     # the Monday checkpoint
```

Add `-k name` to run one test, for example
`python run.py test tests/model -k income`.

## The files this week

| File | Status | What it is |
|---|---|---|
| `src/model_io.py` | PROVIDED | loads the model under test; `score`; `change_income` |
| `tests/unit/test_features_unit.py` | 1 example + 4 TODO tests | properties of your Week 2 functions |
| `tests/data/test_data_quality.py` | 1 example + 4 TODO tests | your real feature table |
| `tests/model/test_model_behavior.py` | 1 example + 3 TODO tests | the registered model |
| `tests/data/conftest.py`, `tests/model/conftest.py` | PROVIDED | the `features` and `production` fixtures (and the skip when there is no data) |
| `scripts/test_report.py` | PROVIDED | `python run.py testreport` |
| `scripts/check_week4.py` | DO NOT EDIT | the checkpoint checks |
| `docs/test_findings_template.md` | PROVIDED | copy to `docs/test_findings.md` |

In this week's TODOs **you write the tests**: each TODO is a test function whose docstring
says what it must check. Replace its `pytest.fail("TODO(student) ...")` line with the test.
Keep the test's name: the check script and the report use it.

## Rules this week

- Never loosen a test to make it pass, and never delete one. Document instead.
- Call models only through `score` (`predict_proba`).
- Do not edit `scripts/check_week4.py` or the earlier weeks' tests.
- Do not commit `data/`, `mlflow.db`, or `mlruns/`.
