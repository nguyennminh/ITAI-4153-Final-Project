# Week 4: Testing and CI (Wednesday)

<!-- PROVIDED -->

On Monday you wrote tests. Today they start running **by themselves**, on a Google Cloud
server, every time anyone opens a pull request, and GitHub refuses to merge the pull request
unless they pass. That is **continuous integration (CI)**.

```
you push a branch → open a pull request → Cloud Build runs cloudbuild.yaml:
    install → lint → test → quality gate
         all green → GitHub allows the merge
         any red   → GitHub blocks the merge (branch protection)
```

## Why a server, and not "we'll run the tests"

On a team, everyone means to run the tests. Someone forgets, on a Friday, with a "tiny
change". A CI server never forgets, runs the same steps every time on a clean machine, and
shows the result on the pull request for everyone to see. Branch protection makes it a rule,
not a habit.

## What CI can and cannot see

Cloud Build gets a fresh copy of your **GitHub repository**: no `data/`, no `mlflow.db`. So:

- **Lint** and the **unit tests** run normally.
- The **data and model tests skip themselves** ("no data on this machine"). They run on your
  laptops (`python run.py testreport`), and their results are committed in
  `reports/test_report.json`.
- The model cannot be trained or scored on the server. So you measure it on your laptop with
  `python run.py evaluate`, which writes `reports/gate_metrics.json`, and you commit that file.

## The quality gate

`scripts/quality_gate.py` reads three committed files and fails the build if the model this
pull request would ship is not good enough:

| Rule (config/gate.yaml) | Compares | Catches |
|---|---|---|
| `min_auc` | validation AUC vs a floor | a broken model |
| `max_cost` | expected cost per applicant at your threshold vs a limit | a model that no longer saves money |
| `max_auc_drop` | validation AUC vs the **registered** model's | a change that makes the model worse, even if it is still "good enough" |
| `allow_failing_model_tests` | model tests that fail (xfail does not count) | a new behavior problem |

**Stale metrics.** What stops a teammate from changing the code and committing the old
`gate_metrics.json`? `evaluate` stores a **fingerprint** (a hash) of the code and config in
the file, and the gate recomputes it from the pull request's code. If they differ, the gate
fails: "stale metrics". So every change to the code must come with metrics measured on that
code.

## The workflow from today on

```
git checkout -b my-change        # never work on main directly
... edit code ...
python run.py evaluate           # measure the model this code would ship
python run.py ci                 # the same steps as Cloud Build, on your laptop
git add . && git commit -m "..." && git push -u origin my-change
                                 # open a pull request on GitHub; wait for the check
```

## Commands

```
python run.py evaluate          # -> reports/gate_metrics.json (about 5 seconds)
python run.py gate              # the quality gate, as Cloud Build runs it
python run.py ci                # lint, then all tests, then the gate: like Cloud Build
python run.py check --stage wed # the Wednesday checkpoint
```

## The files today

| File | Status | What it is |
|---|---|---|
| `cloudbuild.yaml` | install step PROVIDED; **3 steps TODO** | what Cloud Build runs |
| `scripts/quality_gate.py` | MIXED | **`gate_failures`** (yours); reading the files, the fingerprint check |
| `config/gate.yaml` | **TODO** | your thresholds (all `null` until you set them) |
| `src/evaluate.py` | PROVIDED | `python run.py evaluate` |
| `docs/gcp_setup.md` | PROVIDED | Google Cloud and GitHub setup, one page |
| `docs/ci_report_template.md` | PROVIDED | copy to `docs/ci_report.md` |
| `scripts/check_week4.py` | DO NOT EDIT (replaced) | adds `--stage wed` |
| `tests/test_week4_wed.py` | DO NOT EDIT | 11 tests of the gate (3 pass now) |

## Rules from today

- Work on a branch and merge through pull requests; `main` is protected.
- Never commit a key, a token, or a service-account file. Cloud Build needs none.
- Never loosen a gate threshold to get a pull request through. If a threshold is wrong,
  change it in its own pull request, with the reason in `docs/ci_report.md`.
