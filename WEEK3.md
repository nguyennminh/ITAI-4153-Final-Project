# Week 3: Modeling and experiment tracking (Monday)

<!-- PROVIDED -->

Last week you built one feature table: 5,000 applicants, 164 features, in
`data/features/features.parquet`. This week you train real models on it and keep a record
of every one. Today: a decision tree, a boosting model, and variations of it, all tracked in
**MLflow** and compared with **cross-validation**.

## Why track experiments?

By Friday you will have trained more than 50 models. A month from now someone will ask
"why did you ship this one, and can you rebuild it?" You will not remember which settings,
which version of the features, or which commit produced which number. A spreadsheet of
copied numbers goes stale the moment someone reruns something.

**MLflow** records each training run automatically, as it happens:

| What | Example | Why |
|---|---|---|
| params | `model=hist_gradient_boosting`, `learning_rate=0.05` | what was trained |
| metrics | `cv_auc_mean=0.739`, `train_auc=0.957`, `latency_p95_ms=6.1` | how well, how fast, how big |
| tags | `git_commit=3f2a9c1`, `data_version=b3ee6c96b4fc` | which code and which data |
| the model | a saved copy | load or register it later |

All runs go into one file, `mlflow.db`, in your repository root. `python run.py ui` opens a
web page where you can sort, filter, and compare them. The database stays on your computer
(it is in `.gitignore`); `python run.py runs` exports the table to `reports/mlflow_runs.csv`,
which you commit.

**Commit before the runs you want to keep.** A run made with uncommitted changes is tagged
`-dirty`: nobody, including you, can get that exact code back.

## Why cross-validation and not the validation set?

The validation set has 1,000 applicants but only **81 defaults**. Its AUC moves by about
**±0.03** just from which applicants happened to land in it, which is as large as the
differences between the models you compare today. If you pick the best of 10 models by
validation AUC, you partly pick the luckiest one.

**Stratified 5-fold cross-validation** on the 4,000 training applicants (323 defaults):

```
train split:  [ fold 1 | fold 2 | fold 3 | fold 4 | fold 5 ]
round 1:        score    fit      fit      fit      fit
round 2:        fit      score    fit      fit      fit
...             (5 rounds: every applicant is scored once, by a model that never saw them)
```

"Stratified" means every fold gets the same share of defaults (about 8%). You get 5 AUCs:
their **mean** is the number you compare runs by, and their **standard deviation** tells you
how much of a difference is noise. The validation split is still scored for every run, with
a 95% interval, but you do not choose by it.

## Overfitting, and the decision tree

A decision tree asks yes/no questions ("EXT_SOURCE_2 < 0.4?") until each leaf is pure. With
no limits it keeps asking until it has memorized every training applicant: **train AUC
1.000**. On applicants it has never seen it is barely better than a coin flip. Two settings
stop it early:

- `max_depth`: at most this many questions in a row
- `min_samples_leaf`: every leaf must hold at least this many applicants

`python run.py train tree_tuned` tries 30 combinations and keeps the one with the best CV
AUC. Compare its train AUC with its CV AUC: the gap is the overfitting.

## Boosting

Gradient boosting builds hundreds of **small** trees one after another; each new tree fixes
part of the mistakes the trees so far still make. It is the standard model for tabular data
like ours. We use scikit-learn's `HistGradientBoostingClassifier` (no extra installation,
and it handles missing values itself). LightGBM or XGBoost are fine too if your team prefers
(see `make_model`). The settings that matter most:

| Setting | Default | What it does |
|---|---|---|
| `learning_rate` | 0.1 | how much each tree corrects; smaller needs more trees |
| `max_iter` | 100 | number of trees |
| `max_leaf_nodes` | 31 | size of each tree |
| `min_samples_leaf` | 20 | smallest group a leaf may describe |
| `l2_regularization` | 0 | shrinks leaf values toward 0 |
| `class_weight` | none | `balanced` weights each default about 11 times more |

## Only 8% default: class weights and undersampling

Two common responses to a rare class, both in `config/models.yaml`:

- **class weights**: every default counts more in training (`class_weight: balanced`).
- **undersampling**: train on all defaults and only some of the others
  (`undersample: 3` keeps 3 non-defaulters per defaulter).

Undersampling must only ever touch **training rows**. If you undersample before splitting,
the held-out fold has a fake 25% default rate and its AUC means nothing. `fit_model`
undersamples inside each training fold, so your `cross_validate_auc` gets this right by
calling it. Neither trick is guaranteed to help AUC; measure it.

## Commands

```
python run.py train                  # list the runs in config/models.yaml
python run.py train baseline_lr      # train one run (or several names)
python run.py train tree_tuned       # the 30-tree grid search (your tune_tree)
python run.py train --all            # everything in models.yaml, then tree_tuned
python run.py ui                     # MLflow at http://127.0.0.1:5050 (Ctrl+C to stop)
python run.py runs                   # export every run to reports/mlflow_runs.csv
python run.py check --stage mon      # the Monday checkpoint
```

`python run.py train` prints one line per run:

```
hgb_regularized   CV AUC 0.739 ± 0.020   train 0.957   valid 0.757 [0.704, 0.809]   0.4s fit
```

## Using the MLflow UI

1. `python run.py ui`, wait a few seconds, then open http://127.0.0.1:5050. If it opens on
   the **GenAI** pages (Traces, Prompts, an Assistant panel), click **Model training** at the
   top left. Open the experiment `credit-risk-week3`, then **Training runs** in the left menu.
2. With the **Columns** button above the table, add the metrics `cv_auc_mean`, `cv_auc_std`,
   and `train_auc`, and sort by `cv_auc_mean`.
3. `tree_tuning` has a ▸: click it to see its 30 child runs.
4. Tick several runs, then **Compare**: their params and metrics side by side.
5. Leave the terminal running while you use it; stop it with Ctrl+C.

## The files this week

| File | Status | What it is |
|---|---|---|
| `config/models.yaml` | PROVIDED, you add runs | MLflow settings, CV folds, the tree grid, the named runs |
| `src/experiment.py` | PROVIDED | Week 3 config, and the fixed training data (`load_training_data`) |
| `src/tracking.py` | PROVIDED | MLflow setup, tags, the run table export |
| `src/models.py` | MIXED | `tree_preprocessor` (provided), **`make_model`**, **`undersample`** (yours), `fit_model` |
| `src/evaluation.py` | MIXED | **`cross_validate_auc`** (yours), validation metrics, speed, size |
| `src/train.py` | MIXED | `train_and_log` (the worked example), **`tune_tree`** (yours) |
| `scripts/check_week3.py` | DO NOT EDIT | the checkpoint checks |
| `tests/test_week3_mon.py`, `tests/week3_fixtures.py` | DO NOT EDIT | 19 tests (5 test provided code and pass now) |
| `docs/experiment_notes_template.md` | PROVIDED | copy to `docs/experiment_notes.md` (carry-over) |

## Rules this week

- Compare runs by **CV AUC on the train split**. Never choose a model, a setting, or a
  threshold by its validation AUC.
- Every run goes through `python run.py train`, so it is logged with its code and data
  version. A number that is not in MLflow does not count.
- Change one thing per variation, so you know what made the difference.
- Do not edit `scripts/check_week3.py`, `tests/test_week3_mon.py`, or `tests/week3_fixtures.py`.
- Do not commit `data/`, `mlflow.db`, or `mlruns/` (`.gitignore` already blocks them).
