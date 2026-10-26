<!-- PROVIDED: copy this file to docs/experiment_notes.md and replace every TODO. -->

# Experiment notes (Week 3)

Team: TODO

## Setup

Every run below trained on the same rows: the train split of our feature table (4,000
applicants, 323 defaults), compared with 5-fold stratified cross-validation, data_version
TODO (from `python run.py check`). Exported table: `reports/mlflow_runs.csv`.

## Results

One row per run. Copy the numbers from `python run.py train` or the MLflow UI. Every
boosting variation must appear here **by its run name**.

| Run | What changed from hgb_default | CV AUC ± std | Train AUC | Valid AUC [95% CI] |
|---|---|---|---|---|
| baseline_lr | (logistic regression, Week 1 pipeline) | TODO | TODO | TODO |
| tree_unlimited | (decision tree, no limits) | TODO | TODO | TODO |
| tree_tuned | (decision tree, best of the grid) | TODO | TODO | TODO |
| hgb_default | nothing | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO |

## Overfitting

TODO: In 2-3 sentences: what do tree_unlimited's train AUC and CV AUC show, and which
max_depth and min_samples_leaf did tree_tuned pick?

## What helped and what did not

TODO: In 3-6 sentences: which changes helped, which did not, and which differences are
bigger than the CV standard deviation (the rest may be noise)? What would you try next?
