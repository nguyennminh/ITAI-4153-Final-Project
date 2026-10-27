# Experiment notes (Week 3)

Team: Krish Iyer, Minh Nguyen, Calvin Zheng

## Setup

Every run below trained on the same rows: the train split of our feature table (4,000
applicants, 323 defaults), compared with 5-fold stratified cross-validation, data_version
9fb9955ad0ed (from `python run.py check`). All runs were trained from commit be10cb1, so none
is tagged `-dirty`. Exported table: `reports/mlflow_runs.csv`.

## Results

One row per run, from `python run.py train --all`. The default HistGradientBoosting settings are
learning_rate 0.1, max_leaf_nodes 31, min_samples_leaf 20, l2_regularization 0, and no early
stopping on 4,000 rows.

| Run | What changed from hgb_default | CV AUC ± std | Train AUC | Valid AUC [95% CI] |
|---|---|---|---|---|
| baseline_lr | (logistic regression, Week 1 pipeline) | 0.723 ± 0.032 | 0.845 | 0.727 [0.666, 0.783] |
| tree_unlimited | (decision tree, no limits) | 0.546 ± 0.016 | 1.000 | 0.562 [0.518, 0.610] |
| tree_tuned | (decision tree, best of the grid) | 0.697 ± 0.010 | 0.782 | 0.703 [0.645, 0.756] |
| hgb_default | nothing | 0.725 ± 0.039 | 1.000 | 0.697 [0.636, 0.760] |
| hgb_balanced | class_weight balanced (each default weighs about 11 times more) | 0.713 ± 0.045 | 1.000 | 0.726 [0.666, 0.783] |
| hgb_undersample | 3 non-defaulters per defaulter in the training rows | 0.722 ± 0.025 | 0.999 | 0.755 [0.697, 0.809] |
| hgb_l2 | l2_regularization 1.0 | 0.732 ± 0.033 | 1.000 | 0.745 [0.697, 0.796] |
| hgb_lr005 | learning_rate 0.05 | 0.725 ± 0.028 | 1.000 | 0.719 [0.662, 0.775] |
| hgb_leaves15 | max_leaf_nodes 15 | 0.719 ± 0.032 | 1.000 | 0.738 [0.680, 0.791] |
| hgb_minleaf100 | min_samples_leaf 100 | 0.722 ± 0.024 | 1.000 | 0.756 [0.705, 0.807] |
| hgb_early_stop | early_stopping on (10% of the train rows held out) | 0.713 ± 0.022 | 0.980 | 0.751 [0.703, 0.799] |

## Overfitting

tree_unlimited scores a perfect 1.000 on the rows it trained on but only 0.546 in
cross-validation, barely better than a coin flip. It grew until every leaf held a handful of
applicants, so it memorized the training rows instead of learning patterns that carry over.
The grid search picked max_depth 5 and min_samples_leaf 50 for tree_tuned, which raises CV AUC to
0.697 and brings train AUC down to 0.782, a much smaller gap.

## What helped and what did not

No single change to the boosting model beat the noise. The best variation, hgb_l2, scores 0.732,
only 0.007 above hgb_default's 0.725, while the fold-to-fold standard deviation is about 0.03.
Class weights and undersampling did not help ranking (0.713 and 0.722). This is expected, because
they change how much each default counts in training, not how well the model orders applicants.
Smaller trees, slower learning and bigger leaves all landed within 0.006 of the default. Only
early stopping and undersampling moved train AUC below 1.000, so every other boosting run still
memorizes the training rows.

The only clear difference is between model types. Every other run beats tree_unlimited by more
than 0.15, far beyond the noise. Against tree_tuned (0.697) the gap is 0.016 to 0.035, and only the
best runs, hgb_l2 (+0.035) and baseline_lr (+0.026), come close to one standard deviation. Boosting
does not clearly beat baseline_lr (0.732 against 0.723).

Validation AUCs spread from 0.697 to 0.756 across boosting runs whose CV AUCs are nearly equal,
which shows why we choose by CV and not by the 81-default validation split. Next we would tune
several settings together, such as a lower learning rate with more trees plus regularization,
instead of changing one at a time.
