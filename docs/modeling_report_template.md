<!-- PROVIDED: copy this file to docs/modeling_report.md and fill in every placeholder
     (the check fails while the word in capitals below is still anywhere in the file).
     1 to 2 pages. Every number comes from reports/final_model.json, reports/threshold.json,
     or the MLflow UI. Quote every AUC with its 95% interval. -->

# Modeling report (Week 3)

Team: TODO

## 1. The production model

TODO: Which model, which settings (from which Optuna trial), how many features, which
calibration, which threshold, and its registered name and version
(for example `models:/credit-risk@staging`, version 3).

## 2. Production model against the decision tree

| | Production model | tree_tuned |
|---|---|---|
| Validation ROC-AUC [95% CI] | TODO | TODO |
| CV ROC-AUC (train split) | TODO | TODO |
| Latency p95, one applicant (ms) | TODO | TODO |
| Model size (KB) | TODO | TODO |
| Features | TODO | TODO |
| Explainability | TODO | TODO |

TODO: The AUC difference and its paired 95% interval (from `python run.py final`). Is the
production model better by more than chance? Then justify the choice on all four
dimensions: AUC, speed, size, and explainability. What do you give up compared with the
tree, and why is it worth it (or not)?

## 3. The decision threshold

TODO: The threshold, the costs it comes from, and why it is so far from 0.5. Which scores
was it chosen on, and why not the validation split? Expected cost per applicant on the
validation split at your threshold, at 0.5, and when approving everyone. What share of
applicants does it reject?

## 4. Calibration

TODO: Method, Brier score before and after, and what the reliability plot
(`reports/figures/reliability.png`) shows. Did calibration help this model? Why does the
threshold depend on it?

## 5. Limitations and risks

TODO: At least three, for example: the size of the validation set (81 defaults), choices
made on the same train split, what the costs leave out, and sensitive features (Week 1).
