<!-- PROVIDED: copy this file to docs/modeling_report.md and fill in every placeholder
     (the check fails while the word in capitals below is still anywhere in the file).
     1 to 2 pages. Every number comes from reports/final_model.json, reports/threshold.json,
     or the MLflow UI. Quote every AUC with its 95% interval. -->

# Modeling report (Week 3)

**Team:** Group 4

## 1. The production model

The production model is a `HistGradientBoostingClassifier` using `learning_rate=0.01135`, `max_iter=350`, `max_leaf_nodes=4`, `min_samples_leaf=113`, `l2_regularization=0.00245`, and `max_features=0.3124`, with 164 features, isotonic calibration, a decision threshold of 0.08, and registered name/version `models:/credit-risk@staging` (version 1); the exact Optuna trial number is not identified in the supplied files.

## 2. Production model against the decision tree

| | Production model | tree_tuned |
|---|---|---|
| Validation ROC-AUC [95% CI] | 0.747 [0.693, 0.801] | 0.703 [0.645, 0.756] |
| CV ROC-AUC (train split) | 0.7454 ± 0.0213 | 0.6965 ± 0.0095 |
| Latency p95, one applicant (ms) | 16.03 | 3.25 |
| Model size (KB) | 327.57 | 17.69 |
| Features | 164 | 164 |
| Explainability | Lower; ensemble-based predictions are harder to explain directly | Higher; shallow decision rules are easier to inspect |

The production model's paired validation AUC improvement is 0.0445 with a 95% interval of [-0.0131, 0.1051], so it is not demonstrably better than chance in the paired comparison because the interval includes zero, although it has higher observed AUC and CV AUC; this comes at roughly 4.9 times the p95 latency and 18.5 times the model size, while sacrificing the tree's simpler explanations in exchange for better observed predictive performance.

## 3. The decision threshold

The production threshold is 0.08, chosen to balance the specified costs of 10 for approving a defaulter and 1 for rejecting a good applicant, and it is far below 0.5 because missing a defaulter is ten times as costly as rejecting a good applicant; threshold selection should use training out-of-fold scores rather than the validation split to avoid tuning directly to the evaluation set, although `threshold.json` records a threshold of 0.09 for a different, uncalibrated model configuration, so that file's threshold should not be confused with the final production threshold of 0.08. At the production threshold, validation expected cost is 0.533 per applicant, compared with 0.800 at 0.5 and 0.810 when approving everyone, and 25.1% of applicants are rejected.

## 4. Calibration

The model uses isotonic calibration, which slightly improves the validation Brier score from 0.06903 to 0.06872 while aligning mean predicted default risk (0.0810) closely with the observed default rate (0.081), suggesting a modest improvement in probability calibration; the reliability plot itself was not included among the supplied files, and the decision threshold depends on calibration because it is applied to predicted probabilities whose meaning must correspond reasonably well to actual default risk.

## 5. Proposed alerts

The following alerts use the project's PSI thresholds and add checks for changes in predictions and input-data quality.

| Signal | Threshold | Checked | Triggers |
|---|---|---|---|
| Feature drift (PSI) | PSI above 0.10 is moderate; above 0.25 is major | After every 1,000 predictions | Notify the ML team. They investigate the affected features and check for upstream data changes before deciding whether the model needs retraining. |
| Prediction-score drift | Score PSI above 0.10 | After every 1,000 predictions | Notify the ML team. They compare score distributions with the training reference and investigate changes in predicted risk. |
| Approval-rate change | Approval rate changes by more than 5 percentage points from the 74.9% validation baseline | Daily | Notify the model owner and lending operations team. They review recent applicant patterns, scores, and decision thresholds. |
| Missing-value increase | Missingness for any important feature increases by more than 10 percentage points from the training reference | After every 1,000 predictions | Notify the data and ML teams. They check data pipelines and confirm that missing values are being handled correctly before continuing normal monitoring. |

## 6. Limitations

This monitoring identifies changes in input data, prediction scores, and approval rates, but it cannot determine whether individual lending decisions were correct. Actual default outcomes are needed to measure model accuracy and determine whether the model is making reliable decisions. These outcomes may take several months or longer to become available because repayment behavior must be observed over time.

With more time, I would add delayed performance monitoring using actual repayment and default records, regular fairness checks across demographic groups, and automated alerts for unusual changes. I would also expand monitoring to cover more prediction batches and longer periods so that temporary changes can be distinguished from lasting data or model problems.