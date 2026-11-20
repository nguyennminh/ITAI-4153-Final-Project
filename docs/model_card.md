<!-- PROVIDED: copy this file to docs/model_card.md and fill in every placeholder.
     A draft is due this week; Week 7 finishes it. Keep it short and factual: a reader who
     has never seen the project should understand what the model is for and when not to
     trust it. -->

# Model card: credit-risk (draft)

**Team:** Group 4

## Model details

The registered model is `credit-risk`, version `1`, under the `staging` alias. It uses tuned Histogram Gradient Boosting with isotonic calibration and 164 features. The decision threshold is **0.08 (8%)**, based on a 10:1 cost ratio between approving a defaulter and rejecting someone who would repay. The model is served through FastAPI on Google Cloud Run as `credit-risk-api`. The project is maintained by Group 4.

## Intended use

The model estimates an applicant's probability of default through the `/predict` API. It is intended to support credit-risk assessment, not make lending decisions independently. It should not be used for unrelated decisions or without appropriate human review.

## Data

The model uses 5,000 applicants from the Kaggle Home Credit Default Risk dataset. The fixed split contains 4,000 training and 1,000 validation applicants (seed 42). Each row represents one applicant. The target is `TARGET`, indicating payment difficulties. Known data issues include the `DAYS_EMPLOYED` sentinel value and missing `EXT_SOURCE` values. The Kaggle data is subject to competition-use restrictions. See [`docs/data_card.md`](data_card.md).

## Performance

The final model achieved a validation ROC-AUC of **0.7471** on 1,000 applicants, including 81 defaults. Expected cost was 0.533 per applicant, and the approval rate was 74.9%. Isotonic calibration improved the Brier score from 0.0690 to 0.0687. The tuned decision-tree baseline achieved an AUC of 0.7026, while logistic regression achieved 0.7097. Results are promising but limited by the small validation sample.

## Explanations

SHAP analysis identified `EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`, `DAYS_EMPLOYED`, and `PREV_REFUSED_RATE` as important features. Individual rejection explanations are available in [`reports/explain/explanations.md`](../reports/explain/explanations.md), with global importance shown in [`reports/explain/shap_global.png`](../reports/explain/shap_global.png).

## Fairness

Female applicants had a 79.4% approval rate compared with 65.9% for males. Approval rates ranged from 58.7% for ages 18–29 to 91.9% for ages 60+. Smaller age-group default counts make some estimates unreliable. Gender was directly used by the model, and other features predicted gender with an AUC of 0.8634. These findings raise fairness concerns. See [`docs/fairness_analysis.md`](fairness_analysis.md).

## Monitoring

The API logs applicant features, model version, prediction score, decision, and timestamp to Google Cloud Logging. Batch 1 showed little drift. Batch 2 showed major drift in income, external-score missingness, and credit ratios. PSI alerts are set at 0.10 for moderate drift and 0.25 for major drift. Major changes should trigger investigation before relying on predictions. See [`docs/monitoring_report.md`](monitoring_report.md).

## Limitations

The model was trained on a small sample and may not work reliably on applicants unlike those in its training data. Small demographic groups make fairness results uncertain. Upstream data changes can reduce reliability. Live accuracy is also unknown until actual repayment outcomes become available.

## Ethical considerations

Incorrect rejections can prevent qualified applicants from receiving credit, while incorrect approvals can create financial losses. Gender sensitivity and demographic approval gaps are additional concerns. Human review, explanations, an appeal process, fairness checks, and ongoing monitoring should be in place before real lending use. The repository does not confirm that all these protections are implemented.

**Overall:** This is a promising academic prototype, but its fairness concerns and observed data drift mean it should not independently determine real lending decisions.