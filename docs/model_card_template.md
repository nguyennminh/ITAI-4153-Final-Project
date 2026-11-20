<!-- PROVIDED: copy this file to docs/model_card.md and fill in every placeholder.
     A draft is due this week; Week 7 finishes it. Keep it short and factual: a reader who
     has never seen the project should understand what the model is for and when not to
     trust it. -->

# Model card: credit-risk (draft)

Team: TODO · Last updated: TODO

## Model details

TODO: Registered name and version (models:/credit-risk@staging), the algorithm (tuned
gradient boosting with calibration), the number of features, the decision threshold and how
it was chosen (the 10:1 cost ratio), the live service URL, and who owns it.

## Intended use

TODO: What the model is for (who calls it, what decision it supports), and what it must NOT
be used for. Is it meant to decide alone, or to support a person?

## Data

TODO: Training data (the 5,000-applicant sample, the fixed split, the Kaggle source and its
licence), what one row is, the target, and known quirks (DAYS_EMPLOYED sentinel, missing
EXT_SOURCE values). Link docs/data_card.md.

## Performance

TODO: Validation AUC, expected cost per applicant at the threshold, approval rate, and
calibration. Compared with the baseline. Include the uncertainty: how many validation
applicants and defaulters these numbers rest on.

## Explanations

TODO: The features that drive the model globally (reports/explain/shap_global.png), and
how a declined applicant is given reasons (reports/explain/explanations.md).

## Fairness

TODO: The main gaps by gender and age, how reliable they are, and the proxy finding. Link
docs/fairness_analysis.md.

## Monitoring

TODO: What is logged for each decision, how drift is measured, what batch 2 showed, and
the alerts with their thresholds. Link docs/monitoring_report.md.

## Limitations

TODO: When not to trust the model (data it never saw, groups with few applicants, a
changed upstream system...), and what is unknown so far (for example, live performance
before the outcomes are known).

## Ethical considerations

TODO: Who could be harmed by a wrong decision, in which direction (declined good payers,
approved defaulters), and what protects them (appeal, human review, monitoring).
