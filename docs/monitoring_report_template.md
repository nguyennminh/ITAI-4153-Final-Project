<!-- PROVIDED: copy this file to docs/monitoring_report.md and fill in every placeholder.
     Sections 1-4 by Wednesday's class; section 5 (alerts) and 6 by Sunday.
     Use the numbers from reports/monitoring/summary.json and drift_batch*.csv. -->

# Monitoring report (Week 6)

Team: TODO

## 1. What we monitor, and where the data comes from

TODO: How a prediction is logged (which fields, and why each one is there), where the log
goes on Cloud Run, and how you downloaded it (the command, how many records, the time range).
What is the reference you compare with, and why that one?

## 2. Batch 1

TODO: Requests, approval rate and mean score, compared with the validation set. Did any
feature drift (PSI above 0.1)? Evidently flagged some columns as drifted: look at them. Are
they real changes, or what you would expect by chance when you test 164 columns at once?

## 3. Batch 2: what changed?

TODO: Your diagnosis. For each feature that changed, a row of this table, then your
explanation.

| Feature | PSI | KS statistic | Missing (training → batch 2) | Mean (training → batch 2) | What changed |
|---|---|---|---|---|---|
| TODO | | | | | |

TODO: Which changes are connected (one cause, several features)? What most likely happened
upstream, in plain language? Did the scores and the approval rate move, and is that
reassuring or not? (Hint: what can the model's output NOT tell you yet?)

## 4. How we measured it

TODO: PSI and KS in your own words: what each one measures, when the KS p-value is
misleading, and why missing values must count as drift. Where our PSI and Evidently's
verdict agree or disagree, and why.

## 5. Proposed alerts

TODO: At least four alerts. For each: the signal, the threshold, how often it is checked,
and what it triggers (who is told, and what they do first).

| Signal | Threshold | Checked | Triggers |
|---|---|---|---|
| TODO | | | |

## 6. Limitations

TODO: What this monitoring cannot see (for example: whether the decisions were right), how
long until it could, and what you would add with more time.
