# Real performance of the live model

Model version 1, threshold 0.080. Batches: the predictions the live service logged, joined with the true outcomes. Written by python run.py outcomes.

| Applicants | n | Default rate | Mean score | AUC | Cost per applicant | Approval rate | Bad rate among approved |
|---|---|---|---|---|---|---|---|
| validation (before going live) | 1000 | 8.1% | 0.081 | 0.747 | 0.533 | 74.9% | 4.4% |
| batch 1 (live) | 1000 | 10.0% | 0.088 | 0.746 | 0.662 | 74.4% | 6.2% |
| batch 2 (live) | 1000 | 16.0% | 0.090 | 0.711 | 0.898 | 70.1% | 9.8% |

## What the monitoring said in Week 6 (before the outcomes)

| Batch | Approval rate | Mean score | Score PSI | Features with major drift |
|---|---|---|---|---|
| 1 | 74.4% | 0.088 | 0.013 | 0 |
| 2 | 70.1% | 0.090 | 0.014 | 4 |
