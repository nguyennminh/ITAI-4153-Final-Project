# Champion / challenger decision record (Week 7)

Team: Group 4 · Date: November 23, 2026 · Decided by: Group 4 (individual decision makers not yet confirmed)

## 1. What the outcomes showed

The live model was version 1, with a threshold of 0.080. Once the labels arrived, its performance was:

| Slice | AUC | Cost per applicant | Bad rate among approved |
|---|---|---|---|
| Validation | 0.747 | 0.533 | 4.4% |
| Batch 1 | 0.746 | 0.662 | 6.2% |
| Batch 2 | 0.711 | 0.898 | 9.8% |

Compared with validation, batch 2 lost 0.036 AUC and cost an additional 0.365 per applicant, or 365 cost units per 1,000 applicants. Total expected cost rose from 533 to 898 per 1,000. The bad rate among approved applicants rose by 5.4 percentage points. These are the project's weighted cost units: approving a defaulter costs 10 and rejecting a good applicant costs 1.

Week 6 monitoring showed four features with major drift in batch 2, but score PSI was only 0.014, compared with 0.013 in batch 1. Mean scores moved only from 0.088 to 0.090, while approvals fell from 74.4% to 70.1%. Monitoring detected input changes, but the small score-distribution change did not reveal the size of the performance loss. Without labels, it could not measure AUC, realized classification cost, or the increased bad rate among approved applicants. The outcomes later showed a 16% default rate in batch 2, versus 8.1% in validation.

## 2. The challenger

Version 2 uses the champion's histogram gradient boosting recipe, the same hyperparameters and 164 features, and isotonic calibration. It was trained on 4,000 original training applicants plus 1,436 labeled production applicants from batches 1 and 2: 5,436 rows, including 511 defaults. The fixed recent holdout contains 564 production applicants, including 72 defaults; neither model trained on those rows. The original 1,000-row validation slice also stayed out of training.

The challenger threshold is 0.095, selected using out-of-fold predictions and the same cost rule. Production rows help the model adapt to the applicants it is seeing now. However, batch 2 includes upstream changes: income was multiplied by 0.7, income ratios were recomputed, and about 40% of its applicants lost EXT_SOURCE_3. Training on those rows risks learning an upstream data problem as if it were a lasting population change. We should repair and validate the input pipeline, then reassess performance on clean future data.

## 3. The comparison

The paired 95% bootstrap intervals below describe challenger minus champion. Values come from the saved comparison report; differences use its underlying precision.

| Fixed slice | Champion AUC | Challenger AUC | AUC difference [95% interval] | Champion cost | Challenger cost | Cost difference [95% interval] |
|---|---|---|---|---|---|---|
| Validation, 1,000 applicants | 0.747 | 0.751 | +0.004 [-0.024, +0.034] | 0.533 | 0.509 | -0.024 [-0.114, +0.068] |
| Recent, 564 applicants | 0.748 | 0.735 | -0.012 [-0.042, +0.019] | 0.722 | 0.681 | -0.041 [-0.168, +0.066] |

The challenger was cheaper on both slices: an estimated saving of 24 cost units per 1,000 validation applicants and 41 per 1,000 recent applicants. Its validation AUC improved slightly, but its recent AUC decreased. Every difference interval includes zero, so these samples do not establish a clear improvement or regression at the 95% level.

Each model uses its own selected threshold because retraining and calibration can change the score scale and the cost-minimizing operating point. This compares the models as we would use them. The lower cost also comes with fewer approvals: 64.1% versus 74.9% on validation, and 56.2% versus 71.6% on recent applicants.

## 4. The gate

| Rule | Challenger result | Outcome |
|---|---|---|
| Validation AUC at least 0.70 | 0.751 | Pass |
| Validation cost at most 0.60 | 0.509 | Pass |
| Validation AUC no more than 0.01 below champion | 0.751 versus champion 0.747; minimum approximately 0.737 | Pass |
| Zero failed model tests | 2 failed, 1 passed, 1 expected failure | Fail |

We reran the model tests against `reports/retrain/challenger.json` to identify the two failures. `test_minimum_auc` passed. Both `test_application_weekday_does_not_change_the_score` and `test_higher_income_does_not_raise_risk` unexpectedly passed their assertions. Because they still have `xfail(strict=True)` marks from the older model, pytest reports these improvements as failures (`XPASS(strict)`). The gate counts those failures and therefore correctly returns KEEP under the current configuration.

`test_gender_does_not_change_the_score` remains an expected failure: the challenger still violates the gender-invariance requirement. The earlier measurements in `test_findings.md` describe the older model and should not be treated as version 2 measurements. The weekday and income marks need review for the new model, and the gender issue remains unresolved.

## 5. The decision

**Decision: PROMOTE version 2, overriding the automated KEEP decision.**

We chose to override because the challenger had lower measured cost on both fixed slices and slightly better validation AUC. The two failures blocking the gate came from behavior checks that now passed but still carried strict expected-failure marks. Our initial review did not catch that distinction. The override records our judgment rather than changing the saved automated decision.

We accept this as a provisional project decision. Recent AUC is lower, the estimated cost savings remain uncertain, fewer applicants are approved, and gender sensitivity remains a known failure. The override does not resolve that behavior requirement. Evidence of higher cost on new labeled batches, a persistent AUC decline, unacceptable approval changes, or continued problems after repairing the batch 2 inputs would make us reconsider and roll back.

## 6. What we did

The local champion history records promotion from version 1 to version 2 on November 23, 2026, at 15:09:13 EST. The recorded command and reason were:

```bash
python run.py promote --override "overriding because AUC performance is better and the cost is actually cheaper but failed to get caught by our performance check"
```

That command also runs export. `reports/champion.json` and `models/model_info.json` confirm local version 2 with threshold 0.095. The comparison above qualifies the original override wording: AUC improved only on validation, and the gate was blocked by strict unexpected passes.

Upload and live deployment are not confirmed by the local records. The remaining procedure in `cloud_shell_week7.md` is to upload `model.joblib`, `model_info.json`, and `feature_store.parquet` from `models/` to the bucket's `models/v2/` folder, then run in Cloud Shell with the configured bucket and service URL:

```bash
gcloud run services update credit-risk-api --region=us-central1 \
    --update-env-vars=MODEL_BUCKET=gs://$BUCKET/models/v2
curl -s "$URL/model-info"
```

Verify version 2 and threshold 0.095 in `/model-info`, then run `python run.py check` from the configured project environment. The current live version and those checks remain unverified in this record.

To roll back, run `python run.py rollback --version 1` locally to restore the alias and export, then point Cloud Run at the preserved version 1 bundle:

```bash
gcloud run services update credit-risk-api --region=us-central1 \
    --update-env-vars=MODEL_BUCKET=gs://$BUCKET/models/v1
curl -s "$URL/model-info"
```

Confirm version 1 and threshold 0.080. The documented switch takes approximately one minute without rebuilding the image; an actual timed rollback has not been recorded here.
