<!-- PROVIDED: copy this file to docs/decision_record.md and fill in every placeholder.
     Use reports/retrain/outcomes.md and reports/retrain/comparison.md. -->

# Champion / challenger decision record (Week 7)

Team: TODO · Date: TODO · Decided by: TODO (names)

## 1. What the outcomes showed

TODO: The live model's real performance on batches 1 and 2 (AUC, cost per applicant, bad
rate among approved) compared with the validation set. How big is the drop on batch 2, and
what does it cost per 1,000 applicants? What had the monitoring said in Week 6, and what did
it miss?

## 2. The challenger

TODO: Version, how it was trained (which rows, how many, which recipe), and its threshold.
Why train on the production applicants at all, and what is the risk of training on batch 2
(remember what you found was wrong with it)?

## 3. The comparison

TODO: Both fixed slices, both models: AUC and cost with the 95% intervals of the differences.
Which differences are real, and which could be chance? Why does each model use its own
threshold?

## 4. The gate

TODO: What your Week 4 gate said about the challenger, rule by rule (AUC floor, cost limit,
AUC drop against the champion, model tests). If a model test failed or unexpectedly passed,
which one, and what does it mean?

## 5. The decision

**Decision:** TODO (PROMOTE version N / KEEP version N)

TODO: Your reasons, in a few sentences. Did you follow your rule (decision.json), or overrule
it, and why? What would change your mind?

## 6. What we did

TODO: The commands (promote or rollback, export, upload, the service update), the version the
live service runs now (/model-info), and how you checked it. How you would roll back, and how
long it takes.
