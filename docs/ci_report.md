# Quality gate thresholds

- `min_auc: 0.70` requires useful discrimination above chance while allowing the registered model's validation AUC of 0.7471 to pass.
- `max_cost: 0.60` requires at least a 0.21 reduction in expected cost per applicant compared with approving everyone (0.81). The current cost of 0.533 leaves 0.067 of headroom.
- `max_auc_drop: 0.01` permits at most one percentage point of AUC regression against the registered model. With the current baseline of 0.7471, candidates must achieve at least 0.7371, a stricter requirement than the absolute floor.
- `allow_failing_model_tests: 0` requires every non-xfailed model test to pass. Known failures marked xfail remain documented without permitting new failures.
