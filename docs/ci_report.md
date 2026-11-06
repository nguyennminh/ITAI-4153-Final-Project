<!-- PROVIDED: copy this file to docs/ci_report.md and fill in every placeholder.
     The check fails while the word in capitals below is still anywhere in the file. -->

# CI and the quality gate (Week 4)

Team: Group 4

## The pipeline

Cloud Build project: `credit-risk-group4-513720`. Trigger: `pr-checks`, on pull requests to `main`. Branch protection on `main` requires the check: `pr-checks (credit-risk-group4-513720)`.

| Step | What it runs | Why it is there |
|---|---|---|
| install | `pip install -r requirements.txt` | Installs the same packages used during local development so the CI environment is consistent. |
| lint | `python -m ruff check .` | Checks the entire repository for Python formatting, style, import, and common programming errors. |
| test | `python -m pytest` | Runs the project's automated tests. Unit tests run because they use synthetic data and do not require the real dataset. Data tests are skipped because `data/features/features.parquet` is not included in GitHub. Model tests are also skipped because the fresh Cloud Build environment does not have the local data and registered model needed to run them. |
| gate | `python -m scripts.quality_gate` | Checks the committed evaluation metrics against the thresholds in `config/gate.yaml`. It also checks that the metrics match the current code fingerprint so outdated evaluation results cannot pass. |

## Gate thresholds, and why

- **min_auc** = `0.70`. This requires the candidate model's validation ROC-AUC to be at least 0.70. The current registered model has a validation AUC of `0.7471`, so it passes with room above the minimum. The threshold was chosen based on the Week 1 logistic regression baseline, which achieved approximately `0.710`. A result below 0.70 indicates that the model's ability to distinguish defaulting applicants from non-defaulting applicants has dropped below the project's acceptable level.

- **max_cost** = `0.60`. This compares the candidate model's expected cost per applicant at the selected decision threshold against a maximum of 0.60. The current model has a cost of `0.533` at a threshold of `0.08`, so it passes. Approving everyone has a cost of approximately `0.81`, meaning the selected limit requires a meaningful reduction in expected lending cost. A result above 0.60 means the model is not reducing the expected cost enough.

- **max_auc_drop** = `0.01`. This allows the candidate model's validation AUC to decrease by no more than 0.01 compared with the registered model. The registered model's AUC is `0.7471`, making the minimum acceptable AUC under this rule `0.7371`. The value was chosen to prevent a new model from significantly reducing predictive performance while allowing a small amount of variation. A candidate below 0.7371 fails this comparison.

- **allow_failing_model_tests** = `0`. This means that no unexpected model tests are allowed to fail. The current recorded model-test results are 1 passed, 0 failed, and 3 expected failures (`xfail`). The three known failures are documented in `docs/test_findings.md` and marked with `strict=True`. Expected failures do not count as unexpected failures in the quality gate. Any new, unexpected model-test failure causes the gate to fail.

## Evidence

- **A green Cloud Build run on a pull request:** [View successful Cloud Build](https://console.cloud.google.com/cloud-build/builds/e5d9a0fe-498b-46c6-b88b-9ac0aaa0980e?project=196591733163). This build was associated with [Pull Request #1 – Week4 ci](https://github.com/nguyennminh/ITAI-4153-Final-Project/pull/1). The build completed successfully, and all CI steps passed. The pull request was subsequently merged into `main`.

- **A red run, and what made it red:** [View failed Cloud Build](https://console.cloud.google.com/cloud-build/builds/4535f875-8608-4d3c-a8db-88075df0e76a?project=196591733163). This build was associated with [Pull Request #2 – update values to fail check](https://github.com/nguyennminh/ITAI-4153-Final-Project/pull/2). The pull request intentionally changed the gate thresholds to unrealistic values: `min_auc: 0.1`, `max_cost: 0.10`, and `max_auc_drop: 0.99`. The quality gate failed because the candidate model's cost of `0.533` was above the new maximum cost of `0.100`. The gate's failure message was:

  `cost 0.533 is above the limit 0.100`

- **How the red pull request was handled:** Pull Request #2 was closed without being merged. The unrealistic threshold changes were not accepted into `main`, preserving the original quality gate settings of `min_auc: 0.70`, `max_cost: 0.60`, `max_auc_drop: 0.01`, and `allow_failing_model_tests: 0`. This demonstrated that the quality gate correctly identifies unacceptable changes and prevents them from being merged.

## Evidence

- A green Cloud Build run on a pull request: https://github.com/nguyennminh/ITAI-4153-Final-Project/pull/1
  
- A red run, and what made it red: https://github.com/nguyennminh/ITAI-4153-Final-Project/pull/2

- How the red pull request was handled: The failed pull request (#2) was reviewed to identify why the quality gate did not pass. The failure occurred because the maximum allowed cost was lowered to 0.10, while the candidate model's actual cost was 0.533. Since the model exceeded the permitted limit, the quality gate correctly failed the build. The pull request was closed without merging, and the original quality gate thresholds on the main branch were preserved.