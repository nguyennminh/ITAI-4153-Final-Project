# Test findings (Week 4)

Team: Group 4

The counts below come from the committed report dated 2026-11-02, recorded at `b1a100c-dirty`. They describe that saved run, not a new verification of the current branch. Regenerate the report on the machine with the feature table and registered model after these documentation changes.

| Kind | Tests | Passed | Failed | Known failures (xfail) | Skipped |
|---|---|---|---|---|---|
| Unit | 5 | 5 | 0 | 0 | 0 |
| Data | 5 | 5 | 0 | 0 | 0 |
| Model | 4 | 1 | 0 | 3 | 0 |

## Limits we chose, and why

These are the limits implemented in the tests. The committed report shows the data tests passed on that run; it does not record exact feature minima, maxima, or missingness rates. The explanations below distinguish dataset expectations and project policy from measured model failures.

| Test | Limit | Reason |
|---|---|---|
| `test_one_row_per_applicant` | Exactly 5,000 rows and zero duplicate `SK_ID_CURR` values. | The configured sample contains 5,000 applicants. Missing rows or duplicate IDs suggest filtering or an incorrect join and would change the unit of prediction. |
| `test_schema` | All ten listed columns must exist; nonmissing `TARGET` values must be 0 or 1; the listed ID, financial, age, external-score, and ratio columns must be numeric. | These are the inputs and labels required downstream. Missing columns, unexpected labels, or numeric values loaded as text can break training or silently alter preprocessing. The missing-values test separately rejects null targets. |
| `test_value_ranges` | `DAYS_BIRTH` from -25,550 to -6,570, income greater than zero, and every nonmissing `*_RATE` or `*_SHARE` value in [0, 1]. | The age band implements the assignment's approximately 18-to-70-year scope using 365 days per year. Positive income is the dataset assumption and is needed for income ratios. Rates and shares outside [0, 1] indicate invalid aggregation or arithmetic. The age and income limits describe this dataset, not universal rules about credit applicants. |
| `test_missing_values` | No missing income, credit, birth date, or target; `EXT_SOURCE_2` missingness below 1%; no entirely missing column. | Core financial and label fields must be usable. The 1% cap permits a small amount of external-score missingness while flagging a material loss of coverage. For 5,000 rows, fewer than 50 external scores may be missing. An entirely empty feature suggests a failed source or transformation. |
| `test_default_rate` | Default rate from 6% to 10%, inclusive. | The expected rate is about 8%. A two-percentage-point margin on either side allows sampling variation while flagging substantial changes in sampling or labels. This is a coarse data-quality guard, not a statistical confidence interval. |
| `test_minimum_auc` | Validation ROC AUC at least 0.70. | The provided application-only baseline is approximately 0.710. The 0.70 floor catches a major ranking regression without requiring an exact reproduction of that baseline. Passing this floor alone does not establish deployment readiness. |
| `test_application_weekday_does_not_change_the_score` | Maximum absolute probability change at most 0.001 after setting weekday to MONDAY. | The project requires weekday invariance. A 0.001 tolerance means 0.1 percentage points of default probability and is a practical policy tolerance, not merely machine precision. The recorded maximum of 0.004806 exceeds it. |
| `test_gender_does_not_change_the_score` | Maximum absolute probability difference at most 0.01 when scoring every applicant as F versus M. | The assignment specifies this one-percentage-point tolerance to detect material direct sensitivity to gender. It is more permissive than the weekday test and does not prove broader fairness or absence of proxy effects. The recorded maximum of 0.118219 exceeds it. |
| `test_higher_income_does_not_raise_risk` | Increase income by 50%, recompute income-dependent ratios, and allow no probability increase greater than 0.001. | This encodes the project's directional requirement while allowing increases of at most 0.1 percentage points. Testing a 50% change gives a clear perturbation with loan amounts held fixed. The recorded maximum increase of 0.049387 exceeds it. |

We retain the original assertions and thresholds for all three behavior failures. Each is marked `xfail(strict=True)`: its known failure is reported, and an unexpected pass fails the suite so the mark must be revisited. These marks document unmet requirements; they do not fix the model or authorize its use in lending.

## Findings

The measurements below are previously recorded findings. A new test run is needed to confirm they still describe the current registered model. To reproduce the assertion messages despite the marks, run `python run.py test tests/model -v --runxfail`. After investigating, regenerate the standard report without `--runxfail`.

### test_application_weekday_does_not_change_the_score

- **What failed:** Changing every applicant's application weekday to MONDAY changed predicted default probabilities by as much as 0.004806, exceeding the allowed tolerance of 0.001.
- **Why:** The model's predictions are sensitive to the application weekday. The test confirms that changing this input changes predictions, although further feature-importance and model inspection would be needed to establish exactly how the model learned this relationship.
- **Does it matter for the lender?** Yes, because applicants should not receive different risk assessments simply because they submitted applications on different days of the week.
- **Decision:** Keep this as a known failure using @pytest.mark.xfail(strict=True, reason="see docs/test_findings.md") while investigating the feature's contribution and considering removing WEEKDAY_APPR_PROCESS_START from the production model.


### test_gender_does_not_change_the_score

- **What failed:** Changing applicants' gender between F and M changed predicted default probabilities by as much as 0.118219, exceeding the allowed tolerance of 0.01.
- **Why:**  The model is sensitive to CODE_GENDER, demonstrating that gender changes can substantially affect its predictions. Additional feature and model analysis is needed to determine the exact contribution of gender and any related proxy features.
- **Does it matter for the lender?** Yes, this is a significant fairness concern because applicants should not receive different credit-risk assessments based solely on gender.
- **Decision:** Keep this as a known failure using @pytest.mark.xfail(strict=True, reason="see docs/test_findings.md") while investigating the model's use of gender and related proxy features. The long-term goal is to remove protected-attribute influence and verify fairness before production use.


### test_higher_income_does_not_raise_risk

- **What failed:** Increasing applicants' income by 50% caused 35 applicants to receive higher risk scores beyond the allowed tolerance, with the largest increase being 0.049387 instead of the permitted 0.001.
- **Why:**  The model does not consistently associate higher income with lower default risk. The helper also recomputes income-dependent financial ratios. This test alone cannot identify which of those inputs caused the increase; feature and model inspection is still needed.
- **Does it matter for the lender?** Yes, because applicants earning more money should not automatically receive higher default-risk predictions when all other relevant information remains unchanged.
- **Decision:** Keep this as a known failure using @pytest.mark.xfail(strict=True, reason="see docs/test_findings.md") while investigating the income feature, derived ratios, and model behavior. Consider adjusting feature engineering or retraining the model to address the unexpected risk increases.