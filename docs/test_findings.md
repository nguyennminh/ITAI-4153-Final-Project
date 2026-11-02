<!-- PROVIDED: copy this file to docs/test_findings.md and fill in every placeholder.
     The check fails while the word in capitals below is still anywhere in the file. -->

# Test findings (Week 4)

Team: Group 4

Run `python run.py testreport` and copy the counts from `reports/test_report.md`.

| Kind | Tests | Passed | Failed | Known failures (xfail) | Skipped |
|---|---|---|---|---|---|
| Unit | 5 | 5 | 0 | 0 | 0 |
| Data | 5 | 5 | 0 | 0 | 0 |
| Model | 4 | 1 | 3 | 0 | 0 |

## Limits we chose, and why

For each data test and model test, the limit you picked (for example "default rate
between 6% and 10%", "no score moves by more than 0.001") and why that number: what you saw
in the data, and what a change past the limit would mean.

## Findings

Write one section per test that failed. Name the test exactly as in the report (for example
`test_higher_income_does_not_raise_risk`). If no test failed, write "No test failed" and
say which test you expected to be the most likely to fail, and why.

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
- **Why:**  The model does not consistently associate higher income with lower default risk. Increasing income also changes derived financial ratios, which can influence the model's predictions in unexpected ways.
- **Does it matter for the lender?** Yes, because applicants earning more money should not automatically receive higher default-risk predictions when all other relevant information remains unchanged.
- **Decision:** Keep this as a known failure using @pytest.mark.xfail(strict=True, reason="see docs/test_findings.md") while investigating the income feature, derived ratios, and model behavior. Consider adjusting feature engineering or retraining the model to address the unexpected risk increases.