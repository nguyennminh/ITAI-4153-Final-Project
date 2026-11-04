# Test report (Week 4)

Commit `4e69d01`, 2026-11-04T01:48:05+00:00.

## Unit tests (5 passed)

| Test | Outcome | Message |
|---|---|---|
| test_bureau_features_depend_only_on_own_credits | passed |  |
| test_previous_features_ignore_row_order | passed |  |
| test_join_keeps_applicants_without_history | passed |  |
| test_join_refuses_a_table_that_was_not_aggregated | passed |  |
| test_ratios_are_never_infinite | passed |  |

## Data tests (5 passed)

| Test | Outcome | Message |
|---|---|---|
| test_one_row_per_applicant | passed |  |
| test_schema | passed |  |
| test_value_ranges | passed |  |
| test_missing_values | passed |  |
| test_default_rate | passed |  |

## Model tests (1 passed, 3 xfail)

| Test | Outcome | Message |
|---|---|---|
| test_minimum_auc | passed |  |
| test_application_weekday_does_not_change_the_score | xfail | see docs/test_findings.md |
| test_gender_does_not_change_the_score | xfail | see docs/test_findings.md |
| test_higher_income_does_not_raise_risk | xfail | see docs/test_findings.md |
