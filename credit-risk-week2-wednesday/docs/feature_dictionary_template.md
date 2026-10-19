<!-- PROVIDED template. Copy it to docs/feature_dictionary.md and fill that copy in:
       cp docs/feature_dictionary_template.md docs/feature_dictionary.md   (Windows: copy ...)
     Replace every "TODO(student)". Keep the table format: python run.py check reads it.
     Add a row for every feature you create yourself. -->

# Feature dictionary

Team: TODO(student)

One row per engineered feature in `data/features/features.parquet`. Someone who has never
seen your code should understand each feature from this page alone.

- **Meaning:** what the number says about the applicant, in plain language.
- **How it was built:** the source columns, the aggregation, any filtering or special cases
  (missing values, the 365243 sentinel, zero denominators), and what an applicant with no
  records gets (0 or missing).

| Feature | Source table | Meaning | How it was built |
|---|---|---|---|
| BUREAU_COUNT | bureau | How many credits other lenders report for the applicant, open or closed. | Count of the applicant's rows in bureau (`aggregate_bureau`, named aggregation on SK_ID_BUREAU). 0 for applicants with no bureau record (filled in `join_features`). |
| BUREAU_AMT_CREDIT_SUM_MEAN | bureau | TODO(student) | TODO(student) |
| BUREAU_AMT_CREDIT_SUM_MAX | bureau | TODO(student) | TODO(student) |
| BUREAU_AMT_DEBT_MEAN | bureau | TODO(student) | TODO(student) |
| BUREAU_AMT_OVERDUE_SUM | bureau | TODO(student) | TODO(student) |
| BUREAU_DAYS_OVERDUE_MAX | bureau | TODO(student) | TODO(student) |
| BUREAU_DAYS_CREDIT_MAX | bureau | TODO(student) | TODO(student) |
| BUREAU_ACTIVE_COUNT | bureau | TODO(student) | TODO(student) |
| BUREAU_CLOSED_COUNT | bureau | TODO(student) | TODO(student) |
| PREV_COUNT | previous_application | TODO(student) | TODO(student) |
| PREV_APPROVED_RATE | previous_application | TODO(student) | TODO(student) |
| PREV_REFUSED_RATE | previous_application | TODO(student) | TODO(student) |
| PREV_AMT_APPLICATION_MEAN | previous_application | TODO(student) | TODO(student) |
| PREV_AMT_CREDIT_MEAN | previous_application | TODO(student) | TODO(student) |
| PREV_DAYS_DECISION_MAX | previous_application | TODO(student) | TODO(student) |
| PREV_DAYS_TERMINATION_MEAN | previous_application | TODO(student) | TODO(student) |
| PREV_LAST_STATUS | previous_application | TODO(student) | TODO(student) |
| BB_CREDIT_COUNT | bureau_balance | TODO(student) | TODO(student) |
| BB_MONTHS_COUNT | bureau_balance | TODO(student) | TODO(student) |
| BB_LATE_MONTHS_COUNT | bureau_balance | TODO(student) | TODO(student) |
| BB_LATE_SHARE | bureau_balance | TODO(student) | TODO(student) |
| BB_WORST_STATUS | bureau_balance | TODO(student) | TODO(student) |
| POS_LOAN_COUNT | POS_CASH_balance | TODO(student) | TODO(student) |
| POS_MONTHS_COUNT | POS_CASH_balance | TODO(student) | TODO(student) |
| POS_DPD_MAX | POS_CASH_balance | TODO(student) | TODO(student) |
| POS_DPD_MEAN | POS_CASH_balance | TODO(student) | TODO(student) |
| POS_LATE_SHARE | POS_CASH_balance | TODO(student) | TODO(student) |
| POS_COMPLETED_SHARE | POS_CASH_balance | TODO(student) | TODO(student) |
| INST_COUNT | installments_payments | TODO(student) | TODO(student) |
| INST_LATE_SHARE | installments_payments | TODO(student) | TODO(student) |
| INST_DAYS_LATE_MAX | installments_payments | TODO(student) | TODO(student) |
| INST_UNDERPAID_SHARE | installments_payments | TODO(student) | TODO(student) |
| INST_PAYMENT_DIFF_MEAN | installments_payments | TODO(student) | TODO(student) |
| CC_MONTHS_COUNT | credit_card_balance | TODO(student) | TODO(student) |
| CC_BALANCE_MEAN | credit_card_balance | TODO(student) | TODO(student) |
| CC_UTILIZATION_MEAN | credit_card_balance | TODO(student) | TODO(student) |
| CC_UTILIZATION_MAX | credit_card_balance | TODO(student) | TODO(student) |
| CC_DRAWINGS_MEAN | credit_card_balance | TODO(student) | TODO(student) |
| CC_DPD_MAX | credit_card_balance | TODO(student) | TODO(student) |
| CREDIT_INCOME_RATIO | application_train | TODO(student) | TODO(student) |
| ANNUITY_INCOME_RATIO | application_train | TODO(student) | TODO(student) |
| CREDIT_GOODS_RATIO | application_train | TODO(student) | TODO(student) |
| ANNUITY_CREDIT_RATIO | application_train | TODO(student) | TODO(student) |

The application columns (from `application_train`) are described in
`HomeCredit_columns_description.csv` and in your data card; they do not need rows here.
