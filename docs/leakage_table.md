# Leakage table

Team: Not specified

**The question for every feature:** at the moment an applicant submits this application,
before we decide, would we already know this value?

If the answer is No, the feature **leaks**: it carries information from the future (often from
the outcome itself). A leaking feature makes validation scores look great and then fails in
real use, because the value does not exist yet when the decision is made.

**How to check, not guess:** the `DAYS_*` and `MONTHS_*` columns count days or months
**relative to this application** (negative = before it). Look at the largest value of each
time column a feature is built from. If it is above 0, part of that feature describes the
future. One look at `df["COL"].max()` is better than an opinion.

Checks below use the local `data/sample` CSVs: 5,000 applications, 23,680 bureau
rows, 23,009 previous applications, 231,091 bureau-balance rows, and 138,350 POS rows.
Descriptions were checked in `data/raw/HomeCredit_columns_description.csv`. Maxima are
sample evidence, not guarantees about other datasets. Month 0 is treated as available in
the supplied application-time snapshot; monthly granularity alone cannot establish an
exact within-month timestamp.

Other inspected date maxima: bureau `DAYS_CREDIT_ENDDATE` = 31195 and previous_application
`DAYS_LAST_DUE_1ST_VERSION` = 1781 after excluding 365243. These are scheduled dates and
are not inputs to the listed features; positive planned dates alone do not prove an
observed future outcome was used. Future realized outcomes would need exclusion.

## Features you built (and the worked example)

| Feature | Source table | Known at application time? | Why / how you checked |
|---|---|---|---|
| BUREAU_COUNT | bureau | Yes | The bureau report is pulled when the application arrives. Every DAYS_CREDIT in our sample is negative (max -2), so all counted credits were opened before this application. |
| BUREAU_ACTIVE_COUNT | bureau | Yes | The reported Active status is available in the bureau snapshot; DAYS_CREDIT max -2 and DAYS_CREDIT_UPDATE max 0 confirm opening and reporting by application time. |
| BUREAU_CLOSED_COUNT | bureau | Yes | The reported Closed status is available in the bureau snapshot; DAYS_ENDDATE_FACT max 0 and DAYS_CREDIT_UPDATE max 0 place closure and reporting by application time. |
| BUREAU_AMT_CREDIT_SUM_MEAN | bureau | Yes | Mean reported credit amount is available from the bureau snapshot; DAYS_CREDIT max -2 and DAYS_CREDIT_UPDATE max 0. |
| BUREAU_AMT_CREDIT_SUM_MAX | bureau | Yes | Largest reported credit amount is available from the bureau snapshot; DAYS_CREDIT max -2 and DAYS_CREDIT_UPDATE max 0. |
| BUREAU_AMT_DEBT_MEAN | bureau | Yes | Mean current reported debt is known from the bureau snapshot, whose DAYS_CREDIT_UPDATE max is 0; this does not use later repayment outcomes. |
| BUREAU_AMT_OVERDUE_SUM | bureau | Yes | Current reported overdue amounts are known from the bureau snapshot; DAYS_CREDIT_UPDATE max 0 confirms no future update. |
| BUREAU_DAYS_OVERDUE_MAX | bureau | Yes | CREDIT_DAY_OVERDUE measures current days overdue, not a future event date; the bureau snapshot has DAYS_CREDIT_UPDATE max 0. |
| BUREAU_DAYS_CREDIT_MAX | bureau | Yes | Checked DAYS_CREDIT directly: max -2, so even the newest counted credit opened before this application. |
| PREV_COUNT | previous_application | Yes | All counted earlier applications have DAYS_DECISION max -2, so their decisions predate this application. |
| PREV_APPROVED_RATE | previous_application | Yes | Approval decisions belong to earlier applications, not the current outcome; DAYS_DECISION max -2. |
| PREV_REFUSED_RATE | previous_application | Yes | Refusal decisions belong to earlier applications; DAYS_DECISION max -2 confirms they are already known. |
| PREV_AMT_APPLICATION_MEAN | previous_application | Yes | Requested amounts were submitted with earlier applications; their DAYS_DECISION max is -2. |
| PREV_AMT_CREDIT_MEAN | previous_application | Yes | Amounts offered in earlier decisions are already known; DAYS_DECISION max -2. |
| PREV_DAYS_DECISION_MAX | previous_application | Yes | Checked DAYS_DECISION directly: max -2, so the most recent decision predates the current application. |
| PREV_DAYS_TERMINATION_MEAN | previous_application | Yes | Description identifies expected termination relative to the current application. Raw max 365243 is the no-date sentinel; after replacing it with NaN, max -2 (zero real positive dates). aggregate_previous applies that replacement before averaging. |
| PREV_LAST_STATUS | previous_application | Yes | Uses the status of the earlier application with largest DAYS_DECISION (max -2), breaking ties with SK_ID_PREV; no current-application outcome is used. |
| BB_CREDIT_COUNT | bureau_balance | Yes | Counts credits with recorded monthly history; bureau_balance MONTHS_BALANCE max 0 and parent bureau DAYS_CREDIT max -2 place that history by application time. |
| BB_MONTHS_COUNT | bureau_balance | Yes | Counts recorded historical months; bureau_balance MONTHS_BALANCE max 0, with no positive months. |
| BB_LATE_MONTHS_COUNT | bureau_balance | Yes | Late statuses come from monthly records with bureau_balance MONTHS_BALANCE max 0, so no later month is counted. |
| BB_LATE_SHARE | bureau_balance | Yes | Both late-month numerator and recorded-month denominator use bureau_balance MONTHS_BALANCE max 0; no future monthly history enters. |
| BB_WORST_STATUS | bureau_balance | Yes | Worst reported status is taken over bureau_balance months whose MONTHS_BALANCE max is 0; it does not include future months. |
| POS_LOAN_COUNT | POS_CASH_balance | Yes | Counts earlier loans observed in POS_CASH_balance; MONTHS_BALANCE max -1 confirms all observations precede application. |
| POS_MONTHS_COUNT | POS_CASH_balance | Yes | Counts monthly observations with POS_CASH_balance MONTHS_BALANCE max -1, all before application. |
| POS_DPD_MAX | POS_CASH_balance | Yes | Maximum SK_DPD describes delinquency already observed in monthly records; POS_CASH_balance MONTHS_BALANCE max -1. |
| POS_DPD_MEAN | POS_CASH_balance | Yes | Mean SK_DPD uses only previously observed months; POS_CASH_balance MONTHS_BALANCE max -1. |
| POS_LATE_SHARE | POS_CASH_balance | Yes | Late-month numerator and month denominator both use POS_CASH_balance records with MONTHS_BALANCE max -1. |
| POS_COMPLETED_SHARE | POS_CASH_balance | Yes | Completed status is counted only when already recorded in a loan month; POS_CASH_balance MONTHS_BALANCE max -1 confirms completion observations predate application. |

## Application columns worth a second look

These already sit in `application_train` and went into your Week 1 baseline. Some are less
obvious than they look. Find each one in `HomeCredit_columns_description.csv` (in your
download) before you answer.

| Feature | Source table | Known at application time? | Why / how you checked |
|---|---|---|---|
| TARGET | application_train | No | No: the column description defines payment difficulties on installments of this loan, observed after the lending decision; it must be excluded from model inputs. |
| EXT_SOURCE_2 | application_train | Yes | The column description identifies a normalized score from an external data source, an application input rather than this loan's repayment outcome; no event-date column is supplied to independently verify score freshness. |
| DEF_30_CNT_SOCIAL_CIRCLE | application_train | Yes | The description counts observed defaults at 30 days past due in the client's social surroundings. This is an existing application observation, not the current loan's future default; no observation timestamp is supplied for an independent freshness check. |
| AMT_REQ_CREDIT_BUREAU_HOUR | application_train | Yes | The description explicitly counts bureau enquiries in the hour before application, so these enquiries have already happened. |
| DAYS_LAST_PHONE_CHANGE | application_train | Yes | The description measures days before application of the phone change; checked application_train DAYS_LAST_PHONE_CHANGE max 0, so no future change is included. |

## Beyond single columns

Leakage can also come from **how** a feature is computed, not only from which column it uses.
In two or three sentences: name one way your feature code could leak even if every column
above is a Yes, and how your code avoids it.

Fitting an imputer or encoder on all applicants before splitting would let validation data influence training even when every input column is available at application time. The aggregations use each applicant's own history without TARGET, and src/compare.py splits first and fits the preprocessing/model pipeline on train only, then scores validation.
