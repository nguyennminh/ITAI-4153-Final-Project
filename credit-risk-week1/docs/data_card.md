# Data card: Home Credit Default Risk (Week 1 sample)

## 1. Source and license terms

The data comes from the Kaggle competition *Home Credit Default Risk* (2018), published by
Home Credit Group, a lender that serves people with little or no credit history. Each team
member downloads it personally after accepting the competition rules. The rules limit the
data to competition and non-commercial or academic use and forbid sharing or
redistributing it. To respect that, `.gitignore` keeps `data/`, `*.zip` and `kaggle.json`
out of GitHub. The repository holds only code and summary reports (row counts, memory
use, metrics), never the rows themselves.

## 2. What each table holds

| Table | One row is... | Links by |
|---|---|---|
| application_train | one loan applicant and their application, with `TARGET` | `SK_ID_CURR` (primary key) |
| bureau | one credit the applicant had at another lender, as reported to the credit bureau | `SK_ID_CURR`; own key `SK_ID_BUREAU` |
| bureau_balance | one month of status history for one bureau credit | `SK_ID_BUREAU` → bureau (no applicant id) |
| previous_application | one earlier application the applicant made to Home Credit | `SK_ID_CURR`; own key `SK_ID_PREV` |
| POS_CASH_balance | one monthly snapshot of an earlier point-of-sale or cash loan | `SK_ID_PREV`, `SK_ID_CURR` |
| installments_payments | one scheduled installment on an earlier Home Credit loan and what was actually paid | `SK_ID_PREV`, `SK_ID_CURR` |
| credit_card_balance | one monthly snapshot of an earlier Home Credit credit card | `SK_ID_PREV`, `SK_ID_CURR` |

## 3. Full size versus sample size

| Table | Full rows | Sample rows | Sample MB before → after |
|---|---:|---:|---:|
| application_train | 307,511 | 5,000 | 8.2 → 1.6 |
| bureau | 1,716,428 | 23,680 | 6.5 → 1.2 |
| bureau_balance | 27,299,925 | 231,091 | 14.5 → 1.3 |
| previous_application | 1,670,214 | 23,009 | 23.5 → 2.0 |
| POS_CASH_balance | 10,001,358 | 138,350 | 14.7 → 2.9 |
| installments_payments | 13,605,401 | 186,311 | 11.4 → 5.3 |
| credit_card_balance | 3,840,312 | 50,668 | 11.2 → 3.8 |
| **Total** | **58,441,149** | **658,109** | **89.9 → 18.1** |

The sample keeps 5,000 of the 307,511 applicants (1.6%), about 1.1% of all rows. Shrinking
dtypes cuts its memory by 80%. `scripts/make_sample.py` picks applicants with seed 42, separately
within each TARGET class, and then copies every row that belongs to them in all 7 tables. We
sample whole applicants rather than random rows so that each person's credit history is complete.
Random rows would leave applicants with only part of their past loans, and later features built
from those tables would be wrong. Everyone's sample has the same hash, which starts with `5f46b5f1`.

## 4. Target definition and class balance

`TARGET = 1` means the client had payment difficulties: a payment more than a set number of days
late on at least one of the loan's first installments. `TARGET = 0` covers everyone else. The
default rate is 8.07% in the full data, 8.08% in the sample (404 of 5,000), 8.08% in train
(323 of 4,000) and 8.10% in validation (81 of 1,000). The fixed split is stratified so
that both halves have the same rate. Otherwise a lucky or unlucky split would move the
metrics, especially PR-AUC, which depends directly on the default rate.

## 5. Known issues

In application_train, 25% of all cells are empty, and 65 of the 122 columns have gaps. 48 columns
are more than half empty, almost all of them building-description columns (`_AVG`/`_MODE`/`_MEDI`
versions at about 70% missing). `EXT_SOURCE_1` is 57% empty and `OCCUPATION_TYPE` 32%. `DAYS_EMPLOYED`
holds the placeholder 365243 for 897 applicants (17.9%). All of them are pensioners, and they
default less (4.9% vs 8.8%), so `clean_application` replaces the placeholder with NaN and adds the
flag `DAYS_EMPLOYED_ANOM`. All `DAYS_*` columns count days *before* the application as negative
numbers: `DAYS_BIRTH = -12000` means about 33 years old, and the sample ranges from 21 to 69. The
child tables also have gaps (bureau 13.8%, previous_application 17.9%, credit_card_balance
6.3% of cells), and we have not explored them in detail yet.

## 6. Sensitive columns

`CODE_GENDER` and `DAYS_BIRTH` (age) are protected characteristics. `NAME_FAMILY_STATUS`,
`CNT_CHILDREN` and `NAME_EDUCATION_TYPE` are also sensitive. Several columns act as proxies.
`NAME_INCOME_TYPE` and `DAYS_EMPLOYED_ANOM` mark pensioners, so they reveal age, and `OCCUPATION_TYPE`
and the region ratings can be linked to gender and income. Men default more often than women in the
sample (10.6% vs 6.8%), so a model will use gender if it is allowed to. Dropping these columns
alone does not fix this, because the proxies remain. We keep them available for auditing and will
compare error rates by gender and age group before any model is used.

## 7. Limitations of a small sample

The validation set has only 81 defaults, so the metrics are noisy. Recalculating the baseline's
validation ROC-AUC on 2,000 resamples of those 1,000 applicants gives a 95% range of about 0.65–0.76
around the 0.71 result. A gain of a few hundredths between two models may be chance. Rare categories
are thin or missing: `ORGANIZATION_TYPE` has 57 of its 58 values in the sample, and "Academic
degree" has a single applicant, so those rates cannot be trusted. Any category the model never saw
in training is ignored at prediction time. Results on 5,000 applicants should be confirmed on the
full data before drawing firm conclusions.
