# Week 2: Data engineering (Monday part)

<!-- PROVIDED -->

Last week you trained a baseline on **one** table, `application_train`. This week you bring in
the applicant's **credit history** from the six other tables, while keeping exactly one row
per applicant. Today you do the first two tables and measure whether they help.

## Why we aggregate

### The rule: one row = one loan decision

The model answers one question: *should we lend to this applicant?* So every row it trains on,
and every row it scores, must be **one applicant**, with one TARGET and one prediction.

`application_train` already has that shape: 5,000 rows for 5,000 applicants. The other tables
do not. They hold an applicant's **history**, so one person has many rows. Applicant 415899
has three credits at other lenders:

| SK_ID_CURR | SK_ID_BUREAU | CREDIT_ACTIVE | AMT_CREDIT_SUM |
|---|---|---|---|
| 415899 | 5714728 | Closed | 229,050 |
| 415899 | 5714729 | Active | 225,000 |
| 415899 | 5714730 | Active | 450,000 |

In our sample the median applicant has 4 bureau credits, one has 34, and 715 have none.

**Aggregation squashes those rows into one row of summary numbers about the person:**

| SK_ID_CURR | BUREAU_COUNT | BUREAU_ACTIVE_COUNT | BUREAU_AMT_CREDIT_SUM_MEAN | BUREAU_AMT_CREDIT_SUM_MAX |
|---|---|---|---|---|
| 415899 | 3 | 2 | 301,350 | 450,000 |

That row can sit next to the applicant's row in `application_train`.

### What goes wrong without aggregation

**Option 1: ignore the other tables.** That was Week 1 (ROC-AUC about 0.71). The six tables
are the applicant's track record: earlier loans, late payments, refused applications. In
lending, "how did this person handle credit before?" is usually one of the strongest signals.

**Option 2: join the raw tables directly.** On our sample:

| What you join | Rows you get | Should be |
|---|---|---|
| application only | 5,000 | 5,000 |
| application + bureau (raw) | 24,395 | 5,000 |
| application + bureau + previous_application (raw) | 127,657 | 5,000 |

The rows **multiply**: the applicant with 34 bureau credits and 12 earlier applications turns
into 34 x 12 = 408 rows. Then:

1. The same person counts hundreds of times, so people with long histories dominate training.
2. The default rate drifts: 8.08% per applicant becomes 7.53% per row after only one join.
3. Validation metrics count one applicant's single outcome many times.
4. A new applicant gets 408 different scores. Which one decides the loan?
5. Some tables cannot join at all: `bureau_balance` has no applicant id, only `SK_ID_BUREAU`.

**And nothing crashes.** A careless merge runs fine and quietly trains on the wrong data.
That is why `join_features` must check "5,000 in, 5,000 out" and fail loudly when it is not.

### Aggregation is where the lending knowledge goes in

Summaries such as the refusal rate of earlier applications, the share of months paid late,
or the number of credits still open describe **behavior**. Choosing them, and making sure we
would know them at application time (the leakage table), is the real work this week.

## What is new in your repository

Copy the contents of the Week 2 Monday folder into your repository root (see the task card).
It adds these files and replaces only `run.py`:

```
WEEK2.md                         this guide                                       PROVIDED
run.py                           adds compare, check (Week 2), check1 (Week 1)    PROVIDED
config/features.yaml             Week 2 settings and the feature sets to compare  PROVIDED
src/features/common.py           helpers + Week 2 config loader                   PROVIDED
src/features/bureau.py           bureau (worked example), bureau_balance          MIXED
src/features/previous.py         previous_application, POS_CASH_balance           TODO
src/features/join.py             one-row-per-applicant check and the join         TODO
src/uncertainty.py               bootstrap intervals for AUC                      PROVIDED
src/compare.py                   baseline on growing feature sets                 PROVIDED
scripts/check_week2.py           pass/fail checks for the checkpoint              DO NOT EDIT
tests/test_week2_mon.py          the tests your code must pass                    DO NOT EDIT
tests/week2_fixtures.py          tiny hand-made tables for those tests            DO NOT EDIT
docs/leakage_table_template.md   copy it to docs/leakage_table.md                 PROVIDED
```

Your Week 1 files are not touched. `python run.py compare` reuses **your** Week 1 pipeline
(`clean_application`, `get_feature_columns`, `build_preprocessor`, `build_model`, `evaluate`),
so the only thing that changes between feature sets is the features.

## Commands

| Command | Plain equivalent | What it does |
|---|---|---|
| `python run.py compare` | `python -m src.compare` | Baseline on each feature set -> `reports/auc_comparison.json` |
| `python run.py check` | `python -m scripts.check_week2` | Week 2 checkpoint checks |
| `python run.py check1` | `python -m scripts.check_week1` | Week 1 checks (still useful) |
| `python run.py test` | `python -m pytest` | All tests, Week 1 and Week 2 |

Run only today's tests while you work:
`python -m pytest tests/test_week2_mon.py -k previous` (or `-k join`, `-k bureau_balance`, ...).

## Monday path

1. Read `aggregate_bureau` in `src/features/bureau.py`. It is the pattern for everything else.
   Run `python -m pytest tests/test_week2_mon.py -k bureau_worked` to see it pass.
2. Write `aggregate_previous` in `src/features/previous.py`.
3. Write `assert_one_row_per_applicant` and `join_features` in `src/features/join.py`.
4. `python run.py compare`. Read the table it prints, especially the **95% CI of change**
   column, before you decide whether a table "helped".
5. Copy the leakage template to `docs/leakage_table.md` and start filling it in.
6. `python run.py check` until every required row says PASS.

## Carry-over (before the next class)

- `aggregate_bureau_balance` in `src/features/bureau.py`: two stages, credit first, then
  applicant.
- `aggregate_pos_cash` in `src/features/previous.py`.
- Uncomment the matching lines in `config/features.yaml` and rerun `python run.py compare`.
- Keep filling `docs/leakage_table.md`.

`python run.py check` shows the carry-over as INFO rows, so you can see your progress.

## Reading the comparison

The validation set has 1,000 applicants but only 81 defaults. With so few, AUC moves by a few
hundredths just from which applicants landed in validation. `compare` therefore prints:

- **95% CI**: the range the AUC would plausibly take on a different draw of applicants.
- **change** and its **95% CI**: AUC minus the application-only AUC, measured on the same
  applicants. If that interval contains 0, you cannot claim the table helped. Say so; that is
  an honest and correct result.
