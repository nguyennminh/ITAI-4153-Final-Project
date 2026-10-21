# Week 2: Data engineering (Wednesday part)

<!-- PROVIDED -->

On Monday you aggregated four child tables and proved the join keeps one row per applicant.
Today you finish the last two tables, add ratio features, and package everything into **one
command** that builds a **validated** feature table:

```
python run.py features   ->   data/features/features.parquet   (5,000 rows, one per applicant)
                              reports/validation_sample.json   (pass / fail, and why)
```

## Why one script, and why validation

**One script.** So far your features live in functions that `compare` calls. A real team needs
one entry point that anyone (a teammate, a scheduled job, a grader) can run on a fresh clone and
get the same table. That is `src/make_features.py`.

**Validation.** Data changes without asking you. A column gets renamed upstream, a system starts
writing a status in capitals, a refund is booked as a negative payment. Two things can happen:

1. **The pipeline crashes** with a `KeyError` deep inside an aggregation. Someone has to read
   your code to learn that the data, not the code, is wrong.
2. **The pipeline does not crash at all.** If `"Approved"` becomes `"APPROVED"`,
   `PREV_APPROVED_RATE` quietly drops toward 0 for everyone and the model trains on nonsense.
   The second case is worse.

Validation checks the data against written rules **before** and **after** building features,
reports **every** problem in plain language, and **refuses to write** a feature table that broke
a rule. "Fail loudly" means: stop, say exactly what is wrong, and leave nothing half-built behind.

We use **pandera**: a *schema* lists the columns a table must have and the rules each must pass.
`src/validate.py` has a worked example, `bureau_schema()`. Read it first.

## Setup (once, one member)

Add these two lines to the end of `requirements.txt`, then everyone runs `python run.py setup`:

```
pandera>=0.20
pyarrow>=14
```

pandera checks the data; pyarrow writes the Parquet file (a compact, typed table format that
keeps column types, unlike CSV).

## What is new in your repository

Copy the Wednesday folder into your repository root (see the task card). It adds these files
and replaces a few PROVIDED or DO NOT EDIT files. It never touches your `bureau.py`,
`previous.py`, `join.py`, or your docs.

```
WEEK2_WEDNESDAY.md                this guide                                        PROVIDED
run.py                            adds features; check --stage wed                  PROVIDED (replaced)
config/features.yaml              output paths, validation limits, 8 feature sets   PROVIDED (replaced)
src/features/common.py            adds safe_divide                                  PROVIDED (replaced)
src/features/payments.py          installments_payments, credit_card_balance        TODO
src/features/ratios.py            four ratio features                               TODO
src/validate.py                   schemas: bureau worked example, the rest          MIXED
src/make_features.py              the one entry point                               MIXED
src/compare.py                    now covers every table and the ratios             PROVIDED (replaced)
scripts/check_week2.py            Wednesday and submission checks                   DO NOT EDIT (replaced)
tests/test_week2_wed.py           the tests your code must pass                     DO NOT EDIT
tests/test_week2_mon.py           Monday's tests, with the new feature set names    DO NOT EDIT (replaced)
tests/week2_fixtures.py           adds hand-made installments and card tables       DO NOT EDIT (replaced)
docs/feature_dictionary_template.md   copy to docs/feature_dictionary.md            PROVIDED
docs/leakage_table_template.md    adds the new features' rows                       PROVIDED (replaced)
```

`config/features.yaml` is replaced: its feature sets now add one table at a time, so you no
longer uncomment anything.

## Commands

| Command | What it does |
|---|---|
| `python run.py features` | Build and validate `data/features/features.parquet` |
| `python run.py features --sample-dir PATH` | The same, reading the seven tables from another folder |
| `python run.py compare` | AUC for each feature set, one table added at a time |
| `python run.py check --stage wed` | The Wednesday checkpoint checks |
| `python run.py check` | Everything the Week 2 submission needs |

Run only today's tests while you work, for example:
`python -m pytest tests/test_week2_wed.py -k test_installments` (or `test_credit_card`,
`test_ratio`, `test_input_schema`, `test_feature_schema`, `test_build_feature_table`,
`test_command_line`).

## Wednesday path

1. `aggregate_installments` and `aggregate_credit_card` in `src/features/payments.py`.
2. `add_ratio_features` in `src/features/ratios.py`.
3. `application_schema`, `child_schemas`, and `feature_schema` in `src/validate.py`.
4. `build_feature_table` in `src/make_features.py`, then `python run.py features`.
5. `python run.py check --stage wed`.

If you are short on time, get steps 1, 2 and 4 working with a schema that only checks
`SK_ID_CURR`, then come back and add rules: a pipeline that runs end to end beats four perfect
functions that are not connected.

## Reading a validation failure

A failing run prints one line per problem and exits with code 1:

```
FAILED: input validation failed with 2 problem(s):
  - application_train: missing column(s) AMT_INCOME_TOTAL
  - application_train.AMT_CREDIT: 12 missing value(s), none allowed

No feature table written. Report: reports/validation_<folder>.json
```

Each line names the table, the column, the rule, how many rows broke it, and example values.
That is what "loud" means. A Python traceback is not loud: it is a crash.

## Before Monday (submission)

- `docs/feature_dictionary.md`: copy the template and describe every feature.
- `docs/leakage_table.md`: add the INST_, CC_ and *_RATIO rows from the new template, and answer
  every row.
- `python run.py compare`, then 3-5 sentences in your README on which tables helped and
  whether each gain is bigger than its 95% interval.
- `python run.py test`, `python run.py lint`, and `python run.py check` all pass.
- Tag `week2` and push the tag.
