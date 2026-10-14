<!-- PROVIDED template. Copy it to docs/data_card.md and fill that copy in:
       cp docs/data_card_template.md docs/data_card.md
     Replace every "TODO(student)" line with your answer. `python run.py check` fails while any remain.
     Aim for about one page. Use numbers from your own reports/ files. -->

# Data card: Home Credit Default Risk (Week 1 sample)

## 1. Source and license terms

*Where does the data come from, and who published it? What do the Kaggle competition
rules allow and forbid (for example sharing or redistributing the data)? How does this
repository respect those rules?*

TODO(student)

## 2. What each table holds

*One line per table: what one row represents and the key column(s) that link it to the
others. Note which table does not carry the applicant id, and how it links instead.*

| Table | One row is... | Links by |
|---|---|---|
| application_train | TODO(student) | |
| bureau | TODO(student) | |
| bureau_balance | TODO(student) | |
| previous_application | TODO(student) | |
| POS_CASH_balance | TODO(student) | |
| installments_payments | TODO(student) | |
| credit_card_balance | TODO(student) | |

## 3. Full size versus sample size

*How many rows does each table have in the full download and in your sample? How much
memory does the sample use before and after shrinking the dtypes
(reports/table_summary.json)? How was the sample chosen, and why sample whole applicants
instead of random rows?*

TODO(student)

## 4. Target definition and class balance

*What does TARGET = 1 mean? What is the default rate in the full data, in your sample,
and in each split? Why does the split keep the rates similar?*

TODO(student)

## 5. Known issues

*Missing values (which columns, how much), odd or placeholder values (look closely at
DAYS_EMPLOYED), units and sign conventions (what does a negative DAYS_BIRTH mean?), and
anything else that surprised you in the EDA.*

TODO(student)

## 6. Sensitive columns

*Which columns describe sensitive characteristics, or could act as a proxy for them?
How do you plan to handle them?*

TODO(student)

## 7. Limitations of a small sample

*The validation set has about 80 defaults. How much could your metrics move by chance?
What could the sample miss that the full data would show (for example rare categories)?*

TODO(student)
