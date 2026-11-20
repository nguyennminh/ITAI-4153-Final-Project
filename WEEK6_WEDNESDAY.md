# Week 6: Monitoring and responsible AI (Wednesday)

<!-- PROVIDED -->

On Monday you made the model observable. Today you ask two questions a regulator, a
customer, or your own manager will ask: **why was this applicant declined?** and **does the
model treat groups of people differently?** Then you start the model card: the one page
anyone can read before trusting the model.

Nothing today touches the live service or the cloud: everything runs on your laptop, on the
validation set (applicants the model never trained on, with known outcomes).

## 1. Explanations with SHAP

For one applicant, a **SHAP value** says how much each feature pushed the score up
(towards "reject") or down, compared with the average applicant. They add up: base value +
all contributions = this applicant's score (in log-odds, the model's internal scale).

- **Globally**, the mean absolute SHAP value per feature ranks what the model relies on.
  Compare it with what you expected in Week 2 and with the model tests of Week 4.
- **Locally**, for a declined applicant, the features with the largest positive
  contributions are the reasons. You write `explain_applicant`, which turns them into
  sentences a person can read: "external credit score 2 is 0.19 (typical: 0.56)".

`python run.py explain` explains three declined validation applicants: the riskiest, a
typical decline, and a borderline one just above your threshold. Read them critically: is
each reason something you could say to the applicant? Is any reason about who they are
rather than what they did?

Your model is the gradient boosting pipeline inside a calibrator. SHAP's TreeExplainer reads
the boosting model's trees directly (exact and fast); the calibrator only rescales the score,
so a feature that raises one raises the other.

## 2. Fairness by gender and age

`python run.py fairness` scores the validation set with your threshold and compares groups
by `CODE_GENDER` and by age band (from `DAYS_BIRTH`). You write `group_table`; the rest is
provided. For each group:

| Column | Question it answers |
|---|---|
| `approval_rate` | who gets a loan? |
| `default_rate` | how often did this group really default? (a different base rate can explain a different approval rate) |
| `mean_score` vs `default_rate` | is the model calibrated for this group? |
| `tpr` | of the group's defaulters, how many does the model catch? |
| `fpr` | of the group's **good payers**, how many are wrongly declined? |

**How sure are you?** The validation set has about 1,000 applicants and about 80 defaulters.
Split by age, a group's TPR may rest on 6 people. Each rate comes with a 95% **Wilson
interval**; a wide interval means "we cannot tell yet", not "no difference". The table marks
groups that are too small.

**Proxies.** The command also trains a small model to predict gender from the *other*
features. If it can (AUC well above 0.5), deleting the gender column would not remove gender
from your model: other features carry it.

## 3. The model card

`docs/model_card_template.md`: what the model is for and not for, its data, performance,
explanations, fairness, monitoring, and limitations, in plain language. A draft this week,
finished in Week 7.

## Commands

```
python run.py test tests/test_week6_wed.py   # 6 tests (no data needed)
python run.py explain                        # reports/explain/
python run.py fairness                       # reports/fairness/
python run.py evaluate                       # the zip added code under src/
python run.py check --stage wed              # the Wednesday checkpoint
python run.py check                          # everything the Week 6 submission needs
```

## The files today

| File | Status | What it is |
|---|---|---|
| `src/explain.py` | MIXED | SHAP and the report (provided); **`explain_applicant`** (yours) |
| `src/fairness.py` | MIXED | intervals, gaps, proxy check, report (provided); **`group_table`** (yours) |
| `tests/test_week6_wed.py` | DO NOT EDIT | 6 tests |
| `scripts/check_week6.py` | DO NOT EDIT (replaced) | all Week 6 checks, with `--stage wed` |
| `docs/fairness_analysis_template.md` | PROVIDED | copy to `docs/fairness_analysis.md` |
| `docs/model_card_template.md` | PROVIDED | copy to `docs/model_card.md` |
| `run.py` | PROVIDED (replaced) | adds `explain` and `fairness` |

## Rules this week

- Work on a branch, merge through a pull request; `python run.py evaluate` before you commit.
- `reports/explain/` and `reports/fairness/` hold only summaries (no applicant rows except
  the three explained ids and their reasons): commit them. Never commit `data/` or `models/`.
- Do not edit `scripts/check_week6.py` or the tests.
