<!-- PROVIDED: copy this file to docs/fairness_analysis.md and fill in every placeholder.
     Use the numbers in reports/fairness/fairness.md (python run.py fairness). -->

# Fairness analysis (Week 6)

Team: TODO

## 1. What we compared, and on which data

TODO: The groups (gender, age bands), the data (which applicants, how many, why not the
training set), and the decision rule (your threshold). Which of the group attributes are
model features?

## 2. The table

TODO: Paste or summarize the two tables from reports/fairness/fairness.md (approval rate,
default rate, mean score, TPR, FPR, with their intervals), and include
reports/fairness/fairness.png.

## 3. The gaps

TODO: For gender and for age: which group is approved less often? Is that because its
applicants default more often (compare default_rate), or does the model treat equally risky
applicants differently (compare FPR: good payers who are declined)? Is the model calibrated
within each group (mean score close to default rate)?

## 4. How sure are we?

TODO: Which numbers rest on few people? Use the 95% intervals: which gaps are clearly real
(the intervals do not overlap), and which ones could be chance? How many applicants would
you need to tell?

## 5. Proxies

TODO: The proxy check's AUC and strongest features. What does it mean for the idea "just
remove CODE_GENDER (or DAYS_BIRTH) from the model"? Which features carry the information,
and why might they?

## 6. What we recommend

TODO: Would you ship this model as it is? What would you change or monitor (for example a
fairness metric in your alerts), and who should decide? Note any legal or ethical point
(for example: may a lender use age or gender at all, where you live?).
