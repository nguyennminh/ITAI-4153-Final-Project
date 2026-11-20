# Fairness tables

Model version 1, threshold 0.080, 1,000 validation applicants. Written by python run.py fairness. Brackets: 95% Wilson interval.

## By gender

| Group | n | Defaulters | Approval rate | Default rate | Mean score | TPR (risk caught) | FPR (good payers declined) | Note |
|---|---|---|---|---|---|---|---|---|
| F | 669 | 38 | 79.4% [76%, 82%] | 5.7% | 0.072 | 55.3% [40%, 70%] | 18.5% [16%, 22%] |  |
| M | 331 | 43 | 65.9% [61%, 71%] | 13.0% | 0.099 | 62.8% [48%, 76%] | 29.9% [25%, 35%] |  |

Largest gaps: approval 0.1351, TPR 0.0753, FPR 0.1132; lowest / highest approval rate: 0.8298.

## By age_band

| Group | n | Defaulters | Approval rate | Default rate | Mean score | TPR (risk caught) | FPR (good payers declined) | Note |
|---|---|---|---|---|---|---|---|---|
| 18-29 | 143 | 16 | 58.7% [51%, 66%] | 11.2% | 0.123 | 62.5% [39%, 82%] | 38.6% [31%, 47%] | fewer than 30 defaulters: TPR unreliable |
| 30-39 | 282 | 28 | 70.6% [65%, 76%] | 9.9% | 0.092 | 64.3% [46%, 79%] | 25.6% [21%, 31%] | fewer than 30 defaulters: TPR unreliable |
| 40-49 | 235 | 18 | 75.7% [70%, 81%] | 7.7% | 0.074 | 61.1% [39%, 80%] | 21.2% [16%, 27%] | fewer than 30 defaulters: TPR unreliable |
| 50-59 | 216 | 13 | 80.6% [75%, 85%] | 6.0% | 0.066 | 53.8% [29%, 77%] | 17.2% [13%, 23%] | fewer than 30 defaulters: TPR unreliable |
| 60+ | 124 | 6 | 91.9% [86%, 96%] | 4.8% | 0.046 | 33.3% [10%, 70%] | 6.8% [3%, 13%] | fewer than 30 defaulters: TPR unreliable |

Largest gaps: approval 0.3319, TPR 0.3095, FPR 0.318; lowest / highest approval rate: 0.6389.

## Proxy check

Predicting CODE_GENDER (M vs F) from the other 163 features: AUC 0.863 (0.5 = they carry no information about gender). Strongest single features: AMT_INCOME_TOTAL (0.644), CREDIT_INCOME_RATIO (0.599), DAYS_BIRTH (0.593), EXT_SOURCE_1 (0.585), BUREAU_AMT_CREDIT_SUM_MEAN (0.584).
