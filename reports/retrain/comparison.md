# Champion or challenger

Champion version 1 (threshold 0.080), challenger version 2 (threshold 0.095). Each model at its own threshold. Intervals: 95% paired bootstrap of challenger minus champion. Written by python run.py champion.

## The validation slice (1000 applicants, 81 defaults)

| | AUC | Cost per applicant | Approval rate | Bad rate among approved |
|---|---|---|---|---|
| champion | 0.747 | 0.533 | 74.9% | 4.4% |
| challenger | 0.751 | 0.509 | 64.1% | 3.3% |
| difference [95% interval] | +0.004 [-0.024, +0.034] | -0.024 [-0.114, +0.068] | | |

## The recent slice (564 applicants, 72 defaults)

| | AUC | Cost per applicant | Approval rate | Bad rate among approved |
|---|---|---|---|---|
| champion | 0.748 | 0.722 | 71.6% | 7.2% |
| challenger | 0.735 | 0.681 | 56.2% | 6.0% |
| difference [95% interval] | -0.012 [-0.042, +0.019] | -0.041 [-0.168, +0.066] | | |

## Quality gate on the challenger

FAILED: failed model tests 2 is above the limit 0
(model tests: {'passed': 1, 'failed': 2, 'xfail': 1, 'skipped': 0})

## Decision: KEEP

- gate: failed model tests 2 is above the limit 0
