# ITAI 4153 Final Project — Credit Risk Modeling

This repository contains the consolidated credit risk project.

## Project Overview

The project uses the Home Credit Default Risk dataset to build a reproducible binary-classification pipeline. It creates a fixed applicant sample, summarizes the source tables, trains a logistic-regression baseline, aggregates applicant credit-history tables, engineers ratio features, validates the resulting feature table, and compares model performance across feature sets.

## Repository Structure

```text
.
├── config/
│   ├── config.yaml
│   ├── features.yaml
│   └── split_ids.csv
├── docs/
│   ├── data_card.md
│   ├── data_card_template.md
│   ├── feature_dictionary_template.md
│   ├── leakage_table.md
│   ├── leakage_table_template.md
│   ├── problem.md
│   ├── problem_template.md
│   └── implementation_notes_*.md
├── notebooks/
│   └── 01_eda.ipynb
├── reports/                  # Generated reports; data outputs are not committed
├── scripts/
│   ├── make_sample.py
│   ├── check_week1.py
│   └── check_week2.py
├── src/
│   ├── baseline.py
│   ├── compare.py
│   ├── make_features.py
│   ├── summarize.py
│   ├── validate.py
│   └── features/
├── tests/
├── run.py
├── requirements.txt
└── pyproject.toml
```

The `data/` directory is intentionally not included in version control. Put the downloaded source tables in the expected data location or use the sample-generation command below.

## Setup

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, run `python run.py setup` to create the virtual environment and install dependencies.

## Common Commands

Run these commands from the repository root:

| Command | Purpose |
|---|---|
| `python run.py` | List available commands |
| `python run.py sample` | Build the fixed 5,000-applicant sample |
| `python run.py summary` | Summarize sampled tables and write `reports/table_summary.json` |
| `python run.py baseline` | Train the application-only baseline |
| `python run.py features` | Build and validate the consolidated feature table |
| `python run.py compare` | Compare model performance across feature sets |
| `python run.py test` | Run the test suite |
| `python run.py lint` | Run Ruff |
| `python run.py check1` | Run Week 1 checkpoint checks |
| `python run.py check` | Run the full Week 2 checks |
| `python run.py all` | Run the end-to-end workflow |

For sample-generation options, use `python run.py sample --help`. For feature-table options, use `python run.py features --help`.

## Model Comparison Results

The model comparison evaluates how adding credit-history tables and engineered ratio features affects the logistic-regression model's ability to distinguish applicants who default from those who do not.

### Evaluation Setup

- **Random seed:** 42
- **Validation sample:** 1,000 applicants
- **Defaults in validation sample:** 81
- **Bootstrap iterations:** 1,000
- **Baseline:** Application features only
- **Model:** Logistic regression evaluated across progressively expanded feature sets

### Performance by Feature Set

| Feature Set | Number of Features | ROC-AUC | PR-AUC | KS Statistic | ROC-AUC Change vs. Baseline |
|---|---:|---:|---:|---:|---:|
| Application | 121 | 0.7097 | 0.1693 | 0.3560 | 0.0000 |
| + Bureau | 130 | 0.7054 | 0.1601 | 0.3453 | -0.0043 |
| + Previous Applications | 138 | 0.7132 | 0.1691 | 0.3553 | +0.0035 |
| + Bureau Balance | 143 | 0.7154 | 0.1756 | 0.3673 | +0.0057 |
| + POS_CASH | 149 | 0.7212 | 0.1755 | 0.3766 | +0.0115 |
| + Installments | 154 | 0.7207 | 0.1737 | 0.3676 | +0.0110 |
| + Credit Card | 160 | **0.7262** | 0.1786 | 0.3730 | **+0.0165** |
| + Ratios | 164 | 0.7244 | **0.1843** | **0.3879** | +0.0147 |

### Metric Definitions

- **ROC-AUC:** Measures how well the model ranks applicants who default above applicants who do not. A value of 0.5 indicates random ranking; higher values indicate better discrimination.
- **PR-AUC:** Summarizes precision and recall across classification thresholds. It is especially useful when defaults are relatively rare.
- **KS Statistic:** Measures the maximum separation between the score distributions of defaulting and non-defaulting applicants. Higher values indicate stronger separation.
- **ROC-AUC Change:** The difference between a feature set's ROC-AUC and the application-only baseline.
- **ROC-AUC Confidence Interval:** An estimate of uncertainty in ROC-AUC based on bootstrap resampling. The comparison also calculates a bootstrap confidence interval for the ROC-AUC change.

### Key Findings

1. **Credit-card features produced the highest ROC-AUC.** Adding credit-card history increased ROC-AUC from 0.7097 to 0.7262, an improvement of approximately 0.0165.
2. **Ratio features improved precision-recall performance.** The full model with ratios achieved the highest PR-AUC of 0.1843, compared with 0.1693 for the application-only baseline.
3. **The ratio-enhanced model achieved the highest KS statistic.** Its KS value of 0.3879 indicates stronger separation between defaulting and non-defaulting applicants at the most separated threshold.
4. **Bureau features alone did not improve the baseline.** ROC-AUC decreased from 0.7097 to 0.7054 after adding bureau features. Later feature groups improved performance when included in the expanding feature sets.
5. **The improvements are uncertain.** All bootstrap confidence intervals for ROC-AUC changes from the baseline include zero. Therefore, these results do not establish a statistically clear ROC-AUC improvement at the conventional 95% confidence level.

### ROC-AUC Confidence Intervals

| Feature Set | ROC-AUC | 95% Bootstrap Confidence Interval |
|---|---:|---:|
| Application | 0.7097 | [0.6510, 0.7669] |
| + Bureau | 0.7054 | [0.6480, 0.7603] |
| + Previous Applications | 0.7132 | [0.6525, 0.7696] |
| + Bureau Balance | 0.7154 | [0.6549, 0.7712] |
| + POS_CASH | 0.7212 | [0.6612, 0.7747] |
| + Installments | 0.7207 | [0.6661, 0.7752] |
| + Credit Card | 0.7262 | [0.6713, 0.7834] |
| + Ratios | 0.7244 | [0.6675, 0.7807] |

### ROC-AUC Change Confidence Intervals

The following intervals show the estimated change in ROC-AUC compared with the application-only baseline.

| Feature Set | ROC-AUC Change | 95% Bootstrap Confidence Interval for Change |
|---|---:|---:|
| + Bureau | -0.0043 | [-0.0128, 0.0045] |
| + Previous Applications | +0.0035 | [-0.0110, 0.0178] |
| + Bureau Balance | +0.0057 | [-0.0105, 0.0220] |
| + POS_CASH | +0.0115 | [-0.0076, 0.0306] |
| + Installments | +0.0110 | [-0.0163, 0.0395] |
| + Credit Card | +0.0165 | [-0.0116, 0.0466] |
| + Ratios | +0.0147 | [-0.0148, 0.0463] |

### Interpretation and Limitations

The results suggest that historical credit behavior and application-based ratios provide useful additional predictive signals. Credit-card history gives the strongest observed ROC-AUC, while adding ratio features gives the strongest observed PR-AUC and KS statistic.

However, the validation set contains only 81 defaults. The confidence intervals are consequently fairly wide, and all confidence intervals for ROC-AUC changes include zero. The results should be treated as exploratory rather than conclusive evidence that one feature set is statistically superior.

The feature sets are cumulative. For example, the `+ratios` result includes the previously added credit-history tables as well as the ratio features. These comparisons do not isolate each table's independent contribution.

All credit-history features must be restricted to information available at the loan application date to prevent target leakage.
