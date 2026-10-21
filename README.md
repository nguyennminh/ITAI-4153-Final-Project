# ITAI 4153 Final Project — Credit Risk Modeling

This repository contains the consolidated credit-risk project. The weekly copies have been merged into one project structure; where files existed in multiple weekly folders, the most advanced version was retained (Wednesday, then Monday, then Week 1), while files unique to earlier work were preserved.

## Project overview

The project uses the Home Credit Default Risk dataset to build a reproducible binary-classification pipeline. It creates a fixed applicant sample, summarizes the source tables, trains a logistic-regression baseline, aggregates applicant credit-history tables, engineers ratio features, validates the resulting feature table, and compares model performance across feature sets.

## Repository structure

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
# .venv\Scripts\activate      # Windows

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, run `python run.py setup` to create the virtual environment and install dependencies.

## Common commands

Run these from the repository root:

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

## Reproducibility and data safety

- The fixed applicant split is stored in `config/split_ids.csv`; do not edit it casually.
- Keep downloaded data and generated feature tables under `data/`, which is ignored by Git.
- Do not commit credentials, Kaggle tokens, or other secrets.
- Generated reports belong in `reports/`; rerun the relevant command to refresh them.
- Feature aggregation must preserve one row per applicant and avoid using information unavailable at application time.

## Notes

The original Monday and Wednesday checkpoint guides are preserved in `docs/implementation_notes_monday.md` and `docs/implementation_notes_wednesday.md` for reference. They are documentation, not separate project copies.
