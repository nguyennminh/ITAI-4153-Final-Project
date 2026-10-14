# Credit-Risk Project: Week 1 (Frame it and baseline it)

<!-- PROVIDED -->

Your team is the machine learning team at a consumer lender. Each loan application ends
in a decision: approve, refer for manual review, or reject. This week you set up a
reproducible project, build a fixed working sample of the Kaggle **Home Credit Default
Risk** data, write down the business problem and its costs, and train a logistic
regression baseline that anyone can rebuild with one command.

The task is binary classification. `TARGET` is 1 if the client had payment difficulties
(about 8 percent of applicants) and 0 otherwise. We report **ROC-AUC, PR-AUC, and KS** on
a fixed validation set.

## Repository layout

```
run.py                    short commands for every step                     PROVIDED
config/config.yaml        settings: paths, seed, sample size, table names   PROVIDED
config/split_ids.csv      the fixed train/validation split                  DO NOT EDIT
scripts/make_sample.py    builds the 5,000-applicant sample                 PROVIDED
scripts/check_week1.py    pass/fail checks for the checkpoints              DO NOT EDIT
src/utils.py              config, seeding, memory helpers                   PROVIDED
src/data.py               loading tables and the split                      MIXED
src/summarize.py          table summary report                              TODO
src/metrics.py            ROC-AUC, PR-AUC, KS                               MIXED
src/baseline.py           cleaning, pipeline, model, metrics.json           MOSTLY TODO
notebooks/01_eda.ipynb    guided exploration                                MIXED
docs/*_template.md        problem statement and data card templates         PROVIDED
tests/                    the tests your code must pass                     DO NOT EDIT
reports/                  files your code writes (summaries, metrics)
data/                     raw download and sample (ignored by git)
```

### The three markers

Every function or block is marked:

- `# PROVIDED`: finished and working. Read it, understand it, reuse it.
- `# TODO(student)`: yours to write. The docstring says what it must do; the **Hints** say
  what to watch for. The body raises `NotImplementedError` until you replace it.
- `# DO NOT EDIT`: tests, the check script, and the split file. Changing them counts as
  breaking the project.

## Setup

You need Python 3.11, Git, and a Kaggle account with the
[competition rules accepted](https://www.kaggle.com/c/home-credit-default-risk/rules).

1. To download from the command line, create an API token on Kaggle (Settings > API) and
   save `kaggle.json` in `~/.kaggle/`
   (Windows: `C:\Users\<you>\.kaggle\`). On macOS/Linux run `chmod 600 ~/.kaggle/kaggle.json`.
   **Never put the token inside this repository.**
2. Create the environment and install the requirements:

   ```bash
   python run.py setup                # macOS/Linux: use python3 if python is not found
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```

   By hand instead: `python -m venv .venv`, activate it, then `pip install -r requirements.txt`.

`requirements.txt` lists the packages the project needs, with minimum versions:

| Package | Used for |
|---|---|
| pandas | Reading and summarizing the tables |
| numpy | Numbers and the seeded random sampling |
| scikit-learn | The pipeline, logistic regression, ROC-AUC and PR-AUC |
| pyyaml | Reading `config/config.yaml` |
| pytest | Running the tests (`python run.py test`) |
| ruff | Checking code style (`python run.py lint`) |
| jupyter | Running the EDA notebook |
| matplotlib | Plots in the notebook |
| kaggle | Downloading the data from the command line |

If you need another package, add it to `requirements.txt` so your teammates get it too.

## Getting the data

You download the data yourself, with your own Kaggle account. The data may not be shared, so
never send it to anyone, upload it, or commit it. `data/` is already in `.gitignore`.

1. **Accept the rules.** Log in to Kaggle and accept the
   [competition rules](https://www.kaggle.com/c/home-credit-default-risk/rules). If you skip
   this, the download fails with a 403 error. Kaggle may ask new accounts to verify a phone number.
2. **Download.** Choose one:
   - *Command line* (installed by `python run.py setup`): with your token in place (Setup, step 1), run
     ```bash
     kaggle competitions download -c home-credit-default-risk -p data/raw
     ```
   - *Browser*: on the competition's Data tab, click **Download All**, then move
     `home-credit-default-risk.zip` into `data/raw/`.
3. **Unzip** into `data/raw/`. This command works on every operating system:
   ```bash
   python -m zipfile -e data/raw/home-credit-default-risk.zip data/raw
   ```
4. **Check.** `data/raw/` should now hold `application_train.csv`, `bureau.csv`,
   `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`,
   `installments_payments.csv`, and `credit_card_balance.csv` (plus three files this project does
   not use). You can delete the zip afterwards.

The download is large (about 0.7 GB zipped, 2.5 GB unzipped), so check your free disk space and
start it early. Work on the problem statement while it runs.

## Running each step

Run everything from the repository root with `run.py`. Type `python run.py` alone to list the
commands. Each command prints the plain `python -m ...` command it runs, so you can always run
that yourself instead.

| Command | Plain equivalent | What it does |
|---|---|---|
| `python run.py setup` | `python -m venv .venv`, then `pip install -r requirements.txt` | Create the environment |
| `python run.py sample` | `python -m scripts.make_sample` | Build `data/sample/` (5,000 applicants, all 7 tables) |
| `python run.py summary` | `python -m src.summarize` | Write `reports/table_summary.json` |
| `python run.py baseline` | `python -m src.baseline` | Train, evaluate, write `reports/metrics.json` |
| `python run.py test` | `python -m pytest` | Run the tests |
| `python run.py lint` | `python -m ruff check .` | Style check (line length 100) |
| `python run.py check` | `python -m scripts.check_week1` | Pass/fail table for the checkpoints |
| `python run.py all` | the four commands in order | Sample (if missing), summary, baseline, check |

Useful options:

- Rebuild the sample: `python run.py sample --force`.
- Already have the full download in another folder (for example in a second clone)?
  `python run.py sample --raw-dir ../other-clone/data/raw` avoids downloading again.

`python run.py sample` reads the large tables once, in chunks. It takes from under a minute to a
few minutes, depending on your laptop. Do not interrupt it. It prints the row count of each
sampled table, the default rate, and a hash that proves your sample is the same as everyone else's.

## Week 1 path

1. Get the data (see above), then `python run.py sample`. Check the printed row counts and
   default rate (about 8%).
2. Write `summarize_table` and `summarize_all` in `src/summarize.py`, then `python run.py summary`.
3. Copy the templates and start the problem statement: copy `docs/problem_template.md` to
   `docs/problem.md` and `docs/data_card_template.md` to `docs/data_card.md`.
4. Work through `notebooks/01_eda.ipynb` (start it with `jupyter notebook` or `jupyter lab`).
5. Write `load_split` (`src/data.py`), `ks_statistic` and `evaluate` (`src/metrics.py`), and the
   TODOs in `src/baseline.py`. Then `python run.py baseline`.
6. `python run.py test`, `python run.py lint`, and `python run.py check` until everything passes.

Run one test while you work on it, for example `python -m pytest tests/test_week1.py -k test_ks`.

## Your team repository and what to submit

Your team keeps one GitHub repository and builds on it every week. **Only one member** sets it up
from the zip; everyone else clones it. (If each member unzipped their own copy, your histories
would not connect.)

**One member, once:**

1. Unzip the starter and open a terminal in the unzipped folder (the one with this README).
2. On GitHub, create a new **empty** repository: do not add a README, .gitignore, or license.
3. Run, using the URL GitHub shows you:

   ```bash
   git init
   git add .
   git commit -m "Week 1 starter"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```

4. On GitHub, add your teammates under **Settings > Collaborators**, and send the repo link to
   your instructor.

**Every other member:** accept the invitation, then `git clone <repo url>`. Do not use the zip.

**Everyone:** run `python run.py setup`, get the data into `data/raw/`, and commit your own work
from your own account. `python run.py check` reports how many people have committed.

**Commit:** code in `src/`, the notebook with your answers, `docs/problem.md`, `docs/data_card.md`,
and the files your code writes in `reports/`. **Never commit:** `data/`, `kaggle.json`, `.venv/`.
The `.gitignore` already blocks those; do not change it.

**Week 1 submission** (start of next week's first class): your repo link, with the final state tagged:

```bash
git tag week1
git push origin week1
```

The tagged repo must contain:

- `reports/table_summary.json` and `reports/metrics.json`, produced by your code
- `notebooks/01_eda.ipynb` with every question answered and your findings written
- `docs/problem.md` with the cost numbers from the task card
- `docs/data_card.md` (about one page)
- passing `python run.py test` and `python run.py check`

## Rules

**Do**

- Read the tests first, so you know what "done" looks like.
- Keep the function names and signatures you were given.
- Run `python run.py check` before every checkpoint, and fix failures instead of working around them.
- Keep reusable logic in `src/`, not only in the notebook.
- Read paths and settings from `config/config.yaml`.
- Write results to files (`reports/`), not only to the screen.
- Commit often with clear messages. Every team member commits.

**Do not**

- Edit `DO NOT EDIT` files, delete tests, or loosen a test to make it pass.
- Commit data, `kaggle.json`, passwords, or keys.
- Hard-code absolute paths from your own computer.
- Use `TARGET` as a feature.
- Fit anything (imputer, scaler, encoder, model) on the validation set.
- Paste code you cannot explain. Any member can be asked about any line.

## Troubleshooting

| Problem | Fix |
|---|---|
| Kaggle download fails with 403 | Accept the [competition rules](https://www.kaggle.com/c/home-credit-default-risk/rules) while logged in, then retry. |
| Kaggle says it cannot find your credentials (401) | Put `kaggle.json` in `~/.kaggle/` (see Setup), or create a new token. |
| `python: command not found` (macOS/Linux) | Use `python3` instead, or activate `.venv` first. |
| "data/sample already exists" | Add `--force` to rebuild it. |
| Sample hash mismatch in `python run.py check` | Restore `scripts/make_sample.py` and `config/config.yaml` from the template, then `python run.py sample --force`. |
| Tests fail with `NotImplementedError` | Expected until you write that TODO. |
