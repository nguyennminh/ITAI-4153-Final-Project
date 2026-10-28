"""Short commands for every project step. Works the same on macOS, Windows, and Linux.

# PROVIDED: this whole script is finished.

Usage (from the repository root):
    python run.py              list the commands
    python run.py setup        create .venv and install requirements.txt
    python run.py sample       build the 5,000-applicant sample
    python run.py summary      write reports/table_summary.json
    python run.py baseline     train the baseline and write reports/metrics.json
    python run.py features     build and validate data/features/features.parquet
    python run.py compare      baseline on growing feature sets -> reports/auc_comparison.json
    python run.py train        train runs from config/models.yaml and log them to MLflow
    python run.py ui           open the MLflow UI at http://127.0.0.1:5050 (Ctrl+C to stop)
    python run.py runs         export every MLflow run to reports/mlflow_runs.csv
    python run.py tune         add Optuna trials to the tuning study (--trials N)
    python run.py threshold    choose the decision threshold from the costs
    python run.py final        calibrate, check, and register the final model
    python run.py test         run the tests
    python run.py lint         check code style
    python run.py check        run the Week 3 checks (--stage mon / --stage wed: one checkpoint)
    python run.py check2       run the Week 2 checks
    python run.py check1       run the Week 1 checks
    python run.py all          sample (if missing), summary, baseline, features, train --all,
                               tune, threshold, final, runs, check

Options for ``sample``: ``--force`` rebuilds an existing sample, and ``--raw-dir PATH`` uses
a full download in another folder instead of data/raw.
Options for ``features``: ``--sample-dir PATH`` reads the tables from another folder.
Options for ``train``: run names from config/models.yaml, ``tree_tuned``, or ``--all``.

Each command runs the plain ``python -m ...`` command printed before it, so you can always
run that command yourself instead.
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
SAMPLE_FILE = ROOT / "data" / "sample" / "application_train.csv"

COMMANDS = {
    "setup": "create .venv and install requirements.txt",
    "sample": "build the 5,000-applicant sample (--force, --raw-dir PATH)",
    "summary": "write reports/table_summary.json",
    "baseline": "train the baseline and write reports/metrics.json",
    "features": "build and validate data/features/features.parquet (--sample-dir PATH)",
    "compare": "baseline on growing feature sets -> reports/auc_comparison.json",
    "train": "train runs from config/models.yaml (NAME ..., tree_tuned, --all)",
    "ui": "open the MLflow UI at http://127.0.0.1:5050 (Ctrl+C to stop)",
    "runs": "export every MLflow run to reports/mlflow_runs.csv",
    "tune": "add Optuna trials to the tuning study (--trials N)",
    "threshold": "choose the decision threshold from the costs",
    "final": "calibrate, check, and register the final model",
    "test": "run the tests",
    "lint": "check code style",
    "check": "run the Week 3 checks (--stage mon, --stage wed)",
    "check2": "run the Week 2 checks",
    "check1": "run the Week 1 checks",
    "all": "sample (if missing) ... train --all, tune, threshold, final, runs, check",
}
MODULES = {
    "sample": ["-m", "scripts.make_sample"],
    "summary": ["-m", "src.summarize"],
    "baseline": ["-m", "src.baseline"],
    "features": ["-m", "src.make_features"],
    "compare": ["-m", "src.compare"],
    "train": ["-m", "src.train"],
    # Port 5050, because macOS uses MLflow's usual port 5000 for AirPlay.
    "ui": ["-m", "mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", "--port", "5050"],
    "runs": ["-m", "src.tracking"],
    "tune": ["-m", "src.tune"],
    "threshold": ["-m", "src.cost"],
    "final": ["-m", "src.final"],
    "test": ["-m", "pytest"],
    "lint": ["-m", "ruff", "check", "."],
    "check": ["-m", "scripts.check_week3"],
    "check2": ["-m", "scripts.check_week2"],
    "check1": ["-m", "scripts.check_week1"],
}


# PROVIDED
def python() -> str:
    """The project's .venv Python if it exists and is not already running, else this Python."""
    in_a_venv = sys.prefix != sys.base_prefix
    return str(VENV_PYTHON) if VENV_PYTHON.exists() and not in_a_venv else sys.executable


# PROVIDED
def run(args: list[str]) -> int:
    """Print a command, run it from the repository root, and return its exit code."""
    print("$ python " + " ".join(args[1:]), flush=True)
    # Hide MLflow's log line about AI coding agents; it is not about your project.
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    return subprocess.call(args, cwd=ROOT, env=env)


# PROVIDED
def setup() -> int:
    """Create .venv (if missing) and install the requirements into it."""
    if not VENV_PYTHON.exists():
        print(f"Creating {VENV.name} ...", flush=True)
        venv.create(VENV, with_pip=True)
    code = run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])
    code = code or run([str(VENV_PYTHON), "-m", "pip", "install", "-r", "requirements.txt"])
    if code == 0:
        windows = sys.platform == "win32"
        activate = r".venv\Scripts\activate" if windows else "source .venv/bin/activate"
        print(f"\nDone. Activate the environment with:  {activate}")
    return code


# PROVIDED
def run_all() -> int:
    """Run the whole path so far in order, stopping at the first failure."""
    steps = ([] if SAMPLE_FILE.exists() else ["sample"]) + [
        "summary", "baseline", "features", "train", "tune", "threshold", "final",
        "runs", "check",
    ]
    for step in steps:
        extra = ["--all"] if step == "train" else []
        code = run([python(), *MODULES[step], *extra])
        if code != 0:
            print(f"\n'{step}' failed; stopping.")
            return code
    return 0


# PROVIDED
def usage() -> None:
    """List the commands."""
    print("Usage: python run.py <command> [options]\n")
    for name, text in COMMANDS.items():
        print(f"  {name:<10}{text}")


# PROVIDED
def main() -> None:
    """Dispatch the command given on the command line."""
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        usage()
        sys.exit(0 if len(sys.argv) < 2 else 2)
    command, extra = sys.argv[1], sys.argv[2:]
    if command == "setup":
        sys.exit(setup())
    if command == "all":
        sys.exit(run_all())
    sys.exit(run([python(), *MODULES[command], *extra]))


if __name__ == "__main__":
    main()
