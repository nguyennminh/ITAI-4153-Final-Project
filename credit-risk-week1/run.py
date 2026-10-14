"""Short commands for every project step. Works the same on macOS, Windows, and Linux.

# PROVIDED: this whole script is finished.

Usage (from the repository root):
    python run.py              list the commands
    python run.py setup        create .venv and install requirements.txt
    python run.py sample       build the 5,000-applicant sample
    python run.py summary      write reports/table_summary.json
    python run.py baseline     train the baseline and write reports/metrics.json
    python run.py test         run the tests
    python run.py lint         check code style
    python run.py check        run the Week 1 checks
    python run.py all          sample (if missing), summary, baseline, then check

Options for ``sample``: ``--force`` rebuilds an existing sample, and ``--raw-dir PATH`` uses
a full download in another folder instead of data/raw.

Each command runs the plain ``python -m ...`` command printed before it, so you can always
run that command yourself instead.
"""

from __future__ import annotations

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
    "test": "run the tests",
    "lint": "check code style",
    "check": "run the Week 1 checks",
    "all": "sample (if missing), summary, baseline, then check",
}
MODULES = {
    "sample": ["-m", "scripts.make_sample"],
    "summary": ["-m", "src.summarize"],
    "baseline": ["-m", "src.baseline"],
    "test": ["-m", "pytest"],
    "lint": ["-m", "ruff", "check", "."],
    "check": ["-m", "scripts.check_week1"],
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
    return subprocess.call(args, cwd=ROOT)


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
    """Run the Week 1 path in order, stopping at the first failure."""
    steps = ([] if SAMPLE_FILE.exists() else ["sample"]) + ["summary", "baseline", "check"]
    for step in steps:
        code = run([python(), *MODULES[step]])
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
