"""Run the unit, data, and model tests and write the test report you submit.

# PROVIDED

Run it with ``python run.py testreport``. It runs each kind of test separately and writes:

    reports/test_report.json   every test with its outcome (the check script reads this)
    reports/test_report.md     the same as a table, for humans

Outcomes: passed, failed, xfail (a known failure you documented), skipped (for example no
data on this machine).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from src.experiment import load_week3_config
from src.tracking import git_commit

CATEGORIES = {"unit": "tests/unit", "data": "tests/data", "model": "tests/model"}
REPORT_JSON = Path("reports/test_report.json")
REPORT_MD = Path("reports/test_report.md")


# PROVIDED
def outcome(case: ET.Element) -> tuple[str, str]:
    """(outcome, message) of one <testcase> in pytest's JUnit XML."""
    for child in case:
        message = (child.get("message") or "").strip().splitlines()
        text = message[0][:200] if message else ""
        if child.tag in ("failure", "error"):
            return "failed", text
        if child.tag == "skipped":
            return ("xfail" if child.get("type") == "pytest.xfail" else "skipped"), text
    return "passed", ""


# PROVIDED
def run_category(root: Path, folder: str) -> list[dict[str, str]]:
    """Run pytest on one folder and return one record per test."""
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "junit.xml"
        subprocess.run([sys.executable, "-m", "pytest", folder, "-q", "-p", "no:cacheprovider",
                        f"--junitxml={xml_path}"], cwd=root, capture_output=True, text=True)
        if not xml_path.exists():
            return [{"name": folder, "outcome": "failed", "message": "pytest did not run"}]
        cases = ET.parse(xml_path).getroot().iter("testcase")
        return [{"name": case.get("name", "?"), **dict(zip(("outcome", "message"),
                                                            outcome(case), strict=True))}
                for case in cases]


# PROVIDED
def build_report(root: Path) -> dict:
    """Run every category and collect the results."""
    report = {"created": datetime.now(UTC).isoformat(timespec="seconds"),
              "git_commit": git_commit(root), "categories": {}}
    for name, folder in CATEGORIES.items():
        tests = run_category(root, folder)
        report["categories"][name] = {"counts": dict(Counter(t["outcome"] for t in tests)),
                                      "tests": tests}
    return report


# PROVIDED
def to_markdown(report: dict) -> str:
    """The report as Markdown tables."""
    lines = ["# Test report (Week 4)", "",
             f"Commit `{report['git_commit']}`, {report['created']}.", ""]
    for name, category in report["categories"].items():
        counts = ", ".join(f"{n} {k}" for k, n in sorted(category["counts"].items()))
        lines += [f"## {name.capitalize()} tests ({counts})", "",
                  "| Test | Outcome | Message |", "|---|---|---|"]
        lines += [f"| {t['name']} | {t['outcome']} | {t['message'].replace('|', '/')} |"
                  for t in category["tests"]]
        lines.append("")
    return "\n".join(lines)


# PROVIDED
def main() -> None:
    root = load_week3_config()["root"]
    report = build_report(root)
    (root / REPORT_JSON).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_JSON).write_text(json.dumps(report, indent=2) + "\n")
    (root / REPORT_MD).write_text(to_markdown(report))
    for name, category in report["categories"].items():
        counts = ", ".join(f"{n} {k}" for k, n in sorted(category["counts"].items()))
        print(f"{name:<6} {counts}")
    print(f"Wrote {root / REPORT_JSON}\nWrote {root / REPORT_MD}")


if __name__ == "__main__":
    main()
