"""Promote the challenger, or roll back to an earlier champion.

# PROVIDED: this whole module is finished.

    python run.py promote               the challenger becomes the champion
    python run.py rollback --version 1  version 1 becomes the champion again

Both do the same three things:

1. move the MLflow alias ``staging`` (the champion: what ``export`` ships) to that version,
2. write reports/champion.json, the record of the new champion (keep the old records in
   reports/retrain/versions/), with a history line saying who became champion and when,
3. run ``export``: models/ now holds the new champion's bundle.

The live service changes only when you point it at the new bundle in your bucket
(docs/cloud_shell_week7.md): promotion and rollback are the same move in opposite
directions, which is why a rollback takes one minute.

``promote`` refuses when decision.json says "keep" unless you pass ``--override "reason"``:
overruling your own rule is allowed, but it is written down.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.retrain import CHALLENGER, CHAMPION, OUT, champion_record, load_retrain_config

VERSIONS = OUT / "versions"
DECISION = OUT / "decision.json"


# PROVIDED
def save_version(root: Path, record: dict[str, Any]) -> None:
    folder = root / VERSIONS
    folder.mkdir(parents=True, exist_ok=True)
    version = str(record["registered"]["version"])
    (folder / f"v{version}.json").write_text(json.dumps(record, indent=2, default=float) + "\n")


# PROVIDED
def version_record(config: dict[str, Any], version: str) -> dict[str, Any]:
    """The saved record of ``version`` (the Week 3 model, a challenger, or a past champion)."""
    root = config["root"]
    candidates = [root / VERSIONS / f"v{version}.json", root / CHALLENGER,
                  config["week3"]["outputs"]["final_model"]]
    for path in candidates:
        if path.exists():
            record = json.loads(path.read_text())
            if str(record["registered"]["version"]) == str(version):
                return record
    raise SystemExit(f"no record of version {version} in {VERSIONS}/, {CHALLENGER} or "
                     "reports/final_model.json")


# PROVIDED
def make_champion(config: dict[str, Any], record: dict[str, Any], action: str,
                  note: str = "") -> dict[str, Any]:
    """Move the alias, write champion.json, and export the bundle."""
    from mlflow import MlflowClient

    from src.export_model import main as export_main
    from src.tracking import setup_mlflow

    root = config["root"]
    registry = config["week3"]["final"]["registry"]
    name, alias = registry["name"], registry["alias"]
    old = champion_record(config)
    save_version(root, old)
    save_version(root, record)

    setup_mlflow(config)
    version = str(record["registered"]["version"])
    MlflowClient().set_registered_model_alias(name, alias, version)
    history = old.get("history", []) + [{
        "action": action, "version": version, "previous": str(old["registered"]["version"]),
        "at": datetime.now(UTC).isoformat(timespec="seconds"), "note": note}]
    champion = {**record, "registered": {"name": name, "version": version, "alias": alias,
                                         "uri": f"models:/{name}@{alias}"},
                "history": history}
    (root / CHAMPION).write_text(json.dumps(champion, indent=2, default=float) + "\n")
    export_main()
    return champion


# PROVIDED
def promote(config: dict[str, Any], override: str | None) -> dict[str, Any]:
    root = config["root"]
    if not (root / CHALLENGER).exists():
        raise SystemExit("no challenger: python run.py retrain first")
    if not (root / DECISION).exists():
        raise SystemExit("no decision yet: python run.py champion first")
    decision = json.loads((root / DECISION).read_text())
    if decision["decision"] != "promote" and not override:
        raise SystemExit("decision.json says KEEP the champion. To overrule your own rule, "
                         'python run.py promote --override "the reason", and explain it in '
                         "docs/decision_record.md.")
    challenger = json.loads((root / CHALLENGER).read_text())
    note = f"override: {override}" if override else "decision: promote"
    return make_champion(config, challenger, "promote", note)


# PROVIDED
def rollback(config: dict[str, Any], version: str) -> dict[str, Any]:
    return make_champion(config, version_record(config, version), "rollback",
                         f"back to version {version}")


# PROVIDED
def main() -> None:
    parser = argparse.ArgumentParser(description="Promote the challenger or roll back.")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("promote")
    p.add_argument("--override", default=None, help="promote although the decision is keep")
    r = sub.add_parser("rollback")
    r.add_argument("--version", required=True, help="the version to make champion again")
    args = parser.parse_args()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    config = load_retrain_config()
    if args.action == "promote":
        champion = promote(config, args.override)
    else:
        champion = rollback(config, args.version)
    version = champion["registered"]["version"]
    print(f"\nChampion is now version {version} ({champion['registered']['uri']}). "
          f"models/ holds its bundle.")
    print(f"Next: upload the three models/ files to your bucket folder models/v{version}/ and "
          f"point the service at it (docs/cloud_shell_week7.md).")


if __name__ == "__main__":
    main()
