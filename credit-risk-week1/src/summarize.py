"""Summarize every sampled table: size, memory before and after shrinking dtypes, missingness.

Run it with ``python run.py summary`` or ``python -m src.summarize``.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from src.data import load_table
from src.utils import DEFAULT_CONFIG, load_config, memory_mb, reduce_memory

# PROVIDED
SUMMARY_FILENAME = "table_summary.json"
SUMMARY_KEYS = ("table", "rows", "cols", "mb_before", "mb_after", "missing_pct")


def summarize_table(df: pd.DataFrame, name: str) -> dict[str, Any]:
    """Describe one table.

    Args:
        df: The table, as loaded by ``src.data.load_table``.
        name: The table's name, for example ``"bureau"``.

    Returns:
        A dict with exactly the keys in ``SUMMARY_KEYS``:
        - table: the name
        - rows, cols: the table's shape
        - mb_before: memory in MB of ``df`` as loaded
        - mb_after: memory in MB after shrinking the dtypes
        - missing_pct: percentage (0 to 100) of all cells that are missing

    Hints:
        - The helpers in src/utils.py already measure memory and shrink dtypes.
        - Do not change ``df``. Check what ``reduce_memory`` returns.
        - A table can have zero rows. Do not divide by zero.
        - Return plain Python numbers so the result can be written as JSON.
    """

    # Get the number of rows and columns.
    rows, cols = df.shape

    # Calculate memory usage before reducing data types.
    mb_before = memory_mb(df)

    # Reduce memory usage without modifying the original DataFrame.
    reduced_df = reduce_memory(df.copy())

    # Calculate memory usage after reducing data types.
    mb_after = memory_mb(reduced_df)

    # Calculate the percentage of cells that are missing.
    total_cells = rows * cols

    if total_cells == 0:
        missing_pct = 0.0
    else:
        missing_pct = (df.isna().sum().sum() / total_cells) * 100

    # Return exactly the required keys and plain Python numbers.
    return {
        "table": name,
        "rows": int(rows),
        "cols": int(cols),
        "mb_before": float(mb_before),
        "mb_after": float(mb_after),
        "missing_pct": float(missing_pct),
    }

    raise NotImplementedError("TODO(student): summarize one table")


# TODO(student)
def summarize_all(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize every table listed in ``config["tables"]`` and save the report.

    Args:
        config: The loaded config.

    Returns:
        A list with one ``summarize_table`` dict per table, in the order of ``config["tables"]``.
        The same list is written as JSON to ``reports/table_summary.json``
        (``config["paths"]["reports_dir"] / SUMMARY_FILENAME``).

    Hints:
        - Load each table with the provided loader; do not hard-code file paths.
        - The reports folder may not exist yet.
        - Load and summarize one table at a time so memory stays low.
    """

    from pathlib import Path

    # Get the report directory from the configuration.
    reports_dir = Path(config["paths"]["reports_dir"])

    # Create the reports directory if it does not exist.
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Summarize each table in the order specified by the configuration.
    summaries = []

    for table_name in config["tables"]:
        # Load one table at a time to reduce memory usage.
        df = load_table(table_name, config)
        summary = summarize_table(df, table_name)
        summaries.append(summary)

        # Release the table before loading the next one.
        del df

    # Save the summaries as a JSON report.
    report_path = reports_dir / SUMMARY_FILENAME

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    return summaries

    raise NotImplementedError("TODO(student): summarize all tables and write the report")


# PROVIDED
def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print the summary as an aligned table."""
    header = (f"{'table':<24}{'rows':>10}{'cols':>6}"
              f"{'MB before':>11}{'MB after':>10}{'missing %':>11}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['table']:<24}{r['rows']:>10,}{r['cols']:>6}{r['mb_before']:>11.1f}"
              f"{r['mb_after']:>10.1f}{r['missing_pct']:>11.1f}")


# PROVIDED
def main() -> None:
    """Command-line entry point: ``python -m src.summarize [--config PATH]``."""
    parser = argparse.ArgumentParser(description="Summarize the sampled tables.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = summarize_all(config)
    print_summary(rows)
    print(f"\nWrote {config['paths']['reports_dir'] / SUMMARY_FILENAME}")


if __name__ == "__main__":
    main()
