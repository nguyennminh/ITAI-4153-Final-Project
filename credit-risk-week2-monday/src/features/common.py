"""Helpers shared by every aggregation, and the Week 2 config loader.

# PROVIDED: this whole module is finished. Read it and reuse it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utils import DEFAULT_CONFIG, load_config

# Home Credit stores "no date" as this number of days (about 1,000 years). You met it in
# Week 1 as DAYS_EMPLOYED; previous_application uses it in five DAYS_* columns too.
SENTINEL = 365243
FEATURES_CONFIG = Path("config/features.yaml")


# PROVIDED
def load_week2_config(
    path: str | Path = DEFAULT_CONFIG, features_path: str | Path | None = None
) -> dict[str, Any]:
    """Load config.yaml (as in Week 1) and add the Week 2 settings under ``config["week2"]``.

    Args:
        path: Location of config.yaml.
        features_path: Location of features.yaml. Defaults to features.yaml next to config.yaml.

    Returns:
        The Week 1 config, plus ``config["week2"]`` holding features.yaml. Every entry
        under ``config["week2"]["paths"]`` is an absolute Path, like ``config["paths"]``.
    """
    config = load_config(path)
    features_path = Path(features_path or Path(path).resolve().parent / FEATURES_CONFIG.name)
    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} not found; copy it from the Week 2 overlay")
    with features_path.open() as f:
        week2 = yaml.safe_load(f)
    week2["paths"] = {key: config["root"] / value for key, value in week2["paths"].items()}
    config["week2"] = week2
    return config


# PROVIDED
def require_columns(df: pd.DataFrame, columns: list[str], table: str) -> None:
    """Raise a KeyError that names the table and every missing column.

    Call it at the top of an aggregation, so a renamed or missing input column fails with a
    clear message instead of a confusing error ten lines later.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{table} is missing required column(s) {missing}")


# PROVIDED
def replace_sentinel(
    df: pd.DataFrame, columns: list[str], sentinel: int = SENTINEL
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``sentinel`` replaced by NaN in ``columns``.

    Columns that are not in ``df`` are skipped. The input is not changed.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].mask(out[col] == sentinel)
    return out


# PROVIDED
def count_where(condition: pd.Series, keys: pd.Series) -> pd.Series:
    """Count the rows where ``condition`` is True, per key.

    Example: ``count_where(bureau["CREDIT_ACTIVE"].eq("Active"), bureau["SK_ID_CURR"])``
    gives each applicant's number of active credits (0 when they have none).
    """
    return condition.astype(int).groupby(keys).sum()


# PROVIDED
def share_where(condition: pd.Series, keys: pd.Series) -> pd.Series:
    """The share (0 to 1) of rows where ``condition`` is True, per key.

    Every row counts in the denominator, so a key with 2 matches out of 8 rows gets 0.25.
    """
    return condition.astype(float).groupby(keys).mean()
