"""Shared helpers: configuration, seeding, and memory reporting.

# PROVIDED: this whole module is finished. Read it and reuse it.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG = Path("config/config.yaml")
# Text columns with fewer unique values than this share of rows become 'category'.
CATEGORY_MAX_UNIQUE_RATIO = 0.5


# PROVIDED
def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the YAML config and turn every entry under ``paths`` into an absolute Path.

    Paths in the file are relative to the repository root, which is the folder that
    contains the ``config/`` folder. Resolving them here means code works no matter
    which folder you run it from.

    Args:
        path: Location of config.yaml.

    Returns:
        The config as a dict. ``config["paths"]`` values are absolute ``Path`` objects,
        and ``config["root"]`` is the repository root.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        config = yaml.safe_load(f)
    root = path.parent.parent
    config["root"] = root
    config["paths"] = {key: root / value for key, value in config["paths"].items()}
    return config


# PROVIDED
def set_seed(seed: int) -> None:
    """Seed Python's and NumPy's global random generators so runs repeat exactly."""
    random.seed(seed)
    np.random.seed(seed)


# PROVIDED
def memory_mb(df: pd.DataFrame) -> float:
    """Return the memory a DataFrame uses in megabytes, counting the text inside columns."""
    return float(df.memory_usage(deep=True).sum()) / 1024**2


# PROVIDED
def _is_text(series: pd.Series) -> bool:
    """True for object or string columns (pandas 2 and 3 store text differently)."""
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


# PROVIDED
def _shrink_column(series: pd.Series) -> pd.Series:
    """Return a smaller-dtype version of one column, or the column unchanged."""
    if pd.api.types.is_bool_dtype(series):
        return series
    if pd.api.types.is_integer_dtype(series):
        return pd.to_numeric(series, downcast="integer")
    if pd.api.types.is_float_dtype(series):
        return series.astype("float32")
    if _is_text(series) and len(series) > 0:
        if series.nunique(dropna=True) / len(series) < CATEGORY_MAX_UNIQUE_RATIO:
            shrunk = series.astype("category")
            # Keep the change only if it actually saves memory.
            if shrunk.memory_usage(deep=True) < series.memory_usage(deep=True):
                return shrunk
    return series


# PROVIDED
def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with smaller dtypes. The input is not changed.

    - Integers are downcast to the smallest integer type that holds every value.
    - Floats become float32 (about 7 significant digits, enough for these columns).
    - Text columns with few distinct values become 'category'.
    """
    return pd.DataFrame({col: _shrink_column(df[col]) for col in df.columns}, index=df.index)
