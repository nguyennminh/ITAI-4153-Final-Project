"""Fixtures for the data tests: your real feature table.

# PROVIDED

Data tests check the real data, so they need data/features/features.parquet. Where it does
not exist (a fresh clone, or Cloud Build on Wednesday) every data test is SKIPPED, not failed:
the data is never in GitHub.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.experiment import feature_table_path, load_week3_config


# PROVIDED
@pytest.fixture(scope="session")
def features() -> pd.DataFrame:
    """The Week 2 feature table: one row per applicant, as python run.py features wrote it."""
    path = feature_table_path(load_week3_config())
    if not path.exists():
        pytest.skip("no data on this machine (data/features/features.parquet)")
    return pd.read_parquet(path)
