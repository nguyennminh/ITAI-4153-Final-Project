"""Fixtures for the model tests: the model under test and its validation rows.

# PROVIDED

Model tests need the data and a model, so where the data does not exist (a fresh clone, or
Cloud Build on Wednesday) every model test is SKIPPED, not failed.
"""

from __future__ import annotations

import pytest

from src.experiment import load_week3_config
from src.model_io import ProductionModel, load_production_model


# PROVIDED
@pytest.fixture(scope="session")
def production() -> ProductionModel:
    """The registered Week 3 model (or the final.yaml model if none is registered yet)."""
    try:
        return load_production_model(load_week3_config())
    except FileNotFoundError as err:
        pytest.skip(f"no data on this machine ({err})")
