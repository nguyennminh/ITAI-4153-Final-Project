"""Tests for the provided sampling script. These pass on the starter code.

# DO NOT EDIT
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.make_sample import build_sample, sample_hash

APPLICANT_KEYED = ("bureau", "previous_application", "POS_CASH_balance",
                   "installments_payments", "credit_card_balance")


def read_text_csv(path: Path) -> pd.DataFrame:
    """Read a CSV exactly as text, the way the sampler reads it."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def raw(config: dict[str, Any], name: str) -> pd.DataFrame:
    return read_text_csv(config["paths"]["raw_dir"] / config["tables"][name])


def sampled(config: dict[str, Any], name: str) -> pd.DataFrame:
    return read_text_csv(config["paths"]["sample_dir"] / config["tables"][name])


@pytest.fixture
def built(raw_project: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return raw_project, build_sample(raw_project)


def test_exact_sample_size(built):
    config, _ = built
    app = sampled(config, "application_train")
    assert len(app) == config["sample_size"]
    assert app["SK_ID_CURR"].is_unique


def test_default_count_matches_full_rate(built):
    config, _ = built
    full_rate = (raw(config, "application_train")["TARGET"] == "1").mean()
    n_defaults = (sampled(config, "application_train")["TARGET"] == "1").sum()
    assert n_defaults == round(config["sample_size"] * full_rate)


def test_same_hash_on_two_runs(built):
    config, first = built
    second = build_sample(config, force=True)
    assert first["hash"] == second["hash"]
    ids = sampled(config, "application_train")["SK_ID_CURR"].astype(int).to_numpy()
    assert sample_hash(ids) == first["hash"]
    hash_file = config["paths"]["reports_dir"] / "sample_hash.txt"
    assert hash_file.read_text().strip() == first["hash"]


def test_child_rows_belong_to_sampled_applicants(built):
    config, _ = built
    ids = set(sampled(config, "application_train")["SK_ID_CURR"])
    for name in APPLICANT_KEYED:
        sample = sampled(config, name)
        assert set(sample["SK_ID_CURR"]) <= ids, name
        expected = raw(config, name)["SK_ID_CURR"].isin(ids).sum()
        assert len(sample) == expected, f"{name}: some rows of sampled applicants are missing"


def test_bureau_balance_follows_sampled_bureau(built):
    config, _ = built
    bureau_ids = set(sampled(config, "bureau")["SK_ID_BUREAU"])
    bb = sampled(config, "bureau_balance")
    assert len(bb) > 0
    assert set(bb["SK_ID_BUREAU"]) <= bureau_ids
    assert len(bb) == raw(config, "bureau_balance")["SK_ID_BUREAU"].isin(bureau_ids).sum()


def test_row_text_copied_unchanged(built):
    config, _ = built
    for name in config["tables"]:
        sample = sampled(config, name)
        source = raw(config, name)
        assert list(sample.columns) == list(source.columns), name
        key = "SK_ID_BUREAU" if name == "bureau_balance" else "SK_ID_CURR"
        expected = source[source[key].isin(set(sample[key]))]
        pd.testing.assert_frame_equal(sample.reset_index(drop=True),
                                      expected.reset_index(drop=True), obj=name)
    app = sampled(config, "application_train")
    assert {"007"} == set(app["REGION_CODE"])  # not turned into the number 7
    assert set(app["NAME_TYPE_SUITE"]) <= {"Unaccompanied", "Spouse, partner", "", "NA"}


def test_header_only_file_when_nothing_matches(built):
    config, result = built
    cc = sampled(config, "credit_card_balance")
    assert len(cc) == 0
    assert list(cc.columns) == list(raw(config, "credit_card_balance").columns)
    assert result["row_counts"]["credit_card_balance"] == 0


def test_row_counts_report(built):
    config, result = built
    path = config["paths"]["reports_dir"] / "sample_row_counts.json"
    counts = json.loads(path.read_text())
    assert list(counts) == list(config["tables"])
    for name, n in counts.items():
        assert n == len(sampled(config, name)) == result["row_counts"][name]


def test_no_overwrite_without_force(built):
    config, _ = built
    with pytest.raises(FileExistsError):
        build_sample(config)
    build_sample(config, force=True)  # allowed


def test_missing_raw_file_message(raw_project):
    (raw_project["paths"]["raw_dir"] / "bureau.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Getting the data"):
        build_sample(raw_project)
