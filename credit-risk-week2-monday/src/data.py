"""Loading the sampled tables and the fixed train/validation split."""

from __future__ import annotations

from typing import Any

import pandas as pd


# PROVIDED
def load_table(name: str, config: dict[str, Any]) -> pd.DataFrame:
    """Read one sampled table from data/sample/ with pandas' default type inference.

    Args:
        name: A table name from ``config["tables"]``, for example ``"bureau"``.
        config: The loaded config (see ``src.utils.load_config``).

    Returns:
        The table as a DataFrame.
    """
    if name not in config["tables"]:
        raise KeyError(f"Unknown table '{name}'. Known tables: {list(config['tables'])}")
    path = config["paths"]["sample_dir"] / config["tables"][name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python run.py sample` first.")
    return pd.read_csv(path)


# PROVIDED
def load_application(config: dict[str, Any]) -> pd.DataFrame:
    """Return the sampled application_train table (one row per applicant)."""
    return load_table("application_train", config)


# TODO(student)
def load_split(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the sampled application table into train and validation rows.

    Use the fixed split in ``config["paths"]["split_file"]`` (columns SK_ID_CURR and
    ``split``, where ``split`` is ``"train"`` or ``"valid"``). Never create your own split.

    Args:
        config: The loaded config.

    Returns:
        ``(train_df, valid_df)``: application rows, with all their original columns,
        for the train ids and the valid ids. Each has a fresh 0..n-1 index.

    Raises:
        ValueError: if the split file lists an id twice, uses a split value other than
            train/valid, or does not match the applicants in the sample one-to-one.

    Hints:
        - Read the id column so it has the same type as the id column in the application table.
        - Every applicant must land in exactly one of the two frames: no overlap, none missing.
        - Check the split file before you trust it, and fail loudly if it looks wrong.
        - Use ``config["id_col"]`` instead of typing the column name.
    """
    id_col = config["id_col"]
    app = load_application(config)
    # Read the ids with the application table's dtype so the lookups below match exactly
    split = pd.read_csv(config["paths"]["split_file"], dtype={id_col: app[id_col].dtype})

    if split[id_col].duplicated().any():
        dupes = split.loc[split[id_col].duplicated(), id_col].tolist()
        raise ValueError(f"Split file lists ids more than once: {dupes[:5]}")
    bad_values = set(split["split"]) - {"train", "valid"}
    if bad_values:
        raise ValueError(f"Split file has values other than train/valid: {sorted(bad_values)}")
    split_ids, app_ids = set(split[id_col]), set(app[id_col])
    if split_ids != app_ids or len(app_ids) != len(app):
        raise ValueError(
            f"Split file does not match the sample one-to-one: "
            f"{len(app_ids - split_ids)} sample ids missing from the split, "
            f"{len(split_ids - app_ids)} split ids not in the sample"
        )

    train_ids = split.loc[split["split"] == "train", id_col]
    in_train = app[id_col].isin(train_ids)
    return app[in_train].reset_index(drop=True), app[~in_train].reset_index(drop=True)
