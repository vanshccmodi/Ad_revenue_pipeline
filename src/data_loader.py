"""
data_loader.py
==============
Loads and validates the raw ads-performance CSV dataset.

Responsibilities
----------------
* Read from config-specified path (or a custom path).
* Parse date column and sort chronologically.
* Perform basic data-quality checks.
* Aggregate multi-row same-day data into daily totals.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------

def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------
# Public API
# --------------------------------------------------------------

def load_raw_data(data_path: str | None = None, config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Load the raw ads-performance CSV.

    Parameters
    ----------
    data_path   : Explicit CSV path; falls back to config if None.
    config_path : Path to config.yaml.

    Returns
    -------
    pd.DataFrame with parsed `date` index, sorted ascending.
    """
    cfg = _load_config(config_path)
    path = Path(data_path or cfg["paths"]["raw_data"])

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at '{path.resolve()}'")

    logger.info(f"Loading raw data from: {path.resolve()}")
    df = pd.read_csv(path, parse_dates=[cfg["data"]["date_col"]])

    date_col = cfg["data"]["date_col"]
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"  Loaded {len(df):,} rows | Date range: {df[date_col].min().date()} -> {df[date_col].max().date()}")
    _validate(df, cfg)
    return df


def _validate(df: pd.DataFrame, cfg: dict) -> None:
    """Basic quality gate."""
    required = [cfg["data"]["date_col"], cfg["data"]["target_col"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    null_counts = df[required].isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values detected:\n{null_counts[null_counts > 0]}")


def aggregate_daily(df: pd.DataFrame, config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Aggregate per-platform/campaign rows into a **single daily
    observation** by summing numeric columns.

    Parameters
    ----------
    df          : Raw multi-platform dataframe.
    config_path : Path to config.yaml.

    Returns
    -------
    pd.DataFrame indexed by date (daily frequency).
    """
    cfg = _load_config(config_path)
    date_col   = cfg["data"]["date_col"]
    target_col = cfg["data"]["target_col"]
    exog_cols  = cfg["data"]["exog_cols"]

    keep = [date_col] + exog_cols + [target_col]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Sum additive cols; CTR, CPC, CPA, ROAS -> weighted or mean
    agg_dict = {col: "sum" for col in exog_cols + [target_col]}
    # Rates should be averaged
    rate_cols = {"CTR", "CPC", "CPA", "ROAS"}
    for col in rate_cols:
        if col in agg_dict:
            agg_dict[col] = "mean"

    daily = df.groupby(date_col).agg(agg_dict).reset_index()
    daily.set_index(date_col, inplace=True)
    daily.sort_index(inplace=True)

    # Fill missing calendar dates with forward-fill (business continuity)
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range).ffill()
    daily.index.name = date_col

    logger.info(f"  Daily aggregation -> {len(daily):,} rows")
    return daily
