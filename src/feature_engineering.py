"""
feature_engineering.py
=======================
Generates time-series and lag-based features from the daily DataFrame.

Responsibilities
----------------
* Calendar features (day-of-week, month, quarter, week-of-year).
* Rolling statistics (MA7, MA14, MA30) on target and key exogenous cols.
* Lag features (lag1, lag7, lag14, lag30).
* Percentage-change momentum.
* Return the enriched DataFrame (no in-place mutation).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# Calendar features
# --------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week, month, quarter, week-of-year, is_weekend flags."""
    out = df.copy()
    idx = out.index  # DatetimeIndex

    out["dow"]       = idx.dayofweek           # 0=Mon … 6=Sun
    out["month"]     = idx.month
    out["quarter"]   = idx.quarter
    out["weekofyear"]= idx.isocalendar().week.astype(int).values
    out["is_weekend"]= (idx.dayofweek >= 5).astype(int)

    logger.debug("  Added calendar features")
    return out


# --------------------------------------------------------------
# Rolling statistics
# --------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    col: str,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add rolling mean and std for a given column over multiple windows.

    Parameters
    ----------
    df      : DataFrame (DatetimeIndex, time-ordered).
    col     : Column name to compute statistics on.
    windows : List of rolling window sizes (default [7, 14, 30]).

    Returns
    -------
    DataFrame with new rolling_mean_{w} and rolling_std_{w} columns.
    """
    windows = windows or [7, 14, 30]
    out = df.copy()
    for w in windows:
        out[f"{col}_ma{w}"]  = out[col].rolling(w, min_periods=1).mean()
        out[f"{col}_std{w}"] = out[col].rolling(w, min_periods=1).std().fillna(0)
    logger.debug(f"  Rolling features added for '{col}' | windows={windows}")
    return out


# --------------------------------------------------------------
# Lag features
# --------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    col: str,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add lag (shift) features for a given column.

    Parameters
    ----------
    col  : Column name to compute lags from.
    lags : List of lag periods (default [1, 7, 14, 30]).

    Returns
    -------
    DataFrame with new lag columns; NaN rows filled via bfill.
    """
    lags = lags or [1, 7, 14, 30]
    out = df.copy()
    for lag in lags:
        out[f"{col}_lag{lag}"] = out[col].shift(lag)
    out.bfill(inplace=True)
    logger.debug(f"  Lag features added for '{col}' | lags={lags}")
    return out


# --------------------------------------------------------------
# Momentum
# --------------------------------------------------------------

def add_pct_change(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add day-over-day percentage change for a column."""
    out = df.copy()
    out[f"{col}_pct"] = out[col].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    return out


# --------------------------------------------------------------
# Master pipeline
# --------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    target_col: str = "revenue",
    exog_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Full feature-engineering pipeline.

    Applies calendar features, rolling stats, lags, and momentum
    on both the target and high-signal exogenous columns.

    Parameters
    ----------
    df         : Daily DataFrame with DatetimeIndex.
    target_col : Target column name.
    exog_cols  : Exogenous columns to enrich (only high-signal subset used).

    Returns
    -------
    Feature-enriched DataFrame (may contain additional NaN rows bfill'd).
    """
    logger.info("Starting feature engineering …")
    out = df.copy()

    # Calendar
    out = add_calendar_features(out)

    # Target enrichment
    out = add_rolling_features(out, target_col)
    out = add_lag_features(out, target_col)
    out = add_pct_change(out, target_col)

    # Exogenous enrichment (rolling + lag for key signals)
    key_exog = ["ad_spend", "clicks", "impressions"]
    if exog_cols:
        key_exog = [c for c in key_exog if c in exog_cols]

    for col in key_exog:
        if col in out.columns:
            out = add_rolling_features(out, col, windows=[7, 14])
            out = add_lag_features(out, col, lags=[1, 7])

    # Final fill
    out.ffill(inplace=True)
    out.bfill(inplace=True)

    logger.info(f"  Feature engineering complete -> {out.shape[1]} features")
    return out
