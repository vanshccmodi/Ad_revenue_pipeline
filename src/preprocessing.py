"""
preprocessing.py
================
Cleans, scales, and splits the aggregated daily time-series.

Responsibilities
----------------
* Remove outliers (IQR-based capping on target).
* Scale features with MinMaxScaler (fit on train only -> no leakage).
* Forward-chaining train / val / test split.
* Persist scaler artifacts for inference de-normalisation.
"""

import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# --------------------------------------------------------------
# Config helper
# --------------------------------------------------------------

def _cfg(config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------
# Outlier capping
# --------------------------------------------------------------

def cap_outliers(series: pd.Series, iqr_mult: float = 3.0) -> pd.Series:
    """
    Cap extreme values beyond iqr_mult × IQR from Q1/Q3.

    Parameters
    ----------
    series   : Numeric pandas Series.
    iqr_mult : Multiplier for IQR fence (default 3 -> gentle cap).

    Returns
    -------
    Capped copy of the series.
    """
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
    capped = series.clip(lower, upper)
    n_capped = (series != capped).sum()
    if n_capped:
        logger.info(f"  Capped {n_capped} outliers in '{series.name}'")
    return capped


# --------------------------------------------------------------
# Train / Val / Test split
# --------------------------------------------------------------

def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Forward-chaining split (no shuffling).

    Parameters
    ----------
    df          : Time-ordered DataFrame.
    train_ratio : Fraction for train (default 0.70).
    val_ratio   : Fraction for validation (default 0.15).

    Returns
    -------
    (train_df, val_df, test_df)
    """
    n = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = df.iloc[:n_train]
    val   = df.iloc[n_train: n_train + n_val]
    test  = df.iloc[n_train + n_val:]

    logger.info(f"  Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# --------------------------------------------------------------
# Scaling
# --------------------------------------------------------------

def fit_scalers(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    scaler_dir: str = "data/processed",
) -> Tuple[MinMaxScaler, MinMaxScaler]:
    """
    Fit separate MinMaxScalers for features and target on **train only**.

    Persists scalers as pickle files in scaler_dir.

    Returns
    -------
    (feature_scaler, target_scaler)
    """
    Path(scaler_dir).mkdir(parents=True, exist_ok=True)

    feat_scaler   = MinMaxScaler()
    target_scaler = MinMaxScaler()

    feat_scaler.fit(train_df[feature_cols].values)
    target_scaler.fit(train_df[[target_col]].values)

    with open(f"{scaler_dir}/feature_scaler.pkl", "wb") as f:
        pickle.dump(feat_scaler, f)
    with open(f"{scaler_dir}/target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)

    logger.info(f"  Scalers saved to: {scaler_dir}/")
    return feat_scaler, target_scaler


def apply_scaling(
    df: pd.DataFrame,
    feat_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Apply pre-fitted scalers; returns a scaled copy."""
    out = df.copy()
    out[feature_cols] = feat_scaler.transform(df[feature_cols].values)
    out[target_col]   = target_scaler.transform(df[[target_col]].values)
    return out


def inverse_scale_target(
    arr: np.ndarray,
    scaler_dir: str = "data/processed",
) -> np.ndarray:
    """
    Load target scaler and invert normalisation.

    Parameters
    ----------
    arr        : 1-D array of scaled predictions.
    scaler_dir : Directory where target_scaler.pkl was saved.

    Returns
    -------
    Original-scale numpy array.
    """
    with open(f"{scaler_dir}/target_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()


# --------------------------------------------------------------
# Master pipeline
# --------------------------------------------------------------

def preprocess(
    daily_df: pd.DataFrame,
    config_path: str = "config.yaml",
) -> dict:
    """
    End-to-end preprocessing.

    Returns a dict with keys:
        train_raw, val_raw, test_raw,
        train_scaled, val_scaled, test_scaled,
        feat_scaler, target_scaler,
        feature_cols, target_col
    """
    cfg        = _cfg(config_path)
    target_col = cfg["data"]["target_col"]
    exog_cols  = [c for c in cfg["data"]["exog_cols"] if c in daily_df.columns]
    proc_dir   = cfg["paths"]["processed_data"]
    scaler_dir = str(Path(proc_dir).parent)

    # 1. Cap outliers on target
    df = daily_df.copy()
    df[target_col] = cap_outliers(df[target_col])

    # 2. Forward-chain split
    train_r, val_r, test_r = train_val_test_split(
        df,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )

    # 3. Fit scalers on train only
    feat_scaler, target_scaler = fit_scalers(train_r, exog_cols, target_col, scaler_dir)

    # 4. Apply scaling
    train_s = apply_scaling(train_r, feat_scaler, target_scaler, exog_cols, target_col)
    val_s   = apply_scaling(val_r,   feat_scaler, target_scaler, exog_cols, target_col)
    test_s  = apply_scaling(test_r,  feat_scaler, target_scaler, exog_cols, target_col)

    # 5. Save processed snapshot
    Path(proc_dir).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(proc_dir)
    logger.info(f"  Processed dataset saved to: {proc_dir}")

    return dict(
        train_raw=train_r, val_raw=val_r, test_raw=test_r,
        train_scaled=train_s, val_scaled=val_s, test_scaled=test_s,
        feat_scaler=feat_scaler, target_scaler=target_scaler,
        feature_cols=exog_cols, target_col=target_col,
    )
