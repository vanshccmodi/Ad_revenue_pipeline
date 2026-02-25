"""
evaluate.py
===========
Shared evaluation utilities for all time-series models.

Responsibilities
----------------
* Compute MAE, RMSE, MAPE, R².
* Generate residual plots and prediction-vs-actual plots.
* Save plots as artifacts.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # headless backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# Metrics
# --------------------------------------------------------------

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    Uses an epsilon floor to avoid division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict:
    """
    Compute regression metrics.

    Parameters
    ----------
    y_true  : Ground-truth values.
    y_pred  : Predicted values.
    prefix  : Optional string prefix for metric keys.

    Returns
    -------
    dict with keys: mae, rmse, mape, r2 (optionally prefixed).
    """
    mae   = float(mean_absolute_error(y_true, y_pred))
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape_ = mape(y_true, y_pred)
    r2    = float(r2_score(y_true, y_pred))

    p = f"{prefix}_" if prefix else ""
    metrics = {
        f"{p}mae":  mae,
        f"{p}rmse": rmse,
        f"{p}mape": mape_,
        f"{p}r2":   r2,
    }
    logger.info(f"  [{prefix or 'metrics'}] MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape_:.2f}% | R²={r2:.4f}")
    return metrics


# --------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------

def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="x", rotation=30)


def plot_predictions(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: str = "artifacts",
) -> str:
    """
    Plot actual vs predicted values and save as PNG.

    Returns
    -------
    Absolute path to the saved figure.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(dates, y_true, label="Actual",    color="#1f77b4", linewidth=1.8)
    ax.plot(dates, y_pred, label="Predicted", color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax.fill_between(dates, y_true, y_pred, alpha=0.15, color="#aec7e8")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    _style_axes(ax, f"{model_name} - Actual vs Predicted Revenue", "Date", "Revenue ($)")
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = str(Path(save_dir) / f"{model_name}_predictions.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Prediction plot saved -> {out_path}")
    return out_path


def plot_residuals(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: str = "artifacts",
) -> str:
    """
    Plot residuals (actual - predicted) over time and save as PNG.

    Returns
    -------
    Absolute path to the saved figure.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Time-series of residuals
    axes[0].plot(dates, residuals, color="#d62728", linewidth=1.2)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    _style_axes(axes[0], f"{model_name} - Residuals over Time", "Date", "Residual")

    # Distribution of residuals
    axes[1].hist(residuals, bins=30, color="#9467bd", edgecolor="white", linewidth=0.5)
    axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
    _style_axes(axes[1], "Residual Distribution", "Residual", "Frequency")

    plt.suptitle(f"{model_name} - Residual Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = str(Path(save_dir) / f"{model_name}_residuals.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Residual plot saved -> {out_path}")
    return out_path
