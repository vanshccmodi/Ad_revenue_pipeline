"""
arima_model.py
==============
ARIMA time-series model with MLflow parent + child run logging.

The model is trained on the **unscaled** target series because ARIMA
internally handles stationarity via differencing.

MLflow Run Structure
--------------------
Parent: "ARIMA_main_experiment"
  └- Child: "ARIMA_p{p}_d{d}_q{q}"
"""

import logging
import pickle
import time
from itertools import product
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.statsmodels
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.arima.model import ARIMA

from src.evaluate import compute_metrics, plot_predictions, plot_residuals

logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# Config helper
# --------------------------------------------------------------

def _cfg(config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------
# Single ARIMA fit + MLflow child run
# --------------------------------------------------------------

def _run_arima(
    train_y: pd.Series,
    test_y: pd.Series,
    p: int, d: int, q: int,
    trend: str,
    artifact_dir: str,
    train_size: int,
    test_size: int,
    context: dict,
) -> dict:
    """
    Fit one ARIMA(p,d,q) model and log to MLflow as a child run.

    Returns
    -------
    dict with keys: run_id, metrics, model, predictions
    """
    run_name = f"ARIMA_p{p}_d{d}_q{q}"
    logger.info(f"  Fitting {run_name} …")

    with mlflow.start_run(run_name=run_name, nested=True) as child_run:
        # -- Parameters --------------------------------------
        mlflow.log_params({
            "model_type":  "ARIMA",
            "p": p, "d": d, "q": q,
            "train_size": train_size,
            "test_size":  test_size,
        })
        
        # -- Tags ---------------------------------------------
        mlflow.set_tags({
            "dataset_version": context.get("dataset_version", "v1.0"),
            "dataset_hash": context.get("dataset_hash", "unknown"),
            "device": "cpu",
            "features": "none",
            "forecast_horizon": context.get("forecast_horizon", 7)
        })


        # -- Train --------------------------------------------
        t0 = time.perf_counter()
        try:
            model = ARIMA(train_y, order=(p, d, q), trend=trend)
            fitted = model.fit()
        except Exception as exc:
            logger.warning(f"    ARIMA({p},{d},{q}) trend={trend} failed: {exc}")
            mlflow.log_param("error", str(exc))
            return {}
        train_time = time.perf_counter() - t0

        # -- Forecast -----------------------------------------
        fc    = fitted.forecast(steps=len(test_y))
        preds = np.array(fc)

        # -- Metrics ------------------------------------------
        metrics = compute_metrics(test_y.values, preds, prefix="test")
        metrics["training_time_sec"] = round(train_time, 4)
        mlflow.log_metrics(metrics)

        # -- Artifacts ----------------------------------------
        run_dir = Path(artifact_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Model pickle
        model_path = str(run_dir / "arima_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(fitted, f)
        mlflow.log_artifact(model_path, artifact_path="model")

        # Model summary
        summary_path = str(run_dir / "model_summary.txt")
        with open(summary_path, "w") as f:
            f.write(fitted.summary().as_text())
        mlflow.log_artifact(summary_path, artifact_path="model")

        # Plots
        dates = test_y.index
        pred_plot = plot_predictions(dates, test_y.values, preds, run_name, str(run_dir))
        res_plot  = plot_residuals(dates, test_y.values, preds, run_name, str(run_dir))
        mlflow.log_artifact(pred_plot, artifact_path="plots")
        mlflow.log_artifact(res_plot,  artifact_path="plots")

        # Dataset artifact
        if "processed_csv_path" in context:
            mlflow.log_artifact(context["processed_csv_path"], artifact_path="dataset")

        # MLflow Signature & Input Example
        input_example = pd.DataFrame({"revenue": train_y[-5:].values})
        sig = infer_signature(input_example, np.array(preds[:5]))
        mlflow.statsmodels.log_model(
            fitted, 
            artifact_path="mlflow_native_model",
            signature=sig,
            input_example=input_example
        )

        logger.info(f"    {run_name} -> RMSE={metrics['test_rmse']:.4f} | R²={metrics['test_r2']:.4f}")

        return {
            "run_id":      child_run.info.run_id,
            "run_name":    run_name,
            "metrics":     metrics,
            "model":       fitted,
            "predictions": preds,
            "order":       (p, d, q),
        }


# --------------------------------------------------------------
# Parent ARIMA experiment
# --------------------------------------------------------------

def run_arima_experiment(
    data: dict,
    config_path: str = "config.yaml",
    artifact_dir: str = "artifacts/arima",
) -> dict:
    """
    Launch ARIMA hyperparameter grid search under a parent MLflow run.

    Parameters
    ----------
    data        : Output dict from preprocessing.preprocess().
    config_path : Path to config.yaml.
    artifact_dir: Root directory for saved artefacts.

    Returns
    -------
    dict with keys: best_run, all_results
    """
    cfg        = _cfg(config_path)
    target_col = cfg["data"]["target_col"]
    grid       = cfg["arima"]["param_grid"]

    # Unscaled series for ARIMA
    train_y = data["train_raw"][target_col]
    test_y  = data["test_raw"][target_col]

    all_results: list[dict] = []

    with mlflow.start_run(run_name="ARIMA_main_experiment") as parent_run:
        mlflow.set_tag("model_family", "ARIMA")
        mlflow.set_tag("experiment_type", "hyperparameter_search")
        logger.info("ARIMA parent run started …")

        param_combos = list(product(grid["p"], grid["d"], grid["q"], grid.get("trend", ["n"])))
        logger.info(f"  Grid combinations: {len(param_combos)}")

        for p, d, q, trend in param_combos:
            result = _run_arima(
                train_y=train_y, test_y=test_y,
                p=p, d=d, q=q, trend=trend,
                artifact_dir=artifact_dir,
                train_size=len(train_y),
                test_size=len(test_y),
                context=data,
            )
            if result:
                all_results.append(result)

    # -- Select best by lowest RMSE ----------------------------
    if not all_results:
        raise RuntimeError("All ARIMA configurations failed.")

    best = min(all_results, key=lambda r: r["metrics"]["test_rmse"])
    logger.info(
        f"Best ARIMA -> {best['run_name']} | RMSE={best['metrics']['test_rmse']:.4f}"
    )
    return {"best_run": best, "all_results": all_results}
