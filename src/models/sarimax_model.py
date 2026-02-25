"""
sarimax_model.py
================
SARIMAX time-series model with exogenous ad features,
MLflow parent + child run logging.

MLflow Run Structure
--------------------
Parent: "SARIMAX_main_experiment"
  └- Child: "SARIMAX_p{p}_d{d}_q{q}_P{P}_D{D}_Q{Q}_s{s}"
"""

import logging
import pickle
import time
from itertools import product
from pathlib import Path

import mlflow
import mlflow.statsmodels
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.evaluate import compute_metrics, plot_predictions, plot_residuals

logger = logging.getLogger(__name__)


def _cfg(config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------
# Single SARIMAX fit + MLflow child run
# --------------------------------------------------------------

def _run_sarimax(
    train_y:    pd.Series,
    test_y:     pd.Series,
    train_exog: pd.DataFrame,
    test_exog:  pd.DataFrame,
    p: int, d: int, q: int,
    seasonal_order: tuple,
    trend: str,
    artifact_dir: str,
    train_size: int,
    test_size: int,
    context: dict,
) -> dict:
    """Fit one SARIMAX model and log to MLflow as a child run."""
    P, D, Q, s = seasonal_order
    run_name = f"SARIMAX_p{p}_d{d}_q{q}_P{P}_D{D}_Q{Q}_s{s}"
    logger.info(f"  Fitting {run_name} …")

    with mlflow.start_run(run_name=run_name, nested=True) as child_run:
        mlflow.log_params({
            "model_type": "SARIMAX",
            "p": p, "d": d, "q": q,
            "seasonal_P": P, "seasonal_D": D, "seasonal_Q": Q, "seasonal_s": s,
            "n_exog":     train_exog.shape[1],
            "exog_cols":  ",".join(train_exog.columns.tolist()),
            "train_size": train_size,
            "test_size":  test_size,
            "trend":      trend,
        })

        mlflow.set_tags({
            "dataset_version": context.get("dataset_version", "v1.0"),
            "dataset_hash": context.get("dataset_hash", "unknown"),
            "device": "cpu",
            "features": ",".join(train_exog.columns.tolist()),
            "forecast_horizon": context.get("forecast_horizon", len(test_y))
        })

        t0 = time.perf_counter()
        try:
            model = SARIMAX(
                train_y,
                exog=train_exog,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
        except Exception as exc:
            logger.warning(f"    {run_name} failed: {exc}")
            mlflow.log_param("error", str(exc))
            return {}
        train_time = time.perf_counter() - t0

        fc    = fitted.forecast(steps=len(test_y), exog=test_exog)
        preds = np.array(fc)

        metrics = compute_metrics(test_y.values, preds, prefix="test")
        metrics["training_time_sec"] = round(train_time, 4)
        mlflow.log_metrics(metrics)

        # Artifacts
        run_dir = Path(artifact_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        model_path = str(run_dir / "sarimax_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(fitted, f)
        mlflow.log_artifact(model_path, artifact_path="model")

        summary_path = str(run_dir / "model_summary.txt")
        with open(summary_path, "w") as f:
            f.write(fitted.summary().as_text())
        mlflow.log_artifact(summary_path, artifact_path="model")

        dates = test_y.index
        pred_plot = plot_predictions(dates, test_y.values, preds, run_name, str(run_dir))
        res_plot  = plot_residuals(dates, test_y.values, preds, run_name, str(run_dir))
        mlflow.log_artifact(pred_plot, artifact_path="plots")
        mlflow.log_artifact(res_plot,  artifact_path="plots")
        
        # Dataset artifact
        if "processed_csv_path" in context:
            mlflow.log_artifact(context["processed_csv_path"], artifact_path="dataset")

        # MLflow Signature & Input Example
        # SARIMAX takes both `y` and `exog` as input. `mlflow.statsmodels` does not natively 
        # dictate the shape as simply. We can infer on the exog + target.
        sig_input = test_exog[-5:].copy()
        sig_input["target"] = train_y[-5:].values
        sig = infer_signature(sig_input, np.array(preds[:5]))
        
        mlflow.statsmodels.log_model(
            fitted, 
            artifact_path="mlflow_native_model",
            signature=sig,
            input_example=sig_input
        )

        # Feature importance proxy: exog coefficient magnitudes
        try:
            coef_path = str(run_dir / "exog_coefficients.txt")
            params = fitted.params
            exog_params = {k: v for k, v in params.items() if k in train_exog.columns}
            with open(coef_path, "w") as f:
                f.write("Exogenous Variable Coefficients\n")
                f.write("=" * 40 + "\n")
                for k, v in sorted(exog_params.items(), key=lambda x: abs(x[1]), reverse=True):
                    f.write(f"{k:<25}: {v:+.6f}\n")
            mlflow.log_artifact(coef_path, artifact_path="feature_importance")
        except Exception:
            pass

        logger.info(f"    {run_name} -> RMSE={metrics['test_rmse']:.4f} | R²={metrics['test_r2']:.4f}")

        return {
            "run_id":      child_run.info.run_id,
            "run_name":    run_name,
            "metrics":     metrics,
            "model":       fitted,
            "predictions": preds,
            "order":       (p, d, q),
            "seasonal":    seasonal_order,
        }


# --------------------------------------------------------------
# Parent SARIMAX experiment
# --------------------------------------------------------------

def run_sarimax_experiment(
    data: dict,
    config_path: str = "config.yaml",
    artifact_dir: str = "artifacts/sarimax",
) -> dict:
    """
    SARIMAX hyperparameter grid search under a parent MLflow run.

    Uses raw (unscaled) target + unscaled exogenous features because
    SARIMAX handles stationarity internally.
    """
    cfg        = _cfg(config_path)
    target_col = cfg["data"]["target_col"]
    exog_cols  = [c for c in cfg["data"]["exog_cols"] if c in data["train_raw"].columns]
    grid       = cfg["sarimax"]["param_grid"]

    train_y    = data["train_raw"][target_col]
    test_y     = data["test_raw"][target_col]
    train_exog = data["train_raw"][exog_cols]
    test_exog  = data["test_raw"][exog_cols]

    all_results: list[dict] = []

    with mlflow.start_run(run_name="SARIMAX_main_experiment") as parent_run:
        mlflow.set_tag("model_family", "SARIMAX")
        mlflow.set_tag("experiment_type", "hyperparameter_search")
        logger.info("SARIMAX parent run started …")

        combos = list(product(
            grid["p"], grid["d"], grid["q"], grid["seasonal_order"], grid.get("trend", ["n"])
        ))
        logger.info(f"  Grid combinations: {len(combos)}")

        for p, d, q, seasonal_order, trend in combos:
            result = _run_sarimax(
                train_y=train_y, test_y=test_y,
                train_exog=train_exog, test_exog=test_exog,
                p=p, d=d, q=q,
                seasonal_order=tuple(seasonal_order),
                trend=trend,
                artifact_dir=artifact_dir,
                train_size=len(train_y),
                test_size=len(test_y),
                context=data,
            )
            if result:
                all_results.append(result)

    if not all_results:
        raise RuntimeError("All SARIMAX configurations failed.")

    best = min(all_results, key=lambda r: r["metrics"]["test_rmse"])
    logger.info(
        f"Best SARIMAX -> {best['run_name']} | RMSE={best['metrics']['test_rmse']:.4f}"
    )
    return {"best_run": best, "all_results": all_results}
