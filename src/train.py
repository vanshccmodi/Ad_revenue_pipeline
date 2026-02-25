"""
train.py
========
Orchestrates the full training pipeline:

  1. Load + aggregate daily data
  2. Preprocess (scale + split)
  3. Feature engineering
  4. Train ARIMA, SARIMAX, LSTM
  5. Register best model in MLflow Model Registry

MLflow Setup
------------
  Experiment : "Ad_Sales_TimeSeries"
  Tracking   : local ./mlruns folder
"""

import hashlib
import logging
import os
import pickle
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import yaml

from src.data_loader          import load_raw_data, aggregate_daily
from src.feature_engineering  import engineer_features
from src.logger_setup         import setup_logging
from src.models.arima_model   import run_arima_experiment
from src.models.lstm_model    import run_lstm_experiment, LSTMForecaster
from src.models.sarimax_model import run_sarimax_experiment
from src.preprocessing        import preprocess

# Module-level logger (handlers are added by setup_logging at runtime)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------
# Config helper
# --------------------------------------------------------------

def _cfg(config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------
# MLflow setup
# --------------------------------------------------------------

def setup_mlflow(cfg: dict) -> str:
    """
    Configure MLflow tracking URI and experiment.

    Returns
    -------
    experiment_id
    """
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = cfg["mlflow"]["experiment_name"]

    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(exp_name)
        logger.info(f"Created MLflow experiment: '{exp_name}' (id={exp_id})")
    else:
        exp_id = exp.experiment_id
        logger.info(f"Using existing MLflow experiment: '{exp_name}' (id={exp_id})")

    mlflow.set_experiment(exp_name)
    return exp_id


# --------------------------------------------------------------
# Model registry
# --------------------------------------------------------------

def register_best_model(
    best_info: dict,
    model_type: str,
    data: dict,
    cfg: dict,
) -> None:
    """
    Register the single best model across ALL model families
    in the MLflow Model Registry under "Ad_Sales_Forecaster".

    Transitions it to Staging.
    """
    registry_name = cfg["mlflow"]["registry_model_name"]
    run_id = best_info["run_id"]

    client = mlflow.MlflowClient()

    try:
        # Ensure the registered model exists
        client.get_registered_model(registry_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(
            registry_name,
            description="Best Ad Sales Forecaster from time-series experiments",
        )
        logger.info(f"  Registry model '{registry_name}' created.")

    # Connect to the exact artifact path where the child run logged the signed model
    artifact_path = "mlflow_native_model"
    run_uri       = f"runs:/{run_id}/{artifact_path}"

    # Verify the native model exists before registering
    try:
        # Register version directly from the logged artifact
        mv = client.create_model_version(
            name=registry_name,
            source=run_uri,
            run_id=run_id,
            description=f"Best {model_type} model | RMSE={best_info['metrics']['test_rmse']:.4f}",
        )
    except Exception as e:
        logger.error(f"Failed to register model from path '{run_uri}' (maybe it wasn't logged?): {e}")
        raise
    logger.info(f"  Model version created: {mv.version}")

    # Transition to Staging
    client.transition_model_version_stage(
        name=registry_name,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    logger.info(f"  Model v{mv.version} -> Staging ✓")


# --------------------------------------------------------------
# Main training pipeline
# --------------------------------------------------------------

def run_training(config_path: str = "config.yaml", seed: int = 42) -> None:
    """
    Full training pipeline entry-point.

    Parameters
    ----------
    config_path : Path to config.yaml.
    seed        : Global random seed.
    """
    # - File logging (only set up here if not already set up by main.py)
    if not logging.getLogger().handlers:
        setup_logging(run_label="train")

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = _cfg(config_path)
    setup_mlflow(cfg)

    # --- Step 1: Load data -----------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1 - Loading raw data")
    raw_df = load_raw_data(config_path=config_path)
    daily_df = aggregate_daily(raw_df, config_path=config_path)
    logger.info(f"  Daily rows: {len(daily_df)}")

    # --- Step 2: Feature engineering ------------------------
    logger.info("STEP 2 - Feature engineering")
    exog_cols = [c for c in cfg["data"]["exog_cols"] if c in daily_df.columns]
    enriched_df = engineer_features(
        daily_df, target_col=cfg["data"]["target_col"], exog_cols=exog_cols
    )

    # --- Step 3: Preprocess (split + scale) -----------------
    logger.info("STEP 3 - Preprocessing (split + scale)")
    # Use enriched df but keep original exog cols for ARIMA/SARIMAX raw
    data = preprocess(daily_df, config_path=config_path)
    # For LSTM, use enriched features
    eng_data = preprocess(enriched_df, config_path=config_path)

    # Save processed dataset snapshot
    proc_path = cfg["paths"]["processed_data"]
    Path(proc_path).parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(proc_path)
    logger.info(f"  Processed snapshot saved -> {proc_path}")

    # Compute MLOps Context (Hash, Horizon)
    ds_hash = hashlib.md5(pd.util.hash_pandas_object(daily_df, index=True).values).hexdigest()
    horizon = cfg["inference"]["forecast_days"]

    for d_dict in (data, eng_data):
        d_dict["dataset_hash"] = ds_hash
        d_dict["dataset_version"] = "v1.0"
        d_dict["forecast_horizon"] = horizon
        d_dict["processed_csv_path"] = proc_path

    # --- Step 4: Train models --------------------------------
    results_by_model: dict[str, dict] = {}

    logger.info("=" * 60)
    logger.info("STEP 4a - ARIMA Experiment")
    try:
        arima_res = run_arima_experiment(data=data, config_path=config_path)
        results_by_model["ARIMA"] = arima_res["best_run"]
    except Exception as e:
        logger.error(f"  ARIMA failed: {e}")

    logger.info("=" * 60)
    logger.info("STEP 4b - SARIMAX Experiment")
    try:
        sarimax_res = run_sarimax_experiment(data=data, config_path=config_path)
        results_by_model["SARIMAX"] = sarimax_res["best_run"]
    except Exception as e:
        logger.error(f"  SARIMAX failed: {e}")

    logger.info("=" * 60)
    logger.info("STEP 4c - LSTM Experiment")
    try:
        lstm_res = run_lstm_experiment(data=eng_data, config_path=config_path)
        results_by_model["LSTM"] = lstm_res["best_run"]
    except Exception as e:
        logger.error(f"  LSTM failed: {e}")

    if not results_by_model:
        raise RuntimeError("All model families failed. Check configuration and data.")

    # --- Step 5: Select overall best ------------------------
    logger.info("=" * 60)
    logger.info("STEP 5 - Selecting best model")
    overall_best = min(
        results_by_model.items(),
        key=lambda kv: kv[1]["metrics"]["test_rmse"],
    )
    best_type, best_run = overall_best

    logger.info("=" * 60)
    logger.info(f"  OVERALL BEST -> {best_type} | {best_run['run_name']}")
    logger.info(f"  RMSE  : {best_run['metrics']['test_rmse']:.4f}")
    logger.info(f"  MAE   : {best_run['metrics']['test_mae']:.4f}")
    logger.info(f"  MAPE  : {best_run['metrics']['test_mape']:.2f}%")
    logger.info(f"  R²    : {best_run['metrics']['test_r2']:.4f}")
    logger.info("=" * 60)

    # Comparison summary
    logger.info("Model comparison:")
    for mtype, res in results_by_model.items():
        logger.info(
            f"  {mtype:<10} RMSE={res['metrics']['test_rmse']:.4f} "
            f"R²={res['metrics']['test_r2']:.4f}"
        )

    # --- Step 6: Register best model ------------------------
    logger.info("STEP 6 - Registering best model")
    try:
        register_best_model(
            best_info=best_run,
            model_type=best_type,
            data=data,
            cfg=cfg,
        )
    except Exception as e:
        logger.error(f"  Model registration failed (non-fatal): {e}")

    logger.info("Training pipeline complete ✓")
