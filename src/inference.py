"""
inference.py
============
Inference pipeline for the registered "Ad_Sales_Forecaster" model.

Responsibilities
----------------
* Load the latest registered model from MLflow Model Registry.
* Generate 7-day rolling forecast on unseen data.
* Log inference metrics and save predictions CSV as MLflow artifact.
"""

import logging
import pickle
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import yaml

from src.data_loader          import load_raw_data, aggregate_daily
from src.logger_setup         import setup_logging
from src.models.lstm_model    import LSTMForecaster, TimeSeriesDataset
from src.preprocessing        import inverse_scale_target
from src.evaluate             import compute_metrics

logger = logging.getLogger(__name__)


def _cfg(config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------
# Load scalers
# --------------------------------------------------------------

def _load_scalers(proc_dir: str = "data/processed") -> tuple:
    import pickle
    with open(f"{proc_dir}/feature_scaler.pkl", "rb") as f:
        feat_scaler = pickle.load(f)
    with open(f"{proc_dir}/target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)
    return feat_scaler, target_scaler


# --------------------------------------------------------------
# Retrieve best model from registry
# --------------------------------------------------------------

def _get_latest_staging_run(registry_name: str) -> tuple[str, str]:
    """
    Return (run_id, model_type) from the latest Staging version.

    Model type is inferred from the run tags or name.
    """
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(registry_name, stages=["Staging"])
    if not versions:
        versions = client.get_latest_versions(registry_name)
    if not versions:
        raise RuntimeError(
            f"No versions found in registry '{registry_name}'. "
            "Run training first."
        )

    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    run_id = latest.run_id
    logger.info(
        f"  Using model version {latest.version} | stage={latest.current_stage} | run_id={run_id}"
    )

    # Infer model type from run name
    run_info = client.get_run(run_id)
    run_name = run_info.data.tags.get("mlflow.runName", "")
    if "LSTM" in run_name.upper():
        model_type = "LSTM"
    elif "SARIMAX" in run_name.upper():
        model_type = "SARIMAX"
    else:
        model_type = "ARIMA"

    return run_id, model_type


# --------------------------------------------------------------
# ARIMA / SARIMAX forecaster
# --------------------------------------------------------------

def _forecast_statsmodels(
    run_id: str,
    daily_df: pd.DataFrame,
    cfg: dict,
    forecast_days: int,
) -> pd.DataFrame:
    """Download pickled statsmodels model and forecast 'forecast_days' steps."""
    client = mlflow.MlflowClient()

    # Try to find the pkl artifact
    artifacts = client.list_artifacts(run_id, path="model")
    pkl_files  = [a for a in artifacts if a.path.endswith(".pkl")]
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl model artifact found in run {run_id}")

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=pkl_files[0].path
    )

    with open(local_path, "rb") as f:
        fitted = pickle.load(f)

    target_col = cfg["data"]["target_col"]
    exog_cols  = [c for c in cfg["data"]["exog_cols"] if c in daily_df.columns]

    # Extend the series (use last known exog if needed)
    last_exog = daily_df[exog_cols].iloc[-forecast_days:]
    fc         = fitted.forecast(steps=forecast_days, exog=last_exog if len(last_exog) == forecast_days else None)
    preds      = np.array(fc)

    last_date  = daily_df.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
    return pd.DataFrame({"date": future_dates, "predicted_revenue": preds})


# --------------------------------------------------------------
# LSTM forecaster
# --------------------------------------------------------------

def _forecast_lstm(
    run_id: str,
    daily_scaled: pd.DataFrame,
    cfg: dict,
    forecast_days: int,
    feat_scaler,
    target_scaler,
) -> pd.DataFrame:
    """Load LSTM weights and auto-regressively forecast forecast_days steps."""
    client = mlflow.MlflowClient()

    artifacts = client.list_artifacts(run_id, path="model")
    pt_files   = [a for a in artifacts if a.path.endswith(".pt")]
    txt_files  = [a for a in artifacts if a.path.endswith("model_summary.txt")]

    if not pt_files:
        raise FileNotFoundError(f"No .pt model artifact found in run {run_id}")

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=pt_files[0].path
    )

    run_data  = mlflow.MlflowClient().get_run(run_id).data
    params    = run_data.params

    seq_length   = int(params.get("sequence_length", 14))
    hidden_size  = int(params.get("hidden_size", 128))
    num_layers   = int(params.get("num_layers", 2))
    dropout      = float(params.get("dropout", 0.2))
    target_col   = cfg["data"]["target_col"]
    exog_cols    = [c for c in cfg["data"]["exog_cols"] if c in daily_scaled.columns]
    all_cols     = exog_cols + [target_col]
    input_size   = len(all_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMForecaster(input_size, hidden_size, num_layers, dropout).to(device)
    model.load_state_dict(torch.load(local_path, map_location=device))
    model.eval()

    # Seed window
    window_data = daily_scaled[all_cols].values[-seq_length:]

    predictions: list[float] = []
    for _ in range(forecast_days):
        x = torch.tensor(window_data[-seq_length:], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_scaled = model(x).cpu().item()

        # Inverse scale
        pred_real = target_scaler.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(pred_real)

        # Shift window (repeat last exog row, update target)
        next_row = window_data[-1].copy()
        next_row[-1] = pred_scaled           # update target col position
        window_data = np.vstack([window_data[1:], next_row])

    last_date    = daily_scaled.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
    return pd.DataFrame({"date": future_dates, "predicted_revenue": predictions})


# --------------------------------------------------------------
# Main inference entry-point
# --------------------------------------------------------------

def run_inference(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Full inference pipeline.

    1. Connect to MLflow and get the latest Staging model.
    2. Load and preprocess recent data.
    3. Forecast next <inference.forecast_days> days.
    4. Log metrics and save predictions CSV artifact.

    Returns
    -------
    pd.DataFrame with columns [date, predicted_revenue].
    """
    cfg = _cfg(config_path)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # - File logging (only set up here if not already set up by main.py)
    if not logging.getLogger().handlers:
        setup_logging(run_label="infer")

    registry_name  = cfg["mlflow"]["registry_model_name"]
    forecast_days  = cfg["inference"]["forecast_days"]
    proc_dir       = str(Path(cfg["paths"]["processed_data"]).parent)
    target_col     = cfg["data"]["target_col"]

    logger.info("=" * 60)
    logger.info(f"Starting inference pipeline | forecast_days={forecast_days}")

    # -- Load data --------------------------------------------
    raw_df   = load_raw_data(config_path=config_path)
    daily_df = aggregate_daily(raw_df, config_path=config_path)

    # -- Model lookup -----------------------------------------
    run_id, model_type = _get_latest_staging_run(registry_name)
    logger.info(f"  Model type: {model_type}")

    # -- Inference --------------------------------------------
    with mlflow.start_run(run_name=f"{model_type}_inference"):
        mlflow.log_params({
            "model_type":    model_type,
            "inference_run": True,
            "forecast_days": forecast_days,
            "registry_name": registry_name,
            "source_run_id": run_id,
        })

        try:
            feat_scaler, target_scaler = _load_scalers(proc_dir)
        except FileNotFoundError:
            logger.warning("  Scalers not found - running without LSTM inference scaling.")
            feat_scaler = target_scaler = None

        if model_type == "LSTM" and feat_scaler is not None:
            exog_cols    = [c for c in cfg["data"]["exog_cols"] if c in daily_df.columns]
            daily_scaled = daily_df.copy()
            daily_scaled[exog_cols]  = feat_scaler.transform(daily_df[exog_cols].values)
            daily_scaled[target_col] = target_scaler.transform(daily_df[[target_col]].values)
            preds_df = _forecast_lstm(run_id, daily_scaled, cfg, forecast_days, feat_scaler, target_scaler)
        else:
            preds_df = _forecast_statsmodels(run_id, daily_df, cfg, forecast_days)

        # -- Save predictions ----------------------------------
        out_path = "data/processed/inference_predictions.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(out_path, index=False)
        mlflow.log_artifact(out_path, artifact_path="inference")
        logger.info(f"  Predictions saved -> {out_path}")

        # -- Log summary metrics -------------------------------
        mlflow.log_metric("forecast_days", forecast_days)
        mlflow.log_metric("avg_predicted_revenue", float(preds_df["predicted_revenue"].mean()))
        mlflow.log_metric("min_predicted_revenue", float(preds_df["predicted_revenue"].min()))
        mlflow.log_metric("max_predicted_revenue", float(preds_df["predicted_revenue"].max()))

        logger.info("=" * 60)
        logger.info("7-Day Revenue Forecast:")
        logger.info("-" * 40)
        for _, row in preds_df.iterrows():
            logger.info(f"  {row['date'].strftime('%Y-%m-%d')} -> ${row['predicted_revenue']:>12,.2f}")
        logger.info("-" * 40)
        logger.info(f"  Average : ${preds_df['predicted_revenue'].mean():>12,.2f}")
        logger.info("=" * 60)

    return preds_df


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    predictions = run_inference(config_path=cfg_path)
    print("\n" + predictions.to_string(index=False))
