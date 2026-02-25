"""
lstm_model.py
=============
PyTorch LSTM for time-series forecasting with:
  * Automatic CUDA detection
  * Sequence dataset builder
  * Early stopping
  * MLflow parent + child run logging

MLflow Run Structure
--------------------
Parent: "LSTM_main_experiment"
  └- Child: "LSTM_lr{lr}_bs{bs}_seq{seq}"
"""

import logging
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from src.evaluate import compute_metrics, plot_predictions, plot_residuals

logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# Global device
# --------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cfg(config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------
# Dataset
# --------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for LSTM training.

    Parameters
    ----------
    features      : 2-D numpy array [timesteps, n_features].
    targets       : 1-D numpy array [timesteps].
    seq_length    : Number of past timesteps given to LSTM per sample.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_length: int):
        self.X, self.y = [], []
        for i in range(len(targets) - seq_length):
            self.X.append(features[i: i + seq_length])
            self.y.append(targets[i + seq_length])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --------------------------------------------------------------
# Model architecture
# --------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """
    Stacked LSTM with a linear output head.

    Parameters
    ----------
    input_size  : Number of input features (exog + target).
    hidden_size : LSTM hidden-state dimension.
    num_layers  : Number of stacked LSTM layers.
    dropout     : Dropout applied between LSTM layers (0 if num_layers=1).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers:  int = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_size]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :]).squeeze(-1)   # [batch]


# --------------------------------------------------------------
# Early Stopping
# --------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Parameters
    ----------
    patience : Epochs to wait after last improvement.
    delta    : Minimum change to count as improvement.
    """

    def __init__(self, patience: int = 7, delta: float = 1e-4):
        self.patience = patience
        self.delta    = delta
        self.counter  = 0
        self.best_loss: float | None = None
        self.stop = False

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# --------------------------------------------------------------
# Training loop
# --------------------------------------------------------------

def _build_model_summary(model: nn.Module, input_shape: tuple) -> str:
    """Return a string summary of the model architecture."""
    lines = [f"LSTMForecaster Architecture", "=" * 40]
    lines.append(f"Input shape : {input_shape}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, module in model.named_children():
        lines.append(f"  [{name}] {module}")
    lines.append("-" * 40)
    lines.append(f"Total params    : {total_params:,}")
    lines.append(f"Trainable params: {trainable:,}")
    return "\n".join(lines)


def _train_lstm(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    model:        LSTMForecaster,
    lr:           float,
    epochs:       int,
    patience:     int,
    clip_norm:    float,
    weight_decay: float,
) -> tuple[list, list]:
    """
    Run the full training loop with early stopping.

    Returns
    -------
    (train_losses, val_losses) lists for every epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-6
    )
    stopper   = EarlyStopping(patience=patience)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # -- Train --------------------------------------------
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            batch_losses.append(loss.item())

        t_loss = float(np.mean(batch_losses))
        train_losses.append(t_loss)

        # -- Validate -----------------------------------------
        model.eval()
        vb_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred    = model(xb)
                vb_losses.append(criterion(pred, yb).item())

        v_loss = float(np.mean(vb_losses))
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        stopper(v_loss)

        if epoch % 10 == 0 or stopper.stop:
            logger.info(
                f"    Epoch {epoch:3d}/{epochs} | "
                f"train_loss={t_loss:.6f} | val_loss={v_loss:.6f}"
            )

        if stopper.stop:
            logger.info(f"    Early stopping at epoch {epoch}")
            break

    return train_losses, val_losses


# --------------------------------------------------------------
# Single LSTM run + MLflow child
# --------------------------------------------------------------

def _run_lstm(
    train_scaled:   pd.DataFrame,
    val_scaled:     pd.DataFrame,
    test_scaled:    pd.DataFrame,
    target_scaler,
    feature_cols:   list[str],
    target_col:     str,
    lr:             float,
    batch_size:     int,
    seq_length:     int,
    hidden_size:    int,
    num_layers:     int,
    dropout:        float,
    weight_decay:   float,
    epochs:         int,
    patience:       int,
    clip_norm:      float,
    artifact_dir:   str,
    context:        dict,
) -> dict:
    """Fit one LSTM config and log to MLflow as a child run."""
    run_name = f"LSTM_lr{lr}_bs{batch_size}_seq{seq_length}_do{dropout}_wd{weight_decay}"
    logger.info(f"  Fitting {run_name} on {DEVICE} …")

    all_cols = feature_cols + [target_col]

    # Numpy arrays
    train_feat = train_scaled[all_cols].values
    val_feat   = val_scaled[all_cols].values
    test_feat  = test_scaled[all_cols].values

    train_target = train_scaled[target_col].values
    val_target   = val_scaled[target_col].values
    test_target  = test_scaled[target_col].values

    train_ds = TimeSeriesDataset(train_feat, train_target, seq_length)
    val_ds   = TimeSeriesDataset(val_feat,   val_target,   seq_length)
    test_ds  = TimeSeriesDataset(test_feat,  test_target,  seq_length)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = LSTMForecaster(
        input_size=len(all_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)

    with mlflow.start_run(run_name=run_name, nested=True) as child_run:
        mlflow.log_params({
            "model_type":     "LSTM",
            "learning_rate":  lr,
            "batch_size":     batch_size,
            "sequence_length": seq_length,
            "hidden_size":    hidden_size,
            "num_layers":     num_layers,
            "dropout":        dropout,
            "weight_decay":   weight_decay,
            "epochs":         epochs,
            "patience":       patience,
            "train_size":     len(train_scaled),
            "test_size":      len(test_scaled),
            "device_used":    str(DEVICE).upper(),
            "cuda_available": torch.cuda.is_available(),
        })

        mlflow.set_tags({
            "dataset_version": context.get("dataset_version", "v1.0"),
            "dataset_hash": context.get("dataset_hash", "unknown"),
            "device": str(DEVICE),
            "features": ",".join(feature_cols),
            "forecast_horizon": context.get("forecast_horizon", 7)
        })

        t0 = time.perf_counter()
        train_losses, val_losses = _train_lstm(
            train_ld, val_ld, model, lr, epochs, patience, clip_norm, weight_decay
        )
        train_time = time.perf_counter() - t0

        # Epoch-level loss logging
        for ep, (tl, vl) in enumerate(zip(train_losses, val_losses), 1):
            mlflow.log_metrics({"epoch_train_loss": tl, "epoch_val_loss": vl}, step=ep)

        # Test predictions (inverse-scale)
        model.eval()
        all_preds = []
        with torch.no_grad():
            for xb, _ in test_ld:
                all_preds.extend(model(xb.to(DEVICE)).cpu().numpy().tolist())

        preds_scaled = np.array(all_preds)
        preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        # True values (inverse-scale using the same subset length)
        test_true_raw = test_ds.y.numpy()
        y_true = target_scaler.inverse_transform(test_true_raw.reshape(-1, 1)).flatten()

        metrics = compute_metrics(y_true, preds, prefix="test")
        metrics["training_time_sec"] = round(train_time, 4)
        mlflow.log_metrics(metrics)

        # Artifacts
        run_dir = Path(artifact_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = str(run_dir / "lstm_model.pt")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        # Save summary
        summary_str = _build_model_summary(model, (batch_size, seq_length, len(all_cols)))
        summary_path = str(run_dir / "model_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary_str)
        mlflow.log_artifact(summary_path, artifact_path="model")

        # Plots
        dates = test_scaled.index[seq_length:]
        pred_plot = plot_predictions(dates, y_true, preds, run_name, str(run_dir))
        res_plot  = plot_residuals(dates, y_true, preds, run_name, str(run_dir))
        mlflow.log_artifact(pred_plot, artifact_path="plots")
        mlflow.log_artifact(res_plot,  artifact_path="plots")
        
        # Dataset artifact
        if "processed_csv_path" in context:
            mlflow.log_artifact(context["processed_csv_path"], artifact_path="dataset")

        # MLflow Signature & Input Example
        sample_x = train_ds[0][0].numpy()  # [sequence_length, input_size]
        sample_y = train_ds[0][1].numpy()  # scalar
        input_example = sample_x[np.newaxis, ...]  # shape [1, seq_length, input_size]
        sig = infer_signature(input_example, np.array([sample_y]))
        
        mlflow.pytorch.log_model(
            model,
            artifact_path="mlflow_native_model",
            signature=sig,
            input_example=input_example
        )

        logger.info(f"    {run_name} -> RMSE={metrics['test_rmse']:.4f} | R²={metrics['test_r2']:.4f}")

        return {
            "run_id":       child_run.info.run_id,
            "run_name":     run_name,
            "metrics":      metrics,
            "model":        model,
            "model_state":  model.state_dict(),
            "predictions":  preds,
            "config": {
                "input_size":  len(all_cols),
                "hidden_size": hidden_size,
                "num_layers":  num_layers,
                "dropout":     dropout,
                "seq_length":  seq_length,
            },
        }


# --------------------------------------------------------------
# Parent LSTM experiment
# --------------------------------------------------------------

def run_lstm_experiment(
    data: dict,
    config_path:  str = "config.yaml",
    artifact_dir: str = "artifacts/lstm",
) -> dict:
    """
    LSTM hyperparameter grid search under a parent MLflow run.

    Uses **scaled** data (from preprocessing) so LSTM gradients
    are well-conditioned.
    """
    cfg  = _cfg(config_path)
    lcfg = cfg["lstm"]
    grid = lcfg["param_grid"]

    torch.manual_seed(cfg["project"]["seed"])
    np.random.seed(cfg["project"]["seed"])

    all_results: list[dict] = []

    with mlflow.start_run(run_name="LSTM_main_experiment") as parent_run:
        mlflow.set_tag("model_family", "LSTM")
        mlflow.set_tag("device", str(DEVICE))
        mlflow.set_tag("cuda_available", str(torch.cuda.is_available()))
        logger.info(f"LSTM parent run started | device={DEVICE} …")

        combos = [
            (lr, bs, sq, do, wd)
            for lr in grid["learning_rate"]
            for bs in grid["batch_size"]
            for sq in grid["sequence_length"]
            for do in grid.get("dropout", [0.0])
            for wd in grid.get("weight_decay", [0.0])
        ]
        logger.info(f"  Grid combinations: {len(combos)}")

        for lr, bs, sq, do, wd in combos:
            result = _run_lstm(
                train_scaled=data["train_scaled"],
                val_scaled=data["val_scaled"],
                test_scaled=data["test_scaled"],
                target_scaler=data["target_scaler"],
                feature_cols=data["feature_cols"],
                target_col=data["target_col"],
                lr=lr,
                batch_size=bs,
                seq_length=sq,
                hidden_size=lcfg["hidden_size"],
                num_layers=lcfg["num_layers"],
                dropout=do,
                weight_decay=wd,
                epochs=lcfg["epochs"],
                patience=lcfg["patience"],
                clip_norm=lcfg["clip_grad_norm"],
                artifact_dir=artifact_dir,
                context=data,
            )
            if result:
                all_results.append(result)

    if not all_results:
        raise RuntimeError("All LSTM configurations failed.")

    best = min(all_results, key=lambda r: r["metrics"]["test_rmse"])
    logger.info(
        f"Best LSTM -> {best['run_name']} | RMSE={best['metrics']['test_rmse']:.4f}"
    )
    return {"best_run": best, "all_results": all_results}
