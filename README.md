# 📈 Ad Sales Forecasting — MLOps Time Series Project

> **Production-grade** MLOps pipeline for forecasting daily advertisement revenue using ARIMA, SARIMAX, and LSTM models, with full MLflow experiment tracking, model registry, and inference.

---

## 🗂️ Project Structure

```
ad-sales-forecasting/
│
├── venv/                          ← Python virtual environment (created by you)
│
├── data/
│   ├── raw/
│   │   └── global_ads_performance_dataset.csv
│   └── processed/
│       ├── processed_ads.csv
│       ├── feature_scaler.pkl
│       ├── target_scaler.pkl
│       └── inference_predictions.csv
│
├── logs/                          ← ✨ Auto-created per-run log files
│   ├── 20260225_133000_train.log  ← timestamped log (YYYYMMDD_HHMMSS_label)
│   ├── 20260225_140000_infer.log
│   ├── run_history.txt            ← manifest of every run ever launched
│   └── archive/                   ← gzip-compressed old logs (auto-managed)
│       └── 20260224_120000_train.log.gz
│
├── notebooks/                     ← Jupyter exploration notebooks (optional)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             ← Load & aggregate daily time-series
│   ├── preprocessing.py           ← Clean, scale, split (forward-chain)
│   ├── feature_engineering.py     ← Lags, rolling windows, calendar features
│   ├── logger_setup.py            ← ✨ Centralised file + console logging
│   ├── evaluate.py                ← MAE / RMSE / MAPE / R² + plots
│   ├── train.py                   ← Master training orchestrator
│   ├── inference.py               ← 7-day forecast from registered model
│   └── models/
│       ├── __init__.py
│       ├── arima_model.py         ← ARIMA grid search + MLflow logging
│       ├── sarimax_model.py       ← SARIMAX with exogenous features
│       └── lstm_model.py          ← PyTorch LSTM + early stopping + CUDA
│
├── artifacts/                     ← Local artifact store (auto-created)
│   ├── arima/
│   ├── sarimax/
│   └── lstm/
│
├── mlruns/                        ← MLflow local tracking store
│
├── config.yaml                    ← Central configuration file
├── requirements.txt               ← Pinned dependencies
├── main.py                        ← CLI entry-point
└── README.md
```

---

## ⚙️ Environment Setup

### 1. Create Virtual Environment

```bash
cd ad-sales-forecasting
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows (PowerShell)**
```powershell
venv\Scripts\Activate.ps1
```

**Windows (CMD)**
```cmd
venv\Scripts\activate.bat
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> 💡 The project auto-detects CUDA availability.  
> If a GPU is available, LSTM training runs on GPU automatically.

---

## � File Logging

Every run automatically creates a **timestamped log file** in the `logs/` folder.

### Log File Naming

```
logs/
  YYYYMMDD_HHMMSS_train.log   ← one file per training run
  YYYYMMDD_HHMMSS_infer.log   ← one file per inference run
  run_history.txt             ← master manifest of all runs
  archive/
    YYYYMMDD_HHMMSS_train.log.gz   ← auto-gzipped when limit exceeded
```

### How Archive Works

| Setting | Default | Description |
|---|---|---|
| `logging.max_log_files` | `5` | Max active `.log` files in `logs/` |
| Archive trigger | automatic | When the 6th run starts, the **oldest** log is gzip-compressed and moved to `logs/archive/` |
| Compression | gzip | Archived logs shrink ~80-90% compared to plain text |

### `run_history.txt` — Run Manifest

Every time a run starts, one line is appended:

```
2026-02-25 13:30:00  [     train]  → 20260225_133000_train.log
2026-02-25 14:00:00  [     infer]  → 20260225_140000_infer.log
```

This gives you a **permanent audit trail** of every run, even after logs are archived.

### Controlling Log Behaviour via `config.yaml`

```yaml
logging:
  log_dir:      "logs"   # folder where .log files are saved
  max_log_files: 5       # how many active logs before oldest is archived
  level:        "INFO"   # DEBUG | INFO | WARNING | ERROR
```

### Console vs File Output

| Handler | Format | Colours |
|---|---|---|
| Console (stdout) | `TIMESTAMP [LEVEL] module – message` | ✅ ANSI coloured |
| File (`.log`) | `TIMESTAMP [LEVEL] module – message` | ❌ plain text |

---



### Full Training (ARIMA + SARIMAX + LSTM)

```bash
python main.py train
```

### Train with Custom Config / Seed

```bash
python main.py train --config config.yaml --seed 123
```

### Inference Only (after training)

```bash
python main.py infer
```

### Forecast 14 Days

```bash
python main.py infer --days 14
```

### View Project Info

```bash
python main.py info
```

---

## 📊 MLflow Experiment Tracking

### Launch MLflow UI

```bash
mlflow ui --backend-store-uri mlruns
```

Then open your browser at: **http://127.0.0.1:5000**

### View Experiments

Navigate to the **"Ad_Sales_TimeSeries"** experiment to see:

| Level | Name Format | Example |
|---|---|---|
| Parent | `{MODEL}_main_experiment` | `ARIMA_main_experiment` |
| Child | `{MODEL}_p{p}_d{d}_q{q}` | `ARIMA_p1_d1_q1` |
| Child | `LSTM_lr{lr}_bs{bs}_seq{seq}` | `LSTM_lr0.001_bs32_seq14` |

### Metrics Logged (per child run)

| Metric | Description |
|---|---|
| `test_mae` | Mean Absolute Error |
| `test_rmse` | Root Mean Squared Error |
| `test_mape` | Mean Absolute Percentage Error |
| `test_r2` | R² Coefficient of Determination |
| `training_time_sec` | Wall-clock training time |
| `epoch_train_loss` | Per-epoch training loss (LSTM) |
| `epoch_val_loss` | Per-epoch validation loss (LSTM) |

### Artifacts Logged (per child run)

```
model/
  ├── arima_model.pkl / sarimax_model.pkl / lstm_model.pt
  └── model_summary.txt

plots/
  ├── {MODEL}_predictions.png
  └── {MODEL}_residuals.png

feature_importance/
  └── exog_coefficients.txt   (SARIMAX only)

inference/
  └── inference_predictions.csv
```

---

## 🏗️ Model Details

### A) ARIMA

- Grid search over `p ∈ {1,2}`, `d ∈ {1}`, `q ∈ {1,2}`
- Trained on unscaled raw target (handles stationarity internally)
- Uses `statsmodels.tsa.arima.model.ARIMA`

### B) SARIMAX

- Extends ARIMA with **seasonal components** and **exogenous ad features**:
  `impressions, clicks, CTR, CPC, ad_spend, conversions, CPA, ROAS`
- Seasonal order: `(P=1, D=1, Q=1, s=7)` — weekly seasonality
- Logs exog variable coefficients as feature importance

### C) LSTM (PyTorch)

| Component | Detail |
|---|---|
| Architecture | 2-layer stacked LSTM + linear head |
| Input | Sliding window of `seq_length` timesteps |
| Features | All exog cols + target (scaled) |
| CUDA | Auto-detected; logs `device_used` tag |
| Early stopping | Triggered if val loss stagnates (patience=7) |

---

## 🗃️ Model Registry

After training, the **best model** (lowest RMSE) is:

1. Logged to its MLflow run artifact store
2. Registered as **`Ad_Sales_Forecaster`** in MLflow Model Registry
3. Transitioned to **Staging** stage automatically

View in MLflow UI → **Models** tab

---

## 📋 Configuration (`config.yaml`)

```yaml
data:
  train_ratio: 0.70   # 70% training data
  val_ratio:   0.15   # 15% validation
  test_ratio:  0.15   # 15% test

arima:
  param_grid:
    p: [1, 2]
    d: [1]
    q: [1, 2]

sarimax:
  param_grid:
    p: [1, 2]
    d: [1]
    q: [1, 2]
    seasonal_order: [[1, 1, 1, 7]]

lstm:
  param_grid:
    learning_rate: [0.001, 0.01]
    batch_size: [32, 64]
    sequence_length: [14, 21]
  hidden_size: 128
  num_layers: 2
  epochs: 50
  patience: 7
```

---

## 🖥️ Sample Output

```
2026-02-25 13:00:00 [INFO] ══════════════════════════════════════════════════════════
2026-02-25 13:00:00 [INFO] STEP 1 — Loading raw data
2026-02-25 13:00:01 [INFO]   Daily rows: 365
2026-02-25 13:00:01 [INFO] STEP 2 — Feature engineering
2026-02-25 13:00:01 [INFO]   Feature engineering complete → 32 features
2026-02-25 13:00:01 [INFO] STEP 4a — ARIMA Experiment
2026-02-25 13:00:02 [INFO]   ARIMA parent run started …
2026-02-25 13:00:02 [INFO]   Fitting ARIMA_p1_d1_q1 …
2026-02-25 13:00:05 [INFO]     ARIMA_p1_d1_q1 → RMSE=1234.56 | R²=0.8732
...
2026-02-25 13:05:00 [INFO] OVERALL BEST → SARIMAX | SARIMAX_p1_d1_q1_P1_D1_Q1_s7
2026-02-25 13:05:01 [INFO]   RMSE  : 987.43
2026-02-25 13:05:01 [INFO]   MAE   : 654.21
2026-02-25 13:05:01 [INFO]   MAPE  : 8.34%
2026-02-25 13:05:01 [INFO]   R²    : 0.9241
2026-02-25 13:05:02 [INFO]   Model v1 → Staging ✓

7-Day Revenue Forecast:
─────────────────────────────────────
  2026-12-31 →   $ 12,345.67
  2027-01-01 →   $ 13,102.44
  ...
```

---

## 📸 Screenshots

> **MLflow Experiments View**
> *(After running training, open http://127.0.0.1:5000)*

```
Experiment: Ad_Sales_TimeSeries
├── ARIMA_main_experiment
│   ├── ARIMA_p1_d1_q1  [RMSE: 1234.56]
│   ├── ARIMA_p1_d1_q2  [RMSE: 1198.34]
│   ├── ARIMA_p2_d1_q1  [RMSE: 1210.45]
│   └── ARIMA_p2_d1_q2  [RMSE: 1201.77]
├── SARIMAX_main_experiment
│   ├── SARIMAX_p1_d1_q1_P1_D1_Q1_s7  [RMSE: 987.43] ← BEST
│   └── ...
└── LSTM_main_experiment
    ├── LSTM_lr0.001_bs32_seq14
    └── ...

Model Registry: Ad_Sales_Forecaster v1 [Staging]
```

---

## 🔬 Dataset

| Column | Type | Description |
|---|---|---|
| `date` | Date | Daily date index |
| `impressions` | int | Ad impressions |
| `clicks` | int | Ad clicks |
| `CTR` | float | Click-through rate |
| `CPC` | float | Cost per click |
| `ad_spend` | float | Total ad spend ($) |
| `conversions` | int | Conversion count |
| `CPA` | float | Cost per acquisition |
| **`revenue`** | **float** | **Target: daily revenue ($)** |
| `ROAS` | float | Return on ad spend |

---

## 🤖 CUDA Support

```python
# Automatic detection (lstm_model.py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- CPU: Compatible with any machine
- GPU: Significantly faster LSTM training
- `device_used` is logged as an MLflow tag per LSTM run

---

## 🛠️ Troubleshooting

| Issue | Fix |
|---|---|
| `FileNotFoundError: dataset not found` | Ensure CSV is in `data/raw/` |
| `MLflow tracking URI error` | Run from project root directory |
| `ARIMA convergence warning` | Increase `maxiter` in `ARIMA.fit()` |
| `CUDA out of memory` | Reduce `batch_size` in config.yaml |
| `Port 5000 already in use` | Run `mlflow ui --port 5001` |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `statsmodels` | ARIMA, SARIMAX models |
| `torch` | LSTM + GPU support |
| `mlflow` | Experiment tracking + registry |
| `pandas` | Time-series data manipulation |
| `scikit-learn` | Metrics + preprocessing |
| `matplotlib` | Visualization |
| `click` | CLI argument parsing |
| `pyyaml` | Config file parsing |

---

*Built with ❤️ as a production-grade MLOps Time Series project.*
