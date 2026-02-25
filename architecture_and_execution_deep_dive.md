# Ad Sales Forecasting - Deep Dive Architecture & Execution Guide

This document provides an exhaustive, deep-dive explanation of the Ad Sales Forecasting project. It covers the overall architecture, detailing every component, file purpose, and the execution flow of the machine learning pipeline from raw data to model inference.

---

## 🏗️ 1. Overall System Architecture

The project is built as an **end-to-end MLOps time-series forecasting pipeline**. It is designed to predict daily advertising `revenue` using a combination of past revenue (autoregressive features), calendar signals, and exogenous marketing metrics (like `ad_spend`, `clicks`, `impressions`). 

The architecture is strictly modularized into distinct stages: Data Extraction -> Feature Engineering -> Preprocessing -> Model Training & Tuning -> Model Registration -> Inference.

### Core Technologies
*   **Python 3.10+**: Core programming language.
*   **Pandas & NumPy**: For data manipulation, aggregation, and mathematical operations.
*   **Statsmodels**: Provides statistical time-series algorithms (ARIMA, SARIMAX).
*   **PyTorch**: Provides deep learning capabilities for the LSTM network.
*   **MLflow**: Central nervous system for MLOps. It tracks experiments, logs metrics/parameters/artifacts, and acts as the Model Registry.
*   **Click**: Provides a robust Command Line Interface (CLI) to execute different parts of the pipeline cleanly.
*   **PyYAML**: Parses the `config.yaml` file, keeping code and configuration isolated.

---

## 📂 2. Detailed File & Directory Structure Deep Dive

Here is the breakdown of what every single file and directory actually does under the hood.

### 2.1 Root Level Files

*   **`main.py`**: The central execution entry point. It defines a CLI using the `click` library. It exposes commands like `train` (which runs the full end-to-end pipeline), `infer` (for generating predictions), and `info` (for dumping project stats). It acts as the switchboard that imports and invokes the actual logic buried in the `src/` directory.
*   **`config.yaml`**: The central nervous system for hyperparameters and pathways. Instead of hardcoding paths or learning rates in Python files, the pipeline reads from this YAML. It defines:
    *   File paths for raw/processed data and MLflow directories.
    *   Target column (`revenue`) and exogenous columns to use (`impressions`, `clicks`, etc.).
    *   Data splitting ratios (70% Train, 15% Val, 15% Test).
    *   Extensive parameter grids (`param_grid`) for hyperparameter tuning across ARIMA, SARIMAX, and PyTorch LSTM models.
*   **`requirements.txt`**: Specifies all Python dependencies to ensure reproducibility.
*   **`README.md`**: High-level project overview and quickstart instructions.
*   **`.gitignore`**: Tells Git to ignore sensitive or heavy folders like `__pycache__`, `venv`, `mlruns/`, `data/`, and `artifacts/`.

### 2.2 The `src/` Directory (Source Code)

*   **`src/train.py`**: The maestro of the training phase. When you run `python main.py train`, it triggers `run_training()` here. This script:
    1. Orchestrates loading data (`data_loader.py`).
    2. Triggers feature engineerng (`feature_engineering.py`).
    3. Triggers data splitting and scaling (`preprocessing.py`).
    4. Kicks off three separate experiments by calling ARIMA, SARIMAX, and LSTM trainer functions.
    5. Evaluates which of the three families generated the lowest RMSE.
    6. Automatically registers the overall winning model to the MLflow Model Registry and transitions it to the "Staging" state.
*   **`src/inference.py`**: The script responsible for actual forecasting into the future. Used when `python main.py infer` is called. It pulls the latest "Staging" model direct from the MLflow registry. It then fetches the most recent chunk of processed data (using the required lookback window), scales it using the saved scaler artifact, feeds it into the model, generates `X` days of predictions (where `X` is from `config.yaml` or CLI arguments), and inverse-transforms the output back to real dollar values.
*   **`src/data_loader.py`**: Handles ingesting the raw CSV file (`global_ads_performance_dataset.csv`). Crucially, it converts datetime columns and performs a `groupby("date").sum()` operation. This aggregation is necessary because the raw data might have multiple entries per day (e.g., across different campaigns), but time-series forecasting requires a strict daily sequence.
*   **`src/feature_engineering.py`**: Enriches the dataset to give the models "hints" about patterns in time.
    *   **Calendar Features**: Adds integer flags for day-of-week, month, quarter, week-of-year, and weekend status.
    *   **Rolling Statistics**: Calculates Moving Averages (MA7, MA14, MA30) and rolling standard deviations for the target and key exogenous variables (ad spend, clicks, impressions).
    *   **Lag Features**: Shifts data downwards by 1, 7, 14, and 30 days. This allows the model on "Day T" to explicitly "see" what happened on "Day T-7".
    *   **Momentum**: Calculates day-over-day percentage changes.
*   **`src/preprocessing.py`**: Responsible for creating training, validation, and test splits while preventing *data leakage*. It relies strictly on temporal splitting (e.g., first 70% is train, next 15% is validation, final 15% test). Once split, it fits a `MinMaxScaler` **only on the training split** and transforms all splits. The fit scaler is later saved as a `.pkl` artifact so inference can use the exact same scale parameters.
*   **`src/evaluate.py`**: A unified evaluation module containing functions that calculate regression metrics: RMSE, MAE, MAPE, and R-squared. These functions ensure that ARIMA, SARIMAX, and LSTM models are judged fairly using the exact same mathematical formulas.
*   **`src/logger_setup.py`**: Provides standardized logging formatting so that the console Output and log files (in `logs/`) are consistently stamped with timestamps, log levels, and module names.

### 2.3 The `src/models/` Directory

*   **`arima_model.py`**: 
    Iterates through the ARIMA parameter grid specified in `config.yaml`. For each combination of `p` (autoregressive), `d` (differencing), `q` (moving average), and `trend`, it starts an MLflow sub-run. It fits the `statsmodels` ARIMA, executes predictions for the test horizon, evaluates metrics, and logs the model to MLflow.
*   **`sarimax_model.py`**:
    Similar to ARIMA, but introduces Exogenous variables (`X` matrix features like `ad_spend`) and Seasonality. It iterates through combinations including `seasonal_order` (P, D, Q, s) to natively account for 7-day weekly patterns. Logs every run to MLflow.
*   **`lstm_model.py`**:
    Deep learning time-series implementation.
    *   Features a custom PyTorch `nn.Module` containing LSTM layers followed by a fully connected output projection.
    *   Implements a custom PyTorch `Dataset` that utilizes a sliding window (e.g., uses sequence of past 14 days to predict the 15th day).
    *   Iterates over the LSTM hyperparameter grid (`learning_rate`, `batch_size`, `sequence_length`, `dropout`, `weight_decay`) in `config.yaml`.
    *   Trains with an Early Stopping mechanism (monitoring Validation Loss) via an Adam Optimizer and MSE Loss.
    *   Because MLflow supports native `mlflow.pytorch.log_model`, it stores the full PyTorch network weights inside MLflow artifacts.

### 2.4 Data, Logs, and Artifacts

*   **`data/raw/`**: Where the initial, untouched CSV lies.
*   **`data/processed/`**: Saves a snapshot of the dataframe right after feature engineering and aggregation. This `.csv` acts as a historical record to pull features from during Inference.
*   **`logs/`**: Text files auto-generated during runs containing all console outputs for debugging.
*   **`artifacts/`**: Secondary storage for things like text summaries (`model_summary.txt` of statsmodels) and saved Sklearn Scalers (`scaler.pkl`).
*   **`mlruns/`**: The local backend database and local artifact store for MLflow tracking. Contains hashed run IDs, metrics, params, and serialized model files.

---

## ⚙️ 3. Execution Flow Deep Dive

### Phase A: Training Pipeline Execution (`python main.py train`)

1.  **Bootstrap:** `main.py` parses CLI arguments (like `--config`), sets up logging via `logger_setup.py`, and calls `src.train.run_training()`.
2.  **MLflow Initialization:** `train.py` creates (or connects to) the `"Ad_Sales_TimeSeries"` experiment within the `mlruns` directory.
3.  **Data Ingestion & Formatting:** `load_raw_data()` reads the CSV. `aggregate_daily()` creates a strictly temporal dataset (1 row = 1 day).
4.  **Feature Generation:** `engineer_features()` expands the datset column width massively by attaching lag data, rolling averages, and calendar details.
5.  **Preprocessing & Splits:** `preprocessing.py` splits the timeline chronologically into Train (T), Validation (V), and Test (E) segments. It fits the Scaler on (T), applies it to (T), (V), and (E). 
6.  **Experiment Orchestration:**
    *   `run_arima_experiment()` loop begins: Evaluates grid search -> logs all runs to MLflow -> returns best ARIMA.
    *   `run_sarimax_experiment()` loop begins: Evaluates grid search -> logs all runs to MLflow -> returns best SARIMAX.
    *   `run_lstm_experiment()` loop begins: Evaluates deep-learning grid search -> logs all runs to MLflow -> returns best LSTM model.
7.  **Clash of the Titans (Model Selection):** `train.py` evaluates the returned 'best' from each family based strictly on overall Test RMSE.
8.  **Model Registration:** The absolute best run's artifact path is grabbed, and MLflow pushes this specific model state to the **Model Registry** under the name `"Ad_Sales_Forecaster"` and tags it with the **"Staging"** alias. Pipeline completes.

### Phase B: Inference Pipeline Execution (`python main.py infer`)

1.  **Bootstrap:** `main.py` updates the `forecast_days` inside `config.yaml` to match what the user requested at CLI (default 7), then invokes `src.inference.run_inference()`.
2.  **Environment Check:** Connects to MLflow. Queries the Model Registry for the model named `"Ad_Sales_Forecaster"` that currently holds the alias/stage of `"Staging"`.
3.  **Model Hydration:** Downloads and loads the registered model into memory into a generic `pyfunc` or native state.
4.  **Artifact Retrieval:** Re-loads the historical data from `data/processed/processed_ads.csv`. Loads the exact `scaler.pkl` that was associated with the active Staging run.
5.  **Data Window Slicing:** Extracts the precise most recent days required by the model to generate a forward prediction (e.g., the last 14 days of data if the sequence length is 14).
6.  **Forward Pass:** Scales the most recent slice > feeds it to model > outputs next-day scaled prediction. If forecasting 7 days, it often utilizes an autoregressive style decoding (appending Day 1's prediction as a feature to predict Day 2, etc.).
7.  **Inverse Transform:** Takes the output vector (which is scaled between 0-1) and passes it through `scaler.inverse_transform()` to yield real-world Dollar Revenue forecasts.
8.  **Output:** Prints the tabular DataFrame containing Future Dates and `predicted_revenue` to the CLI.

---

## 📈 4. The Power of the MLOps Lifecycle Explained

By packaging the code this way, we've moved away from "Jupyter Notebook ad-hoc scripts" into a production architecture:
*   **Total Reproducibility**: Because of random seed fixing, requirements.txt, and strict config files, running `main.py train` on any machine will yield the exact same parameters and the exact same model weights.
*   **Experiment Provenance**: Every time you train, MLflow records the Git Hash, the dataset Hash, every single hyperparameter tested, and the exact metrics. If a model breaks in production, you trace it back to the precise code state that birthed it.
*   **Seamless Hand-off**: The `main.py infer` script does not care *what* model is currently the best. It simply asks MLflow for the latest "Staging" model. This abstracts the data science component (training/tuning) away from the software engineering component (deployment/inference). If a new run discovers LSTM is better than SARIMAX, it registers the LSTM. On the very next inference run, the system automatically uses the LSTM natively—zero code change required.
