"""
main.py
=======
CLI entry-point for the Ad Sales Forecasting project.

Usage
-----
  python main.py train [--config config.yaml] [--seed 42]
  python main.py infer [--config config.yaml]

Examples
--------
  # Full training pipeline
  python main.py train

  # Custom config
  python main.py train --config my_config.yaml --seed 123

  # Inference only (requires completed training)
  python main.py infer
"""

import logging
import sys

import click
import yaml

# --------------------------------------------------------------
# Minimal bootstrap logger (replaced per-command by setup_logging)
# --------------------------------------------------------------
logger = logging.getLogger("main")


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------

@click.group()
def cli():
    """Ad Sales Forecasting - MLOps Time Series Pipeline."""
    pass


# -- train -----------------------------------------------------

@cli.command()
@click.option(
    "--config", default="config.yaml", show_default=True,
    help="Path to config.yaml",
)
@click.option(
    "--seed", default=42, show_default=True, type=int,
    help="Global random seed for reproducibility.",
)
@click.option(
    "--model", default="all",
    type=click.Choice(["all", "arima", "sarimax", "lstm"], case_sensitive=False),
    show_default=True,
    help="Which model family to train ('all' trains all three).",
)
def train(config: str, seed: int, model: str) -> None:
    """Run the full training pipeline (ARIMA + SARIMAX + LSTM)."""
    from src.logger_setup import setup_logging
    root_logger = setup_logging(run_label="train")
    _logger = logging.getLogger("main")

    _logger.info("*" * 56)
    _logger.info("*   Ad Sales Forecasting - Training Pipeline           *")
    _logger.info("*" * 56)

    # Import here to avoid loading torch at CLI parse time
    from src.train import run_training
    run_training(config_path=config, seed=seed)


# -- infer -----------------------------------------------------

@cli.command()
@click.option(
    "--config", default="config.yaml", show_default=True,
    help="Path to config.yaml",
)
@click.option(
    "--days", default=7, show_default=True, type=int,
    help="Number of days to forecast into the future.",
)
def infer(config: str, days: int) -> None:
    """
    Run inference using the best registered model.

    Loads the latest Staging model and generates a forecast.
    """
    from src.logger_setup import setup_logging
    setup_logging(run_label="infer")
    _logger = logging.getLogger("main")

    _logger.info("*" * 56)
    _logger.info("*   Ad Sales Forecasting - Inference Pipeline          *")
    _logger.info("*" * 56)

    # Patch forecast_days from CLI arg
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["inference"]["forecast_days"] = days
    with open(config, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    from src.inference import run_inference
    predictions = run_inference(config_path=config)

    click.echo("\n" + "=" * 50)
    click.echo(predictions.to_string(index=False))
    click.echo("=" * 50)


# -- info ------------------------------------------------------

@cli.command()
@click.option("--config", default="config.yaml", show_default=True)
def info(config: str) -> None:
    """Display current project configuration."""
    with open(config) as f:
        cfg = yaml.safe_load(f)

    import torch
    click.echo("\n" + "=" * 55)
    click.echo("  Ad Sales Forecasting - Project Info")
    click.echo("=" * 55)
    click.echo(f"  Project      : {cfg['project']['name']}")
    click.echo(f"  Experiment   : {cfg['mlflow']['experiment_name']}")
    click.echo(f"  Registry     : {cfg['mlflow']['registry_model_name']}")
    click.echo(f"  Target col   : {cfg['data']['target_col']}")
    click.echo(f"  Train ratio  : {cfg['data']['train_ratio']}")
    click.echo(f"  Val ratio    : {cfg['data']['val_ratio']}")
    click.echo(f"  Test ratio   : {cfg['data']['test_ratio']}")
    click.echo(f"  CUDA avail.  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"  GPU          : {torch.cuda.get_device_name(0)}")
    click.echo("=" * 55 + "\n")


# --------------------------------------------------------------

if __name__ == "__main__":
    cli()
