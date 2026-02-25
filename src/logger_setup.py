"""
logger_setup.py
===============
Centralised logging configuration for the Ad Sales Forecasting project.

Features
--------
* Console handler (coloured by level via a custom Formatter).
* Rotating file handler - writes to logs/<timestamp>.log
* Auto-archiving - any existing logs that exceed MAX_LOG_FILES are
  moved to logs/archive/ before the new run begins.
* A plain-text run manifest is updated at logs/run_history.txt each
  time setup_logging() is called.

Usage
-----
    from src.logger_setup import setup_logging
    logger = setup_logging(run_label="train")   # call once at entry-point
    # All other modules just use:  logging.getLogger(__name__)
"""

import gzip
import io
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------
# Constants
# --------------------------------------------------------------

LOG_DIR     = Path("logs")
ARCHIVE_DIR = LOG_DIR / "archive"
MAX_LOG_FILES = 5          # keep at most this many active logs before archiving

LOG_FORMAT  = "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ANSI colour codes for console output
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


# --------------------------------------------------------------
# Formatters
# --------------------------------------------------------------

class ColouredConsoleFormatter(logging.Formatter):
    """Adds ANSI colour codes around the level-name for terminals."""

    def format(self, record: logging.LogRecord) -> str:
        colour   = _COLOURS.get(record.levelname, "")
        original = record.levelname
        record.levelname = f"{colour}{record.levelname}{_RESET}"
        result   = super().format(record)
        record.levelname = original          # restore for other handlers
        return result


class PlainFileFormatter(logging.Formatter):
    """Plain formatter (no ANSI) for log files."""
    pass


# --------------------------------------------------------------
# Archive helper
# --------------------------------------------------------------

def _archive_old_logs() -> None:
    """
    If more than MAX_LOG_FILES .log files exist in LOG_DIR,
    compress the oldest ones and move them to ARCHIVE_DIR.

    Old run logs are gzip-compressed to save space.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Only .log files (not sub-dirs, not the archive folder)
    existing = sorted(
        [p for p in LOG_DIR.glob("*.log")],
        key=lambda p: p.stat().st_mtime   # oldest first
    )

    overflow = len(existing) - MAX_LOG_FILES + 1   # +1 to make room for new file
    if overflow <= 0:
        return

    for log_file in existing[:overflow]:
        archive_name = ARCHIVE_DIR / (log_file.name + ".gz")
        with open(log_file, "rb") as f_in, gzip.open(archive_name, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        log_file.unlink()
        logging.getLogger("logger_setup").info(
            f"Archived old log -> {archive_name.name}"
        )


# --------------------------------------------------------------
# Run history manifest
# --------------------------------------------------------------

def _update_run_history(log_file: Path, run_label: str) -> None:
    """Append a line to logs/run_history.txt."""
    history_file = LOG_DIR / "run_history.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(f"{ts}  [{run_label:>10s}]  -> {log_file.name}\n")


# --------------------------------------------------------------
# Public API
# --------------------------------------------------------------

def setup_logging(
    run_label:   str   = "run",
    log_level:   int   = logging.INFO,
    log_dir:     str   = str(LOG_DIR),
    max_files:   int   = MAX_LOG_FILES,
) -> logging.Logger:
    """
    Initialise project-wide logging with console + rotating file output.

    Call this **once** at the start of main.py / train.py / inference.py.
    All other modules should simply call logging.getLogger(__name__).

    Parameters
    ----------
    run_label : Short label for this run (e.g. "train", "infer").
                Embedded in the log filename.
    log_level : Logging threshold (default INFO).
    log_dir   : Directory for log files (default "logs/").
    max_files : Maximum active log files before archiving older ones.

    Returns
    -------
    Root logger configured with both handlers.
    """
    global MAX_LOG_FILES
    MAX_LOG_FILES = max_files

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # -- Archive surplus logs before starting -----------------
    _archive_old_logs()

    # -- New timestamped log file ------------------------------
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{ts}_{run_label}.log"

    # -- Root logger -------------------------------------------
    root = logging.getLogger()
    root.setLevel(log_level)

    # Clear any pre-existing handlers (idempotent re-init)
    root.handlers.clear()

    # -- Console handler (coloured, UTF-8 safe) ----------------
    # Wrap stdout in a UTF-8 stream that replaces unencodable chars
    # instead of raising UnicodeEncodeError on Windows cp1252 terminals.
    try:
        safe_stream = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    except AttributeError:
        # Fallback: stdout has no .buffer (e.g. IDLE, Jupyter)
        safe_stream = sys.stdout

    ch = logging.StreamHandler(stream=safe_stream)
    ch.setLevel(log_level)
    ch.setFormatter(ColouredConsoleFormatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(ch)

    # -- File handler (plain) ----------------------------------
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(PlainFileFormatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(fh)

    # -- Write run manifest ------------------------------------
    _update_run_history(log_file, run_label)

    logger = logging.getLogger("logger_setup")
    logger.info(f"Logging initialised -> file: {log_file}")
    logger.info(f"Archive dir         -> {ARCHIVE_DIR.resolve()}")

    return root
