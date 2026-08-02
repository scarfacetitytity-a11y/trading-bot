"""Shared logging configuration for the trading bot."""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Windows console defaults to cp1252 — unicode in log messages (→, —, ✓)
# raises UnicodeEncodeError inside the logging machinery otherwise.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def setup_logger(
    name: str,
    log_dir: str = "logs",
    log_file: str = "data_pipeline.log",
    level: str = "INFO",
) -> logging.Logger:
    """Return a logger that writes to both the console and a log file.

    Calling this multiple times with the same `name` returns the same
    configured logger without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = PROJECT_ROOT / log_dir
    log_path.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path / log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
