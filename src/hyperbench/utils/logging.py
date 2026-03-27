# Author: Ritik Shah

"""Logging utilities for HyperBench."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


_DEFAULT_LOGGER_NAME = "hyperbench"


def get_logger(name: str = _DEFAULT_LOGGER_NAME) -> logging.Logger:
    """Return a HyperBench logger.

    If the logger has not been configured yet, a default stream handler is added.

    Parameters
    ----------
    name:
        Logger name.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        configure_logger(logger)

    return logger


def configure_logger(
    logger: logging.Logger | None = None,
    *,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    propagate: bool = False,
) -> logging.Logger:
    """Configure a HyperBench logger.

    Parameters
    ----------
    logger:
        Existing logger to configure. If None, uses the default HyperBench logger.
    level:
        Logging level, e.g. `logging.INFO` or `logging.DEBUG`.
    log_file:
        Optional file path for a file handler.
    propagate:
        Whether the logger should propagate to parent loggers.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logger or logging.getLogger(_DEFAULT_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = propagate

    # Avoid duplicate handlers if configure_logger is called multiple times.
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(Path(log_file), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_benchmark_start(logger: logging.Logger, *, num_cases: int, adapter_name: str) -> None:
    """Log the start of a benchmark run."""
    logger.info(
        "Starting benchmark run with %d case(s) using adapter '%s'.",
        num_cases,
        adapter_name,
    )


def log_case_start(logger: logging.Logger, case_id: str, metadata: dict[str, Any] | None = None) -> None:
    """Log the start of a benchmark case."""
    if metadata:
        logger.info("Starting %s | %s", case_id, _format_metadata(metadata))
    else:
        logger.info("Starting %s", case_id)


def log_case_success(
    logger: logging.Logger,
    case_id: str,
    *,
    runtime_seconds: float,
    metrics: dict[str, float] | None = None,
) -> None:
    """Log successful completion of a benchmark case."""
    if metrics:
        metric_text = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
        logger.info(
            "Completed %s successfully in %.3fs | %s",
            case_id,
            runtime_seconds,
            metric_text,
        )
    else:
        logger.info(
            "Completed %s successfully in %.3fs",
            case_id,
            runtime_seconds,
        )


def log_case_failure(
    logger: logging.Logger,
    case_id: str,
    *,
    runtime_seconds: float,
    error: Exception | str,
) -> None:
    """Log failed completion of a benchmark case."""
    logger.error(
        "Failed %s after %.3fs | %s",
        case_id,
        runtime_seconds,
        error,
    )


def _format_metadata(metadata: dict[str, Any]) -> str:
    """Format metadata into a compact key=value string."""
    parts = []
    for key, value in metadata.items():
        parts.append(f"{key}={value}")
    return ", ".join(parts)