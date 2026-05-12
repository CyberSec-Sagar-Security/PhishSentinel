"""
PhishLens Structured Logging Module.

Wraps loguru for consistent, coloured, structured log output across all
PhishLens modules. Provides both file and stderr sinks with rotation.
Security rationale: All ML pipeline errors, API failures, and feature
extraction warnings are logged with context to aid post-incident diagnosis.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def configure_logger(
    log_dir: str = "logs",
    log_file: str = "phishlens.log",
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "14 days",
) -> None:
    """Configure PhishLens logger with file and stderr sinks.

    Args:
        log_dir: Directory for log files.
        log_file: Log filename.
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        rotation: Log rotation threshold (e.g., "10 MB", "1 day").
        retention: How long to keep rotated log files.
    """
    # Remove default handler
    logger.remove()

    # Stderr handler — coloured, human-readable
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — structured, rotated
    # On HF Spaces the working directory is read-only for some paths;
    # fall back to stderr-only if the log directory cannot be created.
    log_path = Path(log_dir)
    try:
        log_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Running in a restricted container (e.g. HF Spaces) — skip file logging
        return
    logger.add(
        str(log_path / log_file),
        level="DEBUG",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} — {message}"
        ),
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=True,       # Thread-safe async writes
    )


def get_logger(name: str):
    """Return a named loguru logger bound to the calling module.

    Args:
        name: Module name, typically `__name__`.

    Returns:
        Loguru logger instance with bound context.
    """
    return logger.bind(module=name)


# Apply default configuration on import
configure_logger()

__all__ = ["logger", "get_logger", "configure_logger"]
