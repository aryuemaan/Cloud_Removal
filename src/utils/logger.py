"""
Centralized logging setup used across the pipeline (training, inference,
preprocessing, evaluation, API). Produces console + rotating file logs.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_dir: str | None = "logs",
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            Path(log_dir) / f"{name.replace('.', '_')}.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
