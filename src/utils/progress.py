# -*- coding: utf-8 -*-
import logging
from logging import Logger
from pathlib import Path
from .paths import LOGS

def get_logger(name: str = "app", log_file: str = "run.log", level: int = logging.INFO) -> Logger:
    """
    Create/reuse a namespaced logger with console + file handlers.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_configured", False):
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S")

    # console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file
    LOGS.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(LOGS / log_file), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger._configured = True
    return logger
