"""logging module

Ref:
    * [1] https://icebee.hatenablog.com/entry/2018/12/16/221533

"""
from __future__ import annotations

from logging import DEBUG, INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path


def setup_logger(exp_version: str, log_path: Path) -> Logger:
    log_file = (log_path / f"{exp_version}.log").resolve()

    logger = getLogger(exp_version, mode="w")
    logger.setLevel(DEBUG)

    formatter = Formatter("[%(levelname)s] %(asctime)s >> \t%(message)s")
    file_handler = FileHandler(log_file)
    file_handler.setFormatter(formatter)

    sream_handler = StreamHandler()
    sream_handler.setLevel(INFO)
    sream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(formatter)
    return logger


def get_logger(exp_version: str) -> Logger:
    return getLogger(exp_version)
