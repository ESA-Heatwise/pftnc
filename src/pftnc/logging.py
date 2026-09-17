import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure package-wide logger.

    Args:
        level: Logging level.

    Returns:
        Configured root package logger.
    """
    logger = logging.getLogger("pftnc")

    if logger.handlers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt=(
            "[%(asctime)s] [%(levelname)s] [%(name)s] "
            "[%(funcName)s:#%(lineno)d] "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # sample log:
    # [2026-06-17 15:36:39] [INFO] [pftnc.train.train] [_train_fold:42] Processing fold: fold_2015

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = log_dir / f"pftnc_{timestamp}.log"

    file_handler = logging.FileHandler(
        log_file,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(file_handler)
    logger.info("Logging to: %s", log_file.resolve())
    logger.propagate = False

    return logger
