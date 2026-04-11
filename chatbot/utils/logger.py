"""
Centralized logging configuration.
PRESERVED EXACTLY from FastAPI version.
"""

import logging
import sys


def setup_logger(name: str = "fodwa", level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a configured logger.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Create default app logger
app_logger = setup_logger()
