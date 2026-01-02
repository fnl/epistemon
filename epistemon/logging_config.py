"""Logging configuration for Epistemon."""

import logging
import sys


class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes to log messages."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        name = record.name
        color = self.COLORS.get(levelname, "")

        record.levelname = f"{color}{levelname}{self.RESET}:"
        record.name = f"{self.BOLD}{name}{self.RESET}"

        formatted = super().format(record)

        record.levelname = levelname
        record.name = name

        return formatted


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with colored output matching uvicorn's style.

    Args:
        level: Logging level (default: logging.INFO)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.handlers:
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = ColoredFormatter(
        fmt="%(levelname)s     %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)
