"""Tests for logging configuration."""

import logging
from collections.abc import Generator

import pytest

from epistemon.logging_config import setup_logging


@pytest.fixture(autouse=True)
def cleanup_logging() -> Generator[None, None, None]:
    """Clean up logging configuration after each test."""
    yield
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)


def test_setup_logging_configures_root_logger() -> None:
    setup_logging()

    root_logger = logging.getLogger()

    assert root_logger.level == logging.INFO


def test_setup_logging_adds_handler_with_formatter() -> None:
    setup_logging()

    root_logger = logging.getLogger()

    assert len(root_logger.handlers) > 0
    handler = root_logger.handlers[0]
    assert handler.formatter is not None


def test_application_logger_inherits_configuration() -> None:
    setup_logging()

    app_logger = logging.getLogger("epistemon")

    assert app_logger.level == logging.NOTSET
    assert app_logger.parent is not None
    assert app_logger.parent.level == logging.INFO
