"""Tests for CLI commands."""

import logging
from unittest.mock import Mock, patch

import pytest

from epistemon.cli import main, upsert_index_command
from epistemon.config import Configuration


@pytest.fixture
def test_config() -> Configuration:
    return Configuration(
        input_directory="./tests/data",
        embedding_provider="fake",
        embedding_model="fake-model",
        vector_store_type="inmemory",
        vector_store_path="./data/chroma_db",
        chunk_size=500,
        chunk_overlap=200,
        search_results_limit=5,
        score_threshold=0.0,
    )


@pytest.fixture
def chroma_config() -> Configuration:
    return Configuration(
        input_directory="./tests/data",
        embedding_provider="huggingface",
        embedding_model="all-MiniLM-L6-v2",
        vector_store_type="chroma",
        vector_store_path="./data/chroma_db",
        chunk_size=500,
        chunk_overlap=200,
        search_results_limit=5,
        score_threshold=0.0,
    )


def test_upsert_index_command_runs_indexing(test_config: Configuration) -> None:
    with (
        patch("epistemon.cli.load_config", return_value=test_config),
        patch("epistemon.cli.create_vector_store") as mock_create_store,
        patch("epistemon.cli.index") as mock_index,
    ):
        mock_vector_store = Mock()
        mock_create_store.return_value = mock_vector_store

        upsert_index_command(None)

        mock_create_store.assert_called_once_with(test_config)
        mock_index.assert_called_once_with(test_config, mock_vector_store)


def test_upsert_index_command_logs_progress_messages(
    chroma_config: Configuration, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        patch("epistemon.cli.load_config", return_value=chroma_config),
        patch("epistemon.cli.create_vector_store") as mock_create_store,
        patch("epistemon.cli.index"),
        caplog.at_level(logging.INFO),
    ):
        mock_vector_store = Mock()
        mock_create_store.return_value = mock_vector_store

        upsert_index_command(None)

        log_messages = " ".join(record.message for record in caplog.records)
        assert "Loading configuration" in log_messages
        assert "vector store" in log_messages
        assert "chroma" in log_messages
        assert "Indexing" in log_messages
        assert "./tests/data" in log_messages


def test_upsert_index_command_handles_config_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch("epistemon.cli.load_config", side_effect=ValueError("Invalid config")),
        patch("sys.exit") as mock_exit,
        caplog.at_level(logging.ERROR),
    ):
        upsert_index_command("invalid.yaml")

        mock_exit.assert_called_once_with(1)
        log_messages = " ".join(record.message for record in caplog.records)
        assert "Upsert failed" in log_messages
        assert "Invalid config" in log_messages


def test_upsert_index_command_handles_indexing_errors(
    test_config: Configuration, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        patch("epistemon.cli.load_config", return_value=test_config),
        patch("epistemon.cli.create_vector_store") as mock_create_store,
        patch("epistemon.cli.index", side_effect=Exception("Indexing failed")),
        patch("sys.exit") as mock_exit,
        caplog.at_level(logging.ERROR),
    ):
        mock_vector_store = Mock()
        mock_create_store.return_value = mock_vector_store

        upsert_index_command(None)

        mock_exit.assert_called_once_with(1)
        log_messages = " ".join(record.message for record in caplog.records)
        assert "Upsert failed" in log_messages
        assert "Indexing failed" in log_messages


def test_main_function_calls_command_without_args() -> None:
    with (
        patch("sys.argv", ["upsert-index"]),
        patch("epistemon.cli.upsert_index_command") as mock_command,
    ):
        main()
        mock_command.assert_called_once_with(None)


def test_main_function_calls_command_with_config_path() -> None:
    with (
        patch("sys.argv", ["upsert-index", "--config", "custom.yaml"]),
        patch("epistemon.cli.upsert_index_command") as mock_command,
    ):
        main()
        mock_command.assert_called_once_with("custom.yaml")
