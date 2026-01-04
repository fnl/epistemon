"""Tests for CLI commands."""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from epistemon.cli import main, upsert_index_command, web_ui_command, web_ui_main
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
        bm25_k1=1.5,
        bm25_b=0.75,
        bm25_top_k=5,
        hybrid_bm25_weight=0.3,
        hybrid_semantic_weight=0.7,
        llm_provider="fake",
        llm_model="fake-model",
        llm_temperature=0.0,
        rag_enabled=False,
        rag_max_context_docs=10,
        rag_prompt_template_path="./prompts/rag_answer_prompt.txt",
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
        bm25_k1=1.5,
        bm25_b=0.75,
        bm25_top_k=5,
        hybrid_bm25_weight=0.3,
        hybrid_semantic_weight=0.7,
        llm_provider="fake",
        llm_model="fake-model",
        llm_temperature=0.0,
        rag_enabled=False,
        rag_max_context_docs=10,
        rag_prompt_template_path="./prompts/rag_answer_prompt.txt",
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


def test_web_ui_command_starts_server(test_config: Configuration) -> None:
    with (
        patch("epistemon.cli.load_config", return_value=test_config),
        patch("epistemon.cli.create_vector_store") as mock_create_store,
        patch("epistemon.cli.create_app") as mock_create_app,
        patch("epistemon.cli.create_vector_store_manager") as mock_create_manager,
        patch("epistemon.cli.uvicorn.run") as mock_uvicorn_run,
    ):
        mock_vector_store = Mock()
        mock_create_store.return_value = mock_vector_store
        mock_app = Mock()
        mock_create_app.return_value = mock_app
        mock_manager = Mock()
        mock_create_manager.return_value = mock_manager

        web_ui_command(None, "127.0.0.1", 8000)

        mock_create_store.assert_called_once_with(test_config)
        mock_create_app.assert_called_once()
        call_args = mock_create_app.call_args[0]
        call_kwargs = mock_create_app.call_args[1]
        assert call_args[0] == mock_vector_store
        assert call_kwargs["base_url"] == "http://127.0.0.1:8000/files"
        assert call_kwargs["score_threshold"] == test_config.score_threshold
        assert call_kwargs["files_directory"] == Path(test_config.input_directory)
        assert call_kwargs["vector_store_manager"] == mock_manager
        mock_uvicorn_run.assert_called_once_with(mock_app, host="127.0.0.1", port=8000)


def test_web_ui_main_function_calls_command_without_args() -> None:
    with (
        patch("sys.argv", ["web-ui"]),
        patch("epistemon.cli.web_ui_command") as mock_command,
    ):
        web_ui_main()
        mock_command.assert_called_once_with(None, "127.0.0.1", 8000)


def test_web_ui_main_function_calls_command_with_args() -> None:
    with (
        patch(
            "sys.argv",
            [
                "web-ui",
                "--config",
                "custom.yaml",
                "--host",
                "0.0.0.0",  # noqa: S104
                "--port",
                "9000",
            ],
        ),
        patch("epistemon.cli.web_ui_command") as mock_command,
    ):
        web_ui_main()
        mock_command.assert_called_once_with(
            "custom.yaml", "0.0.0.0", 9000  # noqa: S104
        )


def test_web_ui_command_creates_bm25_indexer(test_config: Configuration) -> None:
    with (
        patch("epistemon.cli.load_config", return_value=test_config),
        patch("epistemon.cli.create_vector_store") as mock_create_store,
        patch("epistemon.cli.BM25Indexer") as mock_bm25_indexer,
        patch("epistemon.cli.create_app") as mock_create_app,
        patch("epistemon.cli.create_vector_store_manager") as mock_create_manager,
        patch("epistemon.cli.uvicorn.run"),
    ):
        mock_vector_store = Mock()
        mock_create_store.return_value = mock_vector_store
        mock_bm25_instance = Mock()
        mock_bm25_indexer.return_value = mock_bm25_instance
        mock_app = Mock()
        mock_create_app.return_value = mock_app
        mock_manager = Mock()
        mock_create_manager.return_value = mock_manager

        web_ui_command(None, "127.0.0.1", 8000)

        mock_bm25_indexer.assert_called_once_with(
            Path(test_config.input_directory),
            chunk_size=test_config.chunk_size,
            chunk_overlap=test_config.chunk_overlap,
        )
        call_kwargs = mock_create_app.call_args[1]
        assert call_kwargs["bm25_retriever"] == mock_bm25_instance
