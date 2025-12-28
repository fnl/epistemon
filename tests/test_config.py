"""Tests for configuration module."""

import tempfile
from pathlib import Path

from epistemon.config import load_config


def test_load_config_from_yaml_file() -> None:
    """Test loading configuration from a YAML file."""
    config_content = """
input_directory: "./test_docs"
vector_store_type: "chroma"
vector_store_path: "./test_data/chroma_db"
embedding_provider: "huggingface"
embedding_model: "all-MiniLM-L6-v2"
chunk_size: 1000
chunk_overlap: 200
search_results_limit: 5
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as temp_file:
        temp_file.write(config_content)
        temp_file_path = temp_file.name

    try:
        config = load_config(temp_file_path)

        assert config.input_directory == "./test_docs"
        assert config.vector_store_type == "chroma"
        assert config.vector_store_path == "./test_data/chroma_db"
        assert config.embedding_provider == "huggingface"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.search_results_limit == 5
    finally:
        Path(temp_file_path).unlink()


def test_load_config_without_file_uses_defaults() -> None:
    """Test loading configuration without a file uses all default values."""
    config = load_config()

    assert config.input_directory == "./tests/data"
    assert config.embedding_provider == "huggingface"
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.vector_store_type == "chroma"
    assert config.vector_store_path == "./data/chroma_db"
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 200
    assert config.search_results_limit == 5


def test_load_config_with_empty_file_uses_defaults() -> None:
    """Test loading configuration from an empty file uses all defaults."""
    config_content = ""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as temp_file:
        temp_file.write(config_content)
        temp_file_path = temp_file.name

    try:
        config = load_config(temp_file_path)

        assert config.input_directory == "./tests/data"
        assert config.embedding_provider == "huggingface"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.vector_store_type == "chroma"
        assert config.vector_store_path == "./data/chroma_db"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.search_results_limit == 5
    finally:
        Path(temp_file_path).unlink()
