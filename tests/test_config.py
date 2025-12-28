"""Tests for configuration module."""

from collections.abc import Callable
from pathlib import Path

import pytest

from epistemon.config import load_config


@pytest.fixture
def temp_yaml_file(tmp_path: Path) -> Callable[[str], str]:
    """Create a temporary YAML file with given content and return its path."""

    def create_yaml_file(content: str) -> str:
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(content)
        return str(yaml_file)

    return create_yaml_file


def test_load_config_from_yaml_file(temp_yaml_file: Callable[[str], str]) -> None:
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
    config_path = temp_yaml_file(config_content)
    config = load_config(config_path)

    assert config.input_directory == "./test_docs"
    assert config.vector_store_type == "chroma"
    assert config.vector_store_path == "./test_data/chroma_db"
    assert config.embedding_provider == "huggingface"
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 200
    assert config.search_results_limit == 5


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


def test_load_config_with_empty_file_uses_defaults(
    temp_yaml_file: Callable[[str], str],
) -> None:
    """Test loading configuration from an empty file uses all defaults."""
    config_path = temp_yaml_file("")
    config = load_config(config_path)

    assert config.input_directory == "./tests/data"
    assert config.embedding_provider == "huggingface"
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.vector_store_type == "chroma"
    assert config.vector_store_path == "./data/chroma_db"
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 200
    assert config.search_results_limit == 5


def test_load_config_with_partial_override(
    temp_yaml_file: Callable[[str], str],
) -> None:
    """Test loading configuration with some fields overridden uses custom values for those fields and defaults for others."""
    config_content = """
input_directory: "./custom/path"
chunk_size: 500
"""
    config_path = temp_yaml_file(config_content)
    config = load_config(config_path)

    assert config.input_directory == "./custom/path"
    assert config.chunk_size == 500
    assert config.embedding_provider == "huggingface"
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.vector_store_type == "chroma"
    assert config.vector_store_path == "./data/chroma_db"
    assert config.chunk_overlap == 200
    assert config.search_results_limit == 5


def test_load_config_with_invalid_yaml_raises_error(
    temp_yaml_file: Callable[[str], str],
) -> None:
    """Test loading configuration with invalid YAML syntax raises an error."""
    config_content = """
input_directory: "./custom/path"
chunk_size: [invalid
"""
    config_path = temp_yaml_file(config_content)

    with pytest.raises(ValueError, match="Invalid YAML syntax"):
        load_config(config_path)


def test_load_config_with_missing_file_raises_error(tmp_path: Path) -> None:
    """Test loading configuration from a non-existent file raises an error."""
    non_existent_path = str(tmp_path / "this_file_does_not_exist.yaml")

    with pytest.raises(ValueError, match="Configuration file not found"):
        load_config(non_existent_path)
