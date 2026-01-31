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
    assert config.chunk_size == 500
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
    assert config.chunk_size == 500
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


def test_load_config_with_invalid_embedding_provider_raises_error(
    temp_yaml_file: Callable[[str], str],
) -> None:
    """Test loading configuration with invalid embedding_provider raises an error."""
    config_content = """
embedding_provider: "invalid_provider"
"""
    config_path = temp_yaml_file(config_content)

    with pytest.raises(
        ValueError,
        match="Invalid embedding_provider.*Must be one of: fake, huggingface, openai",
    ):
        load_config(config_path)


def test_load_config_with_invalid_vector_store_type_raises_error(
    temp_yaml_file: Callable[[str], str],
) -> None:
    """Test loading configuration with invalid vector_store_type raises an error."""
    config_content = """
vector_store_type: "invalid_store"
"""
    config_path = temp_yaml_file(config_content)

    with pytest.raises(
        ValueError,
        match=r"Invalid vector_store_type.*Must be one of: inmemory, chroma, weaviate, duckdb, qdrant",
    ):
        load_config(config_path)


def test_load_config_with_negative_chunk_size_raises_error(
    temp_yaml_file: Callable[[str], str],
) -> None:
    """Test loading configuration with negative chunk_size raises an error."""
    config_content = """
chunk_size: -100
"""
    config_path = temp_yaml_file(config_content)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        load_config(config_path)


def test_load_config_with_zero_chunk_size_raises_error(
    temp_yaml_file: Callable[[str], str],
) -> None:
    """Test loading configuration with zero chunk_size raises an error."""
    config_content = """
chunk_size: 0
"""
    config_path = temp_yaml_file(config_content)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        load_config(config_path)


def test_load_config_with_negative_chunk_overlap_raises_error(
    temp_yaml_file: Callable[[str], str],
) -> None:
    """Test loading configuration with negative chunk_overlap raises an error."""
    config_content = """
chunk_overlap: -50
"""
    config_path = temp_yaml_file(config_content)

    with pytest.raises(ValueError, match="chunk_overlap must be positive"):
        load_config(config_path)


def test_load_config_with_negative_search_results_limit_raises_error(
    temp_yaml_file: Callable[[str], str],
) -> None:
    """Test loading configuration with negative search_results_limit raises an error."""
    config_content = """
search_results_limit: -5
"""
    config_path = temp_yaml_file(config_content)

    with pytest.raises(ValueError, match="search_results_limit must be positive"):
        load_config(config_path)


def test_configuration_is_immutable() -> None:
    """Test that Configuration instances are immutable."""
    config = load_config()

    with pytest.raises(AttributeError):
        config.chunk_size = 2000  # type: ignore[misc]


def test_load_config_with_string_chunk_size_raises_error(
    temp_yaml_file: Callable[[str], str],
) -> None:
    """Test loading configuration with chunk_size as string raises an error."""
    config_content = """
chunk_size: "500"
"""
    config_path = temp_yaml_file(config_content)

    with pytest.raises(ValueError, match="chunk_size must be an integer"):
        load_config(config_path)


def test_load_config_with_integer_input_directory_raises_error(
    temp_yaml_file: Callable[[str], str],
) -> None:
    """Test loading configuration with input_directory as integer raises an error."""
    config_content = """
input_directory: 123
"""
    config_path = temp_yaml_file(config_content)

    with pytest.raises(ValueError, match="input_directory must be a string"):
        load_config(config_path)


@pytest.mark.parametrize(
    "vector_store_type",
    ["inmemory", "chroma", "weaviate", "duckdb", "qdrant"],
)
def test_load_config_accepts_all_valid_vector_store_types(
    temp_yaml_file: Callable[[str], str], vector_store_type: str
) -> None:
    """Test that all valid vector store types are accepted."""
    config_content = f"""
vector_store_type: "{vector_store_type}"
"""
    config_path = temp_yaml_file(config_content)
    config = load_config(config_path)
    assert config.vector_store_type == vector_store_type


def test_load_config_includes_bm25_defaults() -> None:
    """Test that configuration includes BM25 defaults."""
    config = load_config()

    assert config.bm25_k1 == 1.5
    assert config.bm25_b == 0.75
    assert config.bm25_top_k == 5


def test_load_config_allows_bm25_override(
    temp_yaml_file: Callable[[str], str],
) -> None:
    """Test that BM25 configuration can be overridden."""
    config_content = """
bm25_k1: 2.0
bm25_b: 0.5
bm25_top_k: 10
"""
    config_path = temp_yaml_file(config_content)
    config = load_config(config_path)

    assert config.bm25_k1 == 2.0
    assert config.bm25_b == 0.5
    assert config.bm25_top_k == 10


def test_load_config_includes_llm_defaults() -> None:
    """Test that configuration includes LLM defaults."""
    config = load_config()

    assert config.llm_provider == "openai"
    assert config.llm_model == "gpt-4o-mini"
    assert config.llm_temperature == 0.0
    assert config.rag_enabled is True
    assert config.rag_max_context_docs == 10


def test_load_config_allows_llm_override(
    temp_yaml_file: Callable[[str], str],
) -> None:
    """Test that LLM configuration can be overridden."""
    config_content = """
llm_provider: "fake"
llm_model: "test-model"
llm_temperature: 0.7
rag_enabled: false
rag_max_context_docs: 20
"""
    config_path = temp_yaml_file(config_content)
    config = load_config(config_path)

    assert config.llm_provider == "fake"
    assert config.llm_model == "test-model"
    assert config.llm_temperature == 0.7
    assert config.rag_enabled is False
    assert config.rag_max_context_docs == 20


def test_load_config_includes_hybrid_search_weight_defaults() -> None:
    """Test that configuration includes hybrid search weight defaults."""
    config = load_config()

    assert config.hybrid_bm25_weight == 0.3
    assert config.hybrid_semantic_weight == 0.7


def test_load_config_allows_hybrid_search_weight_override(
    temp_yaml_file: Callable[[str], str],
) -> None:
    """Test that hybrid search weights can be overridden."""
    config_content = """
hybrid_bm25_weight: 0.4
hybrid_semantic_weight: 0.6
"""
    config_path = temp_yaml_file(config_content)
    config = load_config(config_path)

    assert config.hybrid_bm25_weight == 0.4
    assert config.hybrid_semantic_weight == 0.6


def test_load_config_includes_rag_prompt_template_path_default() -> None:
    """Test that configuration includes RAG prompt template path default."""
    config = load_config()

    assert config.rag_prompt_template_path == "./prompts/rag_answer_prompt.txt"


def test_load_config_allows_rag_prompt_template_path_override(
    temp_yaml_file: Callable[[str], str],
) -> None:
    """Test that RAG prompt template path can be overridden."""
    config_content = """
rag_prompt_template_path: "./prompts/custom.txt"
"""
    config_path = temp_yaml_file(config_content)
    config = load_config(config_path)

    assert config.rag_prompt_template_path == "./prompts/custom.txt"


def test_load_config_with_openai_embedding_provider_requires_api_key(
    temp_yaml_file: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that using OpenAI embedding provider without API key raises an error."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_content = """
embedding_provider: "openai"
"""
    config_path = temp_yaml_file(config_content)

    with pytest.raises(
        ValueError,
        match="OPENAI_API_KEY environment variable is required when using openai embedding provider",
    ):
        load_config(config_path)


def test_load_config_with_openai_llm_provider_requires_api_key(
    temp_yaml_file: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that using OpenAI LLM provider without API key raises an error."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_content = """
llm_provider: "openai"
"""
    config_path = temp_yaml_file(config_content)

    with pytest.raises(
        ValueError,
        match="OPENAI_API_KEY environment variable is required when using openai LLM provider",
    ):
        load_config(config_path)


def test_load_config_tracing_disabled_by_default() -> None:
    """Test that tracing_enabled defaults to False when not specified."""
    config = load_config()

    assert config.tracing_enabled is False


def test_load_config_tracing_enabled_without_secret_key_raises_error(
    temp_yaml_file: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that enabling tracing without LANGFUSE_SECRET_KEY raises an error."""
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    config_content = """
tracing_enabled: true
llm_provider: "fake"
"""
    config_path = temp_yaml_file(config_content)

    with pytest.raises(ValueError, match="LANGFUSE_SECRET_KEY"):
        load_config(config_path)


def test_load_config_tracing_enabled_without_public_key_raises_error(
    temp_yaml_file: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that enabling tracing without LANGFUSE_PUBLIC_KEY raises an error."""
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    config_content = """
tracing_enabled: true
llm_provider: "fake"
"""
    config_path = temp_yaml_file(config_content)

    with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
        load_config(config_path)


def test_load_config_tracing_enabled_with_valid_keys_succeeds(
    temp_yaml_file: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that enabling tracing with both LangFuse keys loads successfully."""
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    config_content = """
tracing_enabled: true
llm_provider: "fake"
"""
    config_path = temp_yaml_file(config_content)
    config = load_config(config_path)

    assert config.tracing_enabled is True


def test_load_config_with_openai_providers_and_valid_api_key_succeeds(
    temp_yaml_file: Callable[[str], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that using OpenAI providers with API key succeeds."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
    config_content = """
embedding_provider: "openai"
llm_provider: "openai"
"""
    config_path = temp_yaml_file(config_content)
    config = load_config(config_path)

    assert config.embedding_provider == "openai"
    assert config.llm_provider == "openai"
