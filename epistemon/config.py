"""Configuration module for Epistemon."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Optional

import yaml

VALID_EMBEDDING_PROVIDERS: Final[list[str]] = ["fake", "huggingface", "openai"]
VALID_VECTOR_STORE_TYPES: Final[list[str]] = [
    "inmemory",
    "chroma",
    "weaviate",
    "duckdb",
    "qdrant",
]
VALID_LLM_PROVIDERS: Final[list[str]] = ["fake", "openai"]


@dataclass(frozen=True)
class Configuration:
    """Configuration for the Epistemon semantic search engine."""

    input_directory: str
    vector_store_type: str
    vector_store_path: str
    embedding_provider: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    search_results_limit: int
    score_threshold: float
    bm25_k1: float
    bm25_b: float
    bm25_top_k: int
    hybrid_bm25_weight: float
    hybrid_semantic_weight: float
    llm_provider: str
    llm_model: str
    llm_temperature: float
    rag_enabled: bool
    rag_max_context_docs: int
    rag_prompt_template_path: str


def load_config(config_path: Optional[str] = None) -> Configuration:
    """Load configuration from a YAML file or use defaults."""
    defaults = {
        "input_directory": "./tests/data",
        "embedding_provider": "huggingface",
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_store_type": "chroma",
        "vector_store_path": "./data/chroma_db",
        "chunk_size": 500,
        "chunk_overlap": 200,
        "search_results_limit": 5,
        "score_threshold": 0.0,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "bm25_top_k": 5,
        "hybrid_bm25_weight": 0.3,
        "hybrid_semantic_weight": 0.7,
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "llm_temperature": 0.0,
        "rag_enabled": True,
        "rag_max_context_docs": 10,
        "rag_prompt_template_path": "./prompts/rag_answer_prompt.txt",
    }

    config_data: dict[str, Any]
    if config_path is None:
        config_data = {}
    else:
        config_file = Path(config_path)
        try:
            with config_file.open("r") as file:
                loaded_data = yaml.safe_load(file)
                config_data = loaded_data if loaded_data is not None else {}
        except FileNotFoundError as e:
            raise ValueError(f"Configuration file not found: {config_path}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {config_path}: {e}") from e

    merged_config = {**defaults, **config_data}

    string_fields = [
        "input_directory",
        "vector_store_type",
        "vector_store_path",
        "embedding_provider",
        "embedding_model",
        "llm_provider",
        "llm_model",
        "rag_prompt_template_path",
    ]
    for field in string_fields:
        if not isinstance(merged_config[field], str):
            raise ValueError(
                f"{field} must be a string, got {type(merged_config[field]).__name__}"
            )

    integer_fields = [
        "chunk_size",
        "chunk_overlap",
        "search_results_limit",
        "bm25_top_k",
        "rag_max_context_docs",
    ]
    for field in integer_fields:
        if not isinstance(merged_config[field], int):
            raise ValueError(
                f"{field} must be an integer, got {type(merged_config[field]).__name__}"
            )

    float_fields = [
        "score_threshold",
        "bm25_k1",
        "bm25_b",
        "hybrid_bm25_weight",
        "hybrid_semantic_weight",
        "llm_temperature",
    ]
    for field in float_fields:
        if not isinstance(merged_config[field], (int, float)):
            raise ValueError(
                f"{field} must be a number, got {type(merged_config[field]).__name__}"
            )
        merged_config[field] = float(merged_config[field])

    if not isinstance(merged_config["rag_enabled"], bool):
        raise ValueError(
            f"rag_enabled must be a boolean, got {type(merged_config['rag_enabled']).__name__}"
        )

    if merged_config["embedding_provider"] not in VALID_EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Invalid embedding_provider: {merged_config['embedding_provider']}. "
            f"Must be one of: {', '.join(VALID_EMBEDDING_PROVIDERS)}"
        )

    if merged_config["vector_store_type"] not in VALID_VECTOR_STORE_TYPES:
        raise ValueError(
            f"Invalid vector_store_type: {merged_config['vector_store_type']}. "
            f"Must be one of: {', '.join(VALID_VECTOR_STORE_TYPES)}"
        )

    if merged_config["llm_provider"] not in VALID_LLM_PROVIDERS:
        raise ValueError(
            f"Invalid llm_provider: {merged_config['llm_provider']}. "
            f"Must be one of: {', '.join(VALID_LLM_PROVIDERS)}"
        )

    if merged_config["chunk_size"] <= 0:
        raise ValueError(
            f"chunk_size must be positive, got: {merged_config['chunk_size']}"
        )

    if merged_config["chunk_overlap"] < 0:
        raise ValueError(
            f"chunk_overlap must be positive, got: {merged_config['chunk_overlap']}"
        )

    if merged_config["search_results_limit"] <= 0:
        raise ValueError(
            f"search_results_limit must be positive, got: {merged_config['search_results_limit']}"
        )

    if merged_config["score_threshold"] < 0:
        raise ValueError(
            f"score_threshold must be non-negative, got: {merged_config['score_threshold']}"
        )

    if merged_config["bm25_k1"] < 0:
        raise ValueError(
            f"bm25_k1 must be non-negative, got: {merged_config['bm25_k1']}"
        )

    if merged_config["bm25_b"] < 0 or merged_config["bm25_b"] > 1:
        raise ValueError(
            f"bm25_b must be between 0 and 1, got: {merged_config['bm25_b']}"
        )

    if merged_config["bm25_top_k"] <= 0:
        raise ValueError(
            f"bm25_top_k must be positive, got: {merged_config['bm25_top_k']}"
        )

    if merged_config["llm_temperature"] < 0 or merged_config["llm_temperature"] > 1:
        raise ValueError(
            f"llm_temperature must be between 0 and 1, got: {merged_config['llm_temperature']}"
        )

    if merged_config["rag_max_context_docs"] <= 0:
        raise ValueError(
            f"rag_max_context_docs must be positive, got: {merged_config['rag_max_context_docs']}"
        )

    if (
        merged_config["hybrid_bm25_weight"] < 0
        or merged_config["hybrid_bm25_weight"] > 1
    ):
        raise ValueError(
            f"hybrid_bm25_weight must be between 0 and 1, got: {merged_config['hybrid_bm25_weight']}"
        )

    if (
        merged_config["hybrid_semantic_weight"] < 0
        or merged_config["hybrid_semantic_weight"] > 1
    ):
        raise ValueError(
            f"hybrid_semantic_weight must be between 0 and 1, got: {merged_config['hybrid_semantic_weight']}"
        )

    return Configuration(
        input_directory=merged_config["input_directory"],
        embedding_provider=merged_config["embedding_provider"],
        embedding_model=merged_config["embedding_model"],
        vector_store_type=merged_config["vector_store_type"],
        vector_store_path=merged_config["vector_store_path"],
        chunk_size=merged_config["chunk_size"],
        chunk_overlap=merged_config["chunk_overlap"],
        search_results_limit=merged_config["search_results_limit"],
        score_threshold=merged_config["score_threshold"],
        bm25_k1=merged_config["bm25_k1"],
        bm25_b=merged_config["bm25_b"],
        bm25_top_k=merged_config["bm25_top_k"],
        hybrid_bm25_weight=merged_config["hybrid_bm25_weight"],
        hybrid_semantic_weight=merged_config["hybrid_semantic_weight"],
        llm_provider=merged_config["llm_provider"],
        llm_model=merged_config["llm_model"],
        llm_temperature=merged_config["llm_temperature"],
        rag_enabled=merged_config["rag_enabled"],
        rag_max_context_docs=merged_config["rag_max_context_docs"],
        rag_prompt_template_path=merged_config["rag_prompt_template_path"],
    )
