"""Configuration module for Epistemon."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


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


def load_config(config_path: Optional[str] = None) -> Configuration:
    """Load configuration from a YAML file or use defaults."""
    defaults = {
        "input_directory": "./tests/data",
        "embedding_provider": "huggingface",
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_store_type": "chroma",
        "vector_store_path": "./data/chroma_db",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "search_results_limit": 5,
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

    return Configuration(
        input_directory=merged_config["input_directory"],
        embedding_provider=merged_config["embedding_provider"],
        embedding_model=merged_config["embedding_model"],
        vector_store_type=merged_config["vector_store_type"],
        vector_store_path=merged_config["vector_store_path"],
        chunk_size=merged_config["chunk_size"],
        chunk_overlap=merged_config["chunk_overlap"],
        search_results_limit=merged_config["search_results_limit"],
    )
