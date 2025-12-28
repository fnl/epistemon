"""Configuration module for Epistemon."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    if config_path is None:
        return Configuration(
            input_directory="./tests/data",
            embedding_provider="huggingface",
            embedding_model="all-MiniLM-L6-v2",
            vector_store_type="chroma",
            vector_store_path="./data/chroma_db",
            chunk_size=1000,
            chunk_overlap=200,
            search_results_limit=5,
        )

    config_file = Path(config_path)

    with config_file.open("r") as file:
        config_data = yaml.safe_load(file)

    return Configuration(
        input_directory=config_data["input_directory"],
        vector_store_type=config_data["vector_store_type"],
        vector_store_path=config_data["vector_store_path"],
        embedding_provider=config_data["embedding_provider"],
        embedding_model=config_data["embedding_model"],
        chunk_size=config_data["chunk_size"],
        chunk_overlap=config_data["chunk_overlap"],
        search_results_limit=config_data["search_results_limit"],
    )
