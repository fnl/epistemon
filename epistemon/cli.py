"""Command-line interface for Epistemon."""

import argparse
import logging
import sys
from typing import Optional

from epistemon.config import load_config
from epistemon.indexing.indexer import index
from epistemon.vector_store_factory import create_vector_store

logger = logging.getLogger(__name__)


def upsert_index_command(config_path: Optional[str]) -> None:
    try:
        logger.info("Loading configuration...")
        config = load_config(config_path)

        logger.info(f"Creating vector store ({config.vector_store_type})...")
        vector_store = create_vector_store(config)

        logger.info(f"Indexing {config.input_directory}...")
        index(config, vector_store)

        logger.info("Indexing complete!")
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        sys.exit(1)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Update the search index from markdown files"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (uses defaults if not specified)",
    )
    args = parser.parse_args()
    upsert_index_command(args.config)
