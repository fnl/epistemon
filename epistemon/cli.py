"""Command-line interface for Epistemon."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import uvicorn

from epistemon.config import load_config
from epistemon.indexing.indexer import index
from epistemon.indexing.vector_store_manager import create_vector_store_manager
from epistemon.vector_store_factory import create_vector_store
from epistemon.web import create_app

logger = logging.getLogger(__name__)


def upsert_index_command(config_path: Optional[str]) -> None:
    try:
        logger.info(f"Loading configuration from '{config_path or 'defaults'}'...")
        config = load_config(config_path)

        logger.info(f"Creating vector store ({config.vector_store_type})...")
        vector_store = create_vector_store(config)

        logger.info(f"Indexing {config.input_directory}...")
        index(config, vector_store)

        logger.info("Indexing complete!")
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        sys.exit(1)


def web_ui_command(config_path: Optional[str], host: str, port: int) -> None:
    try:
        logger.info("Loading configuration...")
        config = load_config(config_path)

        logger.info(f"Creating vector store ({config.vector_store_type})...")
        vector_store = create_vector_store(config)

        logger.info("Creating web application...")
        app = create_app(
            vector_store,
            base_url=f"http://{host}:{port}/files",
            score_threshold=config.score_threshold,
            files_directory=Path(config.input_directory),
            vector_store_manager=create_vector_store_manager(
                vector_store, Path(config.input_directory)
            ),
        )

        logger.info(f"Starting server at http://{host}:{port}")
        logger.info(f"Shiny UI: http://{host}:{port}/ (redirects to /app/)")
        logger.info(f"API docs: http://{host}:{port}/docs")
        logger.info("Press Ctrl+C to stop")

        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
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


def web_ui_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Start the Epistemon web UI and API server"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (uses defaults if not specified)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )
    args = parser.parse_args()
    web_ui_command(args.config, args.host, args.port)
