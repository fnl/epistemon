"""Command-line interface for Epistemon."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import uvicorn

from epistemon.config import load_config
from epistemon.indexing.bm25_indexer import BM25Indexer
from epistemon.indexing.indexer import index
from epistemon.indexing.vector_store_manager import create_vector_store_manager
from epistemon.llm_factory import create_llm
from epistemon.logging_config import setup_logging
from epistemon.retrieval.hybrid_retriever import HybridRetriever
from epistemon.retrieval.rag_chain import RAGChain
from epistemon.tracing import create_traced_rag_chain
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
        logger.info(f"Loading configuration from {config_path or 'defaults'}...")
        config = load_config(config_path)

        logger.info(f"Creating vector store ({config.vector_store_type})...")
        vector_store = create_vector_store(config)

        logger.info("Building BM25 keyword search index...")
        bm25_indexer = BM25Indexer(
            Path(config.input_directory),
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        rag_chain = None
        if config.rag_enabled:
            logger.info("Creating RAG chain...")
            hybrid_retriever = HybridRetriever(
                bm25_retriever=bm25_indexer,
                semantic_retriever=vector_store,
                bm25_weight=config.hybrid_bm25_weight,
                semantic_weight=config.hybrid_semantic_weight,
            )
            llm = create_llm(config)
            rag_chain = RAGChain(retriever=hybrid_retriever, llm=llm)
            rag_chain = create_traced_rag_chain(
                rag_chain,
                tracing_enabled=config.tracing_enabled,
                embedding_model=config.embedding_model,
            )

        logger.info("Creating web application...")
        app = create_app(
            vector_store,
            base_url=f"http://{host}:{port}/files",
            files_directory=Path(config.input_directory),
            vector_store_manager=create_vector_store_manager(
                vector_store, Path(config.input_directory)
            ),
            bm25_retriever=bm25_indexer,
            rag_chain=rag_chain,
            config=config,
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
    setup_logging()

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
    setup_logging()

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
