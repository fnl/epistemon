"""Unified indexing API for updating vector stores."""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from epistemon.config import Configuration
from epistemon.indexing.chunker import load_and_chunk_markdown
from epistemon.indexing.file_tracker import (
    detect_file_changes,
    remove_deleted_embeddings,
)
from epistemon.indexing.vector_store_manager import create_vector_store_manager
from epistemon.instrumentation import measure

logger = logging.getLogger(__name__)


def index(config: Configuration, vector_store: VectorStore) -> None:
    with measure("index_total"):
        directory = Path(config.input_directory)
        manager = create_vector_store_manager(vector_store, directory)
        changes = detect_file_changes(directory, manager)

        logger.info(
            "File changes detected: %d new, %d modified, %d deleted",
            len(changes.new),
            len(changes.modified),
            len(changes.deleted),
        )

        if changes.deleted:
            with measure("remove_deleted"):
                logger.info(
                    "Removing embeddings for %d deleted files", len(changes.deleted)
                )
                remove_deleted_embeddings(changes.deleted, manager)

        new_file_chunks: list[Document] = []
        for file_path in changes.new:
            chunks = load_and_chunk_markdown(
                file_path,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                base_directory=directory,
            )
            if chunks:
                new_file_chunks.extend(chunks)

        if new_file_chunks:
            with measure("add_new_documents"):
                logger.info(
                    "Adding %d chunks from %d new files",
                    len(new_file_chunks),
                    len(changes.new),
                )
                manager.add_documents(new_file_chunks)

        modified_chunks_total = 0
        for file_path in changes.modified:
            chunks = load_and_chunk_markdown(
                file_path,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                base_directory=directory,
            )
            if chunks:
                modified_chunks_total += len(chunks)
                with measure("update_modified_file"):
                    manager.update_documents_for_file(file_path, chunks)

        if changes.modified:
            logger.info(
                "Updated %d chunks from %d modified files",
                modified_chunks_total,
                len(changes.modified),
            )
