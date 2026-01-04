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
from epistemon.indexing.vector_store_manager import (
    VectorStoreManager,
    create_vector_store_manager,
)
from epistemon.instrumentation import measure

logger = logging.getLogger(__name__)


def _process_deleted_files(
    deleted_files: list[Path], manager: VectorStoreManager
) -> None:
    if deleted_files:
        with measure("remove_deleted"):
            logger.info("Removing embeddings for %d deleted files", len(deleted_files))
            remove_deleted_embeddings(deleted_files, manager)


def _process_new_files(
    new_files: list[Path],
    config: Configuration,
    directory: Path,
    manager: VectorStoreManager,
) -> None:
    new_file_chunks: list[Document] = []
    for file_path in new_files:
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
                len(new_files),
            )
            manager.add_documents(new_file_chunks)


def _process_modified_files(
    modified_files: list[Path],
    config: Configuration,
    directory: Path,
    manager: VectorStoreManager,
) -> None:
    modified_chunks_total = 0
    for file_path in modified_files:
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

    if modified_files:
        logger.info(
            "Updated %d chunks from %d modified files",
            modified_chunks_total,
            len(modified_files),
        )


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

        _process_deleted_files(changes.deleted, manager)
        _process_new_files(changes.new, config, directory, manager)
        _process_modified_files(changes.modified, config, directory, manager)
