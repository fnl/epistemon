"""Unified indexing API for updating vector stores."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from epistemon.config import Configuration
from epistemon.indexing.chunker import load_and_chunk_markdown
from epistemon.indexing.file_tracker import (
    detect_file_changes,
    remove_deleted_embeddings,
    update_embeddings_for_file,
)


def index(config: Configuration, vector_store: VectorStore) -> None:
    directory = Path(config.input_directory)
    changes = detect_file_changes(directory, vector_store)

    if changes.deleted:
        remove_deleted_embeddings(changes.deleted, vector_store, directory)

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
        vector_store.add_documents(new_file_chunks)

    for file_path in changes.modified:
        chunks = load_and_chunk_markdown(
            file_path,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            base_directory=directory,
        )
        if chunks:
            update_embeddings_for_file(file_path, chunks, vector_store, directory)
