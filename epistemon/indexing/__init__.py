"""Indexing module for scanning and embedding markdown files."""

from epistemon.indexing.chunker import load_and_chunk_markdown
from epistemon.indexing.file_tracker import (
    detect_file_changes,
    remove_deleted_embeddings,
    update_embeddings_for_file,
)
from epistemon.indexing.scanner import scan_markdown_files
from epistemon.indexing.vector_store_factory import create_vector_store

__all__ = [
    "load_and_chunk_markdown",
    "scan_markdown_files",
    "detect_file_changes",
    "remove_deleted_embeddings",
    "update_embeddings_for_file",
    "create_vector_store",
]
