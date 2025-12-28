"""Indexing module for scanning and embedding markdown files."""

from epistemon.indexing.chunker import load_and_chunk_markdown
from epistemon.indexing.file_tracker import (
    FileChanges,
    collect_markdown_files,
    detect_file_changes,
    remove_deleted_embeddings,
    update_embeddings_for_file,
)

__all__ = [
    "load_and_chunk_markdown",
    "collect_markdown_files",
    "detect_file_changes",
    "remove_deleted_embeddings",
    "update_embeddings_for_file",
    "FileChanges",
]
