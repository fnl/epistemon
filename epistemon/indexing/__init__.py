"""Indexing module for scanning and embedding markdown files."""

from epistemon.indexing.chunker import load_and_chunk_markdown
from epistemon.indexing.file_tracker import detect_file_changes
from epistemon.indexing.indexer import embed_and_index
from epistemon.indexing.scanner import scan_markdown_files

__all__ = [
    "load_and_chunk_markdown",
    "embed_and_index",
    "scan_markdown_files",
    "detect_file_changes",
]
