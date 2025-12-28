"""Indexing module for scanning and embedding markdown files."""

from epistemon.indexing.chunker import load_and_chunk_markdown
from epistemon.indexing.indexer import embed_and_index

__all__ = ["load_and_chunk_markdown", "embed_and_index"]
