"""Document loading and chunking functionality."""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
)

from epistemon.instrumentation import measure

logger = logging.getLogger(__name__)


def load_and_chunk_markdown(
    file_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    base_directory: Path | None = None,
) -> list[Document]:
    with measure("load_and_chunk_markdown"):
        content = file_path.read_text()
        mtime = file_path.stat().st_mtime

        if base_directory is not None:
            source = str(file_path.relative_to(base_directory))
        else:
            source = str(file_path)

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        md_header_splits = markdown_splitter.split_text(content)

        for chunk in md_header_splits:
            chunk.metadata["source"] = source
            chunk.metadata["last_modified"] = mtime

        text_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks: list[Document] = text_splitter.split_documents(md_header_splits)

        if not chunks:
            logger.warning(
                "File %s produced no chunks (empty or whitespace-only)", source
            )

        return chunks
