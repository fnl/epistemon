"""Document loading and chunking functionality."""

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter

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

        document = Document(
            page_content=content, metadata={"source": source, "last_modified": mtime}
        )

        text_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

        chunks: list[Document] = text_splitter.split_documents([document])

        if not chunks:
            logger.warning(
                "File %s produced no chunks (empty or whitespace-only)", source
            )

        return chunks
