"""Document loading and chunking functionality."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_markdown(
    file_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    base_directory: Path | None = None,
) -> list[Document]:
    content = file_path.read_text()

    if base_directory is not None:
        source = str(file_path.relative_to(base_directory))
    else:
        source = str(file_path)

    document = Document(page_content=content, metadata={"source": source})

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    return text_splitter.split_documents([document])
