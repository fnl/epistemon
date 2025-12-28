"""File change detection for incremental indexing."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from epistemon.indexing.chunker import load_and_chunk_markdown
from epistemon.indexing.scanner import scan_markdown_files


def detect_file_changes(
    directory: Path, vector_store: InMemoryVectorStore
) -> dict[str, list[Path]]:
    current_files = scan_markdown_files(directory)
    current_files_map = {
        str(f): f.stat().st_mtime for f in current_files if f.stat().st_size > 0
    }

    indexed_files: dict[str, float] = {}
    for _doc_id, doc_dict in vector_store.store.items():
        source = doc_dict["metadata"]["source"]
        mtime = doc_dict["metadata"]["last_modified"]
        indexed_files[str(directory / source)] = mtime

    new_files = []
    modified_files = []

    for path, mtime in current_files_map.items():
        chunks = load_and_chunk_markdown(Path(path), chunk_size=1000, chunk_overlap=200)
        if not chunks:
            continue

        if path not in indexed_files:
            new_files.append(Path(path))
        elif indexed_files[path] != mtime:
            modified_files.append(Path(path))

    deleted_files = [
        Path(path) for path in indexed_files.keys() if path not in current_files_map
    ]

    return {"new": new_files, "modified": modified_files, "deleted": deleted_files}


def remove_deleted_embeddings(
    deleted_files: list[Path], vector_store: InMemoryVectorStore, base_directory: Path
) -> None:
    deleted_sources = {str(f.relative_to(base_directory)) for f in deleted_files}

    doc_ids_to_remove = [
        doc_id
        for doc_id, doc_dict in vector_store.store.items()
        if doc_dict["metadata"]["source"] in deleted_sources
    ]

    for doc_id in doc_ids_to_remove:
        del vector_store.store[doc_id]


def update_embeddings_for_file(
    file_path: Path,
    new_chunks: list[Document],
    vector_store: InMemoryVectorStore,
    base_directory: Path,
) -> None:
    remove_deleted_embeddings([file_path], vector_store, base_directory)
    vector_store.add_documents(new_chunks)
