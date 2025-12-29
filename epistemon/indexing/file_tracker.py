"""File change detection for incremental indexing."""

from pathlib import Path
from typing import NamedTuple, cast

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from epistemon.indexing.chunker import load_and_chunk_markdown
from epistemon.instrumentation import measure


class FileChanges(NamedTuple):
    new: list[Path]
    modified: list[Path]
    deleted: list[Path]


def collect_markdown_files(directory: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*.md" if recursive else "*.md"
    return sorted(directory.glob(pattern))


def get_indexed_files_from_vector_store(
    vector_store: VectorStore, directory: Path
) -> dict[str, float]:
    indexed_files: dict[str, float] = {}

    if hasattr(vector_store, "get"):
        chroma_store = cast(Chroma, vector_store)
        result = chroma_store.get()
        if result and "metadatas" in result:
            for metadata in result["metadatas"]:
                if metadata and "source" in metadata and "last_modified" in metadata:
                    source = metadata["source"]
                    mtime = metadata["last_modified"]
                    indexed_files[str(directory / source)] = mtime
    elif hasattr(vector_store, "store"):
        inmemory_store = cast(InMemoryVectorStore, vector_store)
        for _doc_id, doc_dict in inmemory_store.store.items():
            source = doc_dict["metadata"]["source"]
            mtime = doc_dict["metadata"]["last_modified"]
            indexed_files[str(directory / source)] = mtime

    return indexed_files


def detect_file_changes(directory: Path, vector_store: VectorStore) -> FileChanges:
    with measure("detect_file_changes"):
        current_files = collect_markdown_files(directory)
        current_files_map = {
            str(f): f.stat().st_mtime for f in current_files if f.stat().st_size > 0
        }

        indexed_files = get_indexed_files_from_vector_store(vector_store, directory)

        new_files = []
        modified_files = []

        for path, mtime in current_files_map.items():
            chunks = load_and_chunk_markdown(
                Path(path), chunk_size=1000, chunk_overlap=200
            )
            if not chunks:
                continue

            if path not in indexed_files:
                new_files.append(Path(path))
            elif indexed_files[path] != mtime:
                modified_files.append(Path(path))

        deleted_files = [
            Path(path) for path in indexed_files.keys() if path not in current_files_map
        ]

        return FileChanges(
            new=new_files, modified=modified_files, deleted=deleted_files
        )


def remove_deleted_embeddings(
    deleted_files: list[Path], vector_store: VectorStore, base_directory: Path
) -> None:
    deleted_sources = {str(f.relative_to(base_directory)) for f in deleted_files}

    doc_ids_to_remove = []

    if hasattr(vector_store, "get"):
        chroma_store = cast(Chroma, vector_store)
        result = chroma_store.get()
        if result and "ids" in result and "metadatas" in result:
            for doc_id, metadata in zip(
                result["ids"], result["metadatas"], strict=False
            ):
                if metadata and metadata.get("source") in deleted_sources:
                    doc_ids_to_remove.append(doc_id)
    elif hasattr(vector_store, "store"):
        inmemory_store = cast(InMemoryVectorStore, vector_store)
        doc_ids_to_remove = [
            doc_id
            for doc_id, doc_dict in inmemory_store.store.items()
            if doc_dict["metadata"]["source"] in deleted_sources
        ]

    if doc_ids_to_remove:
        if hasattr(vector_store, "delete"):
            vector_store.delete(doc_ids_to_remove)
        elif hasattr(vector_store, "store"):
            inmemory_store = cast(InMemoryVectorStore, vector_store)
            for doc_id in doc_ids_to_remove:
                del inmemory_store.store[doc_id]


def update_embeddings_for_file(
    file_path: Path,
    new_chunks: list[Document],
    vector_store: VectorStore,
    base_directory: Path,
) -> None:
    remove_deleted_embeddings([file_path], vector_store, base_directory)
    vector_store.add_documents(new_chunks)
