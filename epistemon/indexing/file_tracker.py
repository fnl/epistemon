"""File change detection for incremental indexing."""

from pathlib import Path
from typing import NamedTuple

from epistemon.indexing.chunker import load_and_chunk_markdown
from epistemon.indexing.vector_store_manager import VectorStoreManager
from epistemon.instrumentation import measure


class FileChanges(NamedTuple):
    new: list[Path]
    modified: list[Path]
    deleted: list[Path]


def collect_markdown_files(directory: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*.md" if recursive else "*.md"
    return sorted(directory.glob(pattern))


def detect_file_changes(directory: Path, manager: VectorStoreManager) -> FileChanges:
    with measure("detect_file_changes"):
        current_files = collect_markdown_files(directory)
        current_files_map = {
            str(f): f.stat().st_mtime for f in current_files if f.stat().st_size > 0
        }

        indexed_files = manager.get_indexed_files()

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
    deleted_files: list[Path], manager: VectorStoreManager
) -> None:
    deleted_sources = {
        str(f.relative_to(manager.base_directory)) for f in deleted_files
    }
    manager.remove_documents_by_source(deleted_sources)
