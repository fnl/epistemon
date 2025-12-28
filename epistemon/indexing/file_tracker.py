"""File change detection for incremental indexing."""

from pathlib import Path

from langchain_core.vectorstores import InMemoryVectorStore

from epistemon.indexing.scanner import scan_markdown_files


def detect_file_changes(
    directory: Path, vector_store: InMemoryVectorStore
) -> dict[str, list[Path]]:
    current_files = scan_markdown_files(directory)
    current_files_map = {str(f): f.stat().st_mtime for f in current_files}

    indexed_files: dict[str, float] = {}
    for _doc_id, doc_dict in vector_store.store.items():
        source = doc_dict["metadata"]["source"]
        mtime = doc_dict["metadata"]["last_modified"]
        indexed_files[str(directory / source)] = mtime

    new_files = [
        Path(path) for path in current_files_map.keys() if path not in indexed_files
    ]

    modified_files = [
        Path(path)
        for path, mtime in current_files_map.items()
        if path in indexed_files and indexed_files[path] != mtime
    ]

    deleted_files = [
        Path(path) for path in indexed_files.keys() if path not in current_files_map
    ]

    return {"new": new_files, "modified": modified_files, "deleted": deleted_files}
