from pathlib import Path

import pytest
from langchain_core.vectorstores import InMemoryVectorStore

from epistemon.indexing import (
    detect_file_changes,
    embed_and_index,
    load_and_chunk_markdown,
    remove_deleted_embeddings,
    scan_markdown_files,
)


@pytest.fixture
def test_data_directory() -> Path:
    return Path("tests/data")


def files_that_produce_chunks(
    files: list[Path], chunk_size: int = 500, chunk_overlap: int = 100
) -> list[Path]:
    return [
        f
        for f in files
        if load_and_chunk_markdown(
            f, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    ]


def create_test_vector_store(
    files: list[Path],
    directory: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    old_mtime: bool = False,
) -> InMemoryVectorStore:
    all_chunks = []
    for file in files:
        chunks = load_and_chunk_markdown(
            file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            base_directory=directory,
        )
        if old_mtime:
            for chunk in chunks:
                chunk.metadata["last_modified"] = 0.0
        all_chunks.extend(chunks)

    return embed_and_index(all_chunks)


def test_scan_markdown_files_non_recursively() -> None:
    directory = Path("tests/data")

    markdown_files = scan_markdown_files(directory, recursive=False)

    assert len(markdown_files) > 0
    assert all(f.suffix == ".md" for f in markdown_files)
    assert all(f.parent == directory for f in markdown_files)


def test_scan_markdown_files_recursively() -> None:
    directory = Path("tests/data")

    markdown_files = scan_markdown_files(directory)

    assert len(markdown_files) > 0
    assert all(f.suffix == ".md" for f in markdown_files)

    paths = [str(f.relative_to(directory)) for f in markdown_files]
    assert any("subdir" in path for path in paths)


def test_load_and_chunk_markdown() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert len(chunks) > 0
    assert all(chunk.page_content for chunk in chunks)


def test_load_and_chunk_markdown_with_relative_source() -> None:
    base_dir = Path("tests/data")
    test_file = base_dir / "subdir" / "nested_doc.md"
    chunks = load_and_chunk_markdown(
        test_file, chunk_size=500, chunk_overlap=100, base_directory=base_dir
    )

    assert all(chunk.metadata["source"] == "subdir/nested_doc.md" for chunk in chunks)


def test_load_and_chunk_markdown_includes_modification_time() -> None:
    test_file = Path("tests/data/sample.md")
    expected_mtime = test_file.stat().st_mtime

    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert all("last_modified" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["last_modified"] == expected_mtime for chunk in chunks)


def test_load_and_chunk_markdown_handles_empty_file() -> None:
    test_file = Path("tests/data/empty.md")

    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert len(chunks) == 0


def test_load_and_chunk_markdown_logs_warning_for_empty_file(
    caplog: pytest.LogCaptureFixture,
) -> None:
    test_file = Path("tests/data/empty.md")

    load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert any(
        record.levelname == "WARNING" and "empty.md" in record.message
        for record in caplog.records
    )


def test_load_and_chunk_markdown_handles_whitespace_only_file() -> None:
    test_file = Path("tests/data/whitespace_only.md")

    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert len(chunks) == 0


def test_load_and_chunk_markdown_handles_malformed_markdown() -> None:
    test_file = Path("tests/data/malformed.md")

    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert len(chunks) > 0
    assert all(chunk.page_content for chunk in chunks)


def test_detect_new_files(test_data_directory: Path) -> None:
    all_files = scan_markdown_files(test_data_directory)
    files_with_chunks = files_that_produce_chunks(all_files)
    vector_store = create_test_vector_store(files_with_chunks[:2], test_data_directory)

    changes = detect_file_changes(test_data_directory, vector_store)

    assert len(changes["new"]) == len(files_with_chunks) - 2
    assert len(changes["modified"]) == 0
    assert len(changes["deleted"]) == 0


def test_detect_modified_files(test_data_directory: Path) -> None:
    all_files = scan_markdown_files(test_data_directory, recursive=False)
    files_with_chunks = files_that_produce_chunks(all_files)
    files = files_with_chunks[:2]
    vector_store = create_test_vector_store(files, test_data_directory, old_mtime=True)

    changes = detect_file_changes(test_data_directory, vector_store)

    all_files_with_chunks = files_that_produce_chunks(
        scan_markdown_files(test_data_directory)
    )
    assert len(changes["new"]) == len(all_files_with_chunks) - 2
    assert len(changes["modified"]) == 2
    assert len(changes["deleted"]) == 0
    assert all(str(f) in [str(p) for p in changes["modified"]] for f in files)


def test_retrieve_indexed_files_from_vector_store(test_data_directory: Path) -> None:
    files = scan_markdown_files(test_data_directory, recursive=False)[:2]
    vector_store = create_test_vector_store(files, test_data_directory)

    indexed_files: dict[str, float] = {}
    for _doc_id, doc_dict in vector_store.store.items():
        source = doc_dict["metadata"]["source"]
        mtime = doc_dict["metadata"]["last_modified"]
        indexed_files[str(test_data_directory / source)] = mtime

    assert len(indexed_files) == len(files)
    for file in files:
        assert str(file) in indexed_files


def test_skip_unchanged_files(test_data_directory: Path) -> None:
    all_files = scan_markdown_files(test_data_directory)
    vector_store = create_test_vector_store(all_files, test_data_directory)

    changes = detect_file_changes(test_data_directory, vector_store)

    assert len(changes["new"]) == 0
    assert len(changes["modified"]) == 0
    assert len(changes["deleted"]) == 0


def test_skip_whitespace_only_files(test_data_directory: Path) -> None:
    files_with_content = [
        f
        for f in scan_markdown_files(test_data_directory)
        if f.name not in ["empty.md", "whitespace_only.md"]
    ]
    vector_store = create_test_vector_store(files_with_content, test_data_directory)

    changes = detect_file_changes(test_data_directory, vector_store)

    assert len(changes["new"]) == 0
    assert len(changes["modified"]) == 0
    assert len(changes["deleted"]) == 0


def test_embed_and_index() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    vector_store = embed_and_index(chunks)

    stored_docs = vector_store.similarity_search("LangChain", k=3)

    assert len(stored_docs) > 0


def test_detect_deleted_files(test_data_directory: Path) -> None:
    all_files = scan_markdown_files(test_data_directory)
    files_with_chunks = files_that_produce_chunks(all_files)
    vector_store = create_test_vector_store(files_with_chunks[:2], test_data_directory)

    fake_deleted_file = test_data_directory / "deleted_file.md"
    fake_chunks = load_and_chunk_markdown(
        files_with_chunks[0],
        chunk_size=500,
        chunk_overlap=100,
        base_directory=test_data_directory,
    )
    for chunk in fake_chunks:
        chunk.metadata["source"] = "deleted_file.md"
        chunk.metadata["last_modified"] = 0.0

    vector_store.add_documents(fake_chunks)

    changes = detect_file_changes(test_data_directory, vector_store)

    assert len(changes["deleted"]) == 1
    assert changes["deleted"][0] == fake_deleted_file


def test_remove_deleted_embeddings(test_data_directory: Path) -> None:
    all_files = scan_markdown_files(test_data_directory)
    files_with_chunks = files_that_produce_chunks(all_files)
    vector_store = create_test_vector_store(files_with_chunks[:2], test_data_directory)

    fake_chunks = load_and_chunk_markdown(
        files_with_chunks[0],
        chunk_size=500,
        chunk_overlap=100,
        base_directory=test_data_directory,
    )
    for chunk in fake_chunks:
        chunk.metadata["source"] = "deleted_file.md"
        chunk.metadata["last_modified"] = 0.0

    vector_store.add_documents(fake_chunks)

    initial_doc_count = len(vector_store.store)

    deleted_files = [test_data_directory / "deleted_file.md"]
    remove_deleted_embeddings(deleted_files, vector_store, test_data_directory)

    final_doc_count = len(vector_store.store)

    assert final_doc_count < initial_doc_count
    assert final_doc_count == initial_doc_count - len(fake_chunks)

    remaining_sources = {
        doc_dict["metadata"]["source"] for doc_dict in vector_store.store.values()
    }
    assert "deleted_file.md" not in remaining_sources
