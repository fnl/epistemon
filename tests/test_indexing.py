from pathlib import Path

from epistemon.indexing import (
    detect_file_changes,
    embed_and_index,
    load_and_chunk_markdown,
    scan_markdown_files,
)


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


def test_load_and_chunk_markdown_handles_malformed_markdown() -> None:
    test_file = Path("tests/data/malformed.md")

    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert len(chunks) > 0
    assert all(chunk.page_content for chunk in chunks)


def test_detect_new_files() -> None:
    directory = Path("tests/data")
    all_files = scan_markdown_files(directory)

    all_chunks = []
    for file in all_files[:2]:
        chunks = load_and_chunk_markdown(
            file, chunk_size=500, chunk_overlap=100, base_directory=directory
        )
        all_chunks.extend(chunks)

    vector_store = embed_and_index(all_chunks)

    changes = detect_file_changes(directory, vector_store)

    assert len(changes["new"]) == len(all_files) - 2
    assert len(changes["modified"]) == 0
    assert len(changes["deleted"]) == 0


def test_retrieve_indexed_files_from_vector_store() -> None:
    directory = Path("tests/data")
    files = scan_markdown_files(directory, recursive=False)[:2]

    all_chunks = []
    for file in files:
        chunks = load_and_chunk_markdown(
            file, chunk_size=500, chunk_overlap=100, base_directory=directory
        )
        all_chunks.extend(chunks)

    vector_store = embed_and_index(all_chunks)

    indexed_files: dict[str, float] = {}
    for _doc_id, doc_dict in vector_store.store.items():
        source = doc_dict["metadata"]["source"]
        mtime = doc_dict["metadata"]["last_modified"]
        indexed_files[str(directory / source)] = mtime

    assert len(indexed_files) == len(files)
    for file in files:
        assert str(file) in indexed_files


def test_embed_and_index() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    vector_store = embed_and_index(chunks)

    stored_docs = vector_store.similarity_search("LangChain", k=3)

    assert len(stored_docs) > 0
