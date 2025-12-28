from pathlib import Path

from epistemon.indexing import (
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


def test_embed_and_index() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    vector_store = embed_and_index(chunks)

    stored_docs = vector_store.similarity_search("LangChain", k=3)

    assert len(stored_docs) > 0
