from pathlib import Path

from epistemon.indexing import load_and_chunk_markdown


def test_load_and_chunk_markdown() -> None:
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    assert len(chunks) > 0
    assert all(chunk.page_content for chunk in chunks)
    assert all("source" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata["source"] == str(test_file) for chunk in chunks)
