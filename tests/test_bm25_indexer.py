from pathlib import Path

from epistemon.indexing.bm25_indexer import BM25Indexer


def test_bm25_indexer_initializes_from_directory() -> None:
    directory = Path("tests/data")
    chunk_size = 500
    chunk_overlap = 100

    indexer = BM25Indexer(directory, chunk_size, chunk_overlap)

    assert indexer is not None
    assert indexer.bm25_index is not None
    assert len(indexer.documents) > 0
