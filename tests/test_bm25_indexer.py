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


def test_bm25_indexer_can_query_and_return_results() -> None:
    directory = Path("tests/data")
    indexer = BM25Indexer(directory)

    results = indexer.search("LangChain framework", top_k=3)

    assert len(results) > 0
    assert len(results) <= 3
    assert all(isinstance(result, str) for result in results)
