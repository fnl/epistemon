from pathlib import Path

from langchain_core.documents import Document

from epistemon.indexing.bm25_indexer import BM25Indexer, highlight_keywords


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

    results = indexer.retrieve("LangChain framework", top_k=3)

    assert len(results) > 0
    assert len(results) <= 3
    assert all(isinstance(doc, Document) for doc, score in results)
    assert all(isinstance(score, float) for doc, score in results)


def test_bm25_retriever_returns_documents_with_scores() -> None:
    directory = Path("tests/data")
    indexer = BM25Indexer(directory)

    results = indexer.retrieve("LangChain framework", top_k=3)

    assert len(results) > 0
    assert len(results) <= 3
    assert all(isinstance(doc, Document) for doc, score in results)
    assert all(isinstance(score, float) for doc, score in results)
    assert all(score >= 0 for doc, score in results)
    assert results[0][1] >= results[1][1]


def test_highlight_keywords_wraps_matched_keywords_with_mark_tags() -> None:
    text = "LangChain is a framework for building applications with LLMs."
    query = "langchain framework"

    result = highlight_keywords(text, query)

    assert "<mark>LangChain</mark>" in result
    assert "<mark>framework</mark>" in result


def test_highlight_keywords_only_matches_whole_words() -> None:
    text = "Creativity is important in this situation."
    query = "it"

    result = highlight_keywords(text, query)

    assert "<mark>it</mark>" not in result
    assert "Creativity" in result
    assert "<mark>" not in result


def test_highlight_keywords_matches_whole_word_boundaries() -> None:
    text = "The item is on the list."
    query = "item is"

    result = highlight_keywords(text, query)

    assert "<mark>item</mark>" in result
    assert "<mark>is</mark>" in result
    assert "The <mark>item</mark> <mark>is</mark> on the list." == result


def test_bm25_indexer_uses_case_insensitive_indexing() -> None:
    directory = Path("tests/data")
    indexer = BM25Indexer(directory)

    results_upper = indexer.retrieve("LANGCHAIN", top_k=3)
    results_lower = indexer.retrieve("langchain", top_k=3)

    assert len(results_upper) > 0
    assert len(results_lower) > 0
    assert results_upper[0][0].page_content == results_lower[0][0].page_content
    assert abs(results_upper[0][1] - results_lower[0][1]) < 0.01
