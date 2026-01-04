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
    assert all(score > 0 for doc, score in results)
    if len(results) >= 2:
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
    query = "item list"

    result = highlight_keywords(text, query)

    assert "<mark>item</mark>" in result
    assert "<mark>list</mark>" in result
    assert "The <mark>item</mark> is on the <mark>list</mark>." == result


def test_bm25_indexer_uses_case_insensitive_indexing() -> None:
    directory = Path("tests/data")
    indexer = BM25Indexer(directory)

    results_upper = indexer.retrieve("LANGCHAIN", top_k=3)
    results_lower = indexer.retrieve("langchain", top_k=3)

    assert len(results_upper) > 0
    assert len(results_lower) > 0
    assert results_upper[0][0].page_content == results_lower[0][0].page_content
    assert abs(results_upper[0][1] - results_lower[0][1]) < 0.01


def test_bm25_indexer_filters_common_stopwords() -> None:
    directory = Path("tests/data")
    indexer = BM25Indexer(directory)

    results_with_stopwords = indexer.retrieve("the a in", top_k=3)
    results_content = indexer.retrieve("content", top_k=3)

    assert len(results_content) > 0
    for _doc, score in results_with_stopwords:
        assert score < results_content[0][1] or score == 0.0


def test_bm25_indexer_filters_query_words() -> None:
    directory = Path("tests/data")
    indexer = BM25Indexer(directory)

    results_with_query_words = indexer.retrieve("how what when where why", top_k=3)
    results_content = indexer.retrieve("langchain", top_k=3)

    assert len(results_content) > 0
    for _doc, score in results_with_query_words:
        assert score < results_content[0][1] or score == 0.0


def test_highlight_keywords_filters_stopwords() -> None:
    text = "The framework is a tool for building applications."
    query = "the framework building"

    result = highlight_keywords(text, query)

    assert "<mark>framework</mark>" in result
    assert "<mark>building</mark>" in result
    assert "<mark>The</mark>" not in result
    assert "<mark>the</mark>" not in result
    assert (
        result
        == "The <mark>framework</mark> is a tool for <mark>building</mark> applications."
    )


def test_bm25_returns_empty_results_for_completely_unmatched_query() -> None:
    directory = Path("tests/data")
    indexer = BM25Indexer(directory)

    results = indexer.retrieve("xyzqwertynonexistent", top_k=5)

    assert len(results) == 0
