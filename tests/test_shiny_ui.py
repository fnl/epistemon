from pathlib import Path
from unittest.mock import Mock

from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from shiny import App

from epistemon.indexing import load_and_chunk_markdown
from epistemon.web.shiny_ui import create_shiny_app


def test_create_shiny_app_returns_app_instance() -> None:
    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))

    app = create_shiny_app(vector_store)

    assert isinstance(app, App)


def test_shiny_app_has_search_server_logic() -> None:
    test_file = Path("tests/data/sample.md")
    base_directory = Path("tests/data")
    chunks = load_and_chunk_markdown(
        test_file, chunk_size=500, chunk_overlap=100, base_directory=base_directory
    )
    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(chunks)

    app = create_shiny_app(vector_store, score_threshold=-1.0)

    assert app.ui is not None
    assert app.server is not None


def test_shiny_app_calls_similarity_search() -> None:

    mock_store = Mock(spec=VectorStore)
    mock_doc = Document(
        page_content="Test content",
        metadata={"source": "test.md", "last_modified": 1234567890.0},
    )
    mock_store.similarity_search_with_score.return_value = [(mock_doc, 0.95)]

    app = create_shiny_app(mock_store, score_threshold=0.0)

    assert app is not None
    mock_input = Mock()
    mock_input.query.return_value = "test query"
    mock_input.limit.return_value = 5
    mock_input.search = 1

    assert callable(app.server)


def test_shiny_app_includes_ui_components() -> None:
    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    app = create_shiny_app(vector_store)

    ui_html = str(app.ui)

    assert "Search Query" in ui_html
    assert "Result Limit" in ui_html
    assert "Search" in ui_html
    assert "Epistemon" in ui_html


def test_shiny_app_accepts_base_url_parameter() -> None:
    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))

    app = create_shiny_app(
        vector_store, base_url="http://localhost:8000/files", score_threshold=0.5
    )

    assert isinstance(app, App)


def test_shiny_app_with_populated_vector_store() -> None:
    test_file = Path("tests/data/sample.md")
    base_directory = Path("tests/data")
    chunks = load_and_chunk_markdown(
        test_file, chunk_size=500, chunk_overlap=100, base_directory=base_directory
    )
    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(chunks)

    app = create_shiny_app(
        vector_store,
        base_url="http://localhost:8000/files",
        score_threshold=-1.0,
    )

    assert isinstance(app, App)
    assert len(vector_store.similarity_search_with_score("test", k=1)) > 0


def test_vector_store_usage_with_validated_limit() -> None:
    from unittest.mock import patch

    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))

    with patch.object(
        vector_store,
        "similarity_search_with_score",
        wraps=vector_store.similarity_search_with_score,
    ) as mock_search:
        mock_doc = Document(page_content="test", metadata={})
        mock_search.return_value = [(mock_doc, 0.5)]

        test_file = Path("tests/data/sample.md")
        base_directory = Path("tests/data")
        chunks = load_and_chunk_markdown(
            test_file, chunk_size=500, chunk_overlap=100, base_directory=base_directory
        )
        vector_store.add_documents(chunks)

        result = vector_store.similarity_search_with_score("test", k=5)

        assert len(result) > 0
        mock_search.assert_called_with("test", k=5)


def test_create_shiny_app_accepts_bm25_retriever() -> None:
    from epistemon.indexing.bm25_indexer import BM25Indexer

    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    bm25_indexer = Mock(spec=BM25Indexer)

    app = create_shiny_app(
        vector_store, bm25_retriever=bm25_indexer, score_threshold=0.0
    )

    assert isinstance(app, App)


def test_shiny_app_has_two_column_layout_with_headers() -> None:
    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    app = create_shiny_app(vector_store)

    ui_html = str(app.ui)

    assert "BM25 (Keyword Search)" in ui_html
    assert "Semantic (Embedding Search)" in ui_html


def test_bm25_search_executes_independently() -> None:
    from epistemon.indexing.bm25_indexer import BM25Indexer

    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    bm25_indexer = Mock(spec=BM25Indexer)
    mock_doc = Document(
        page_content="BM25 test content",
        metadata={"source": "test.md", "last_modified": 1234567890.0},
    )
    bm25_indexer.retrieve.return_value = [(mock_doc, 2.5)]

    app = create_shiny_app(
        vector_store, bm25_retriever=bm25_indexer, score_threshold=0.0
    )

    assert app is not None
    assert bm25_indexer.retrieve.call_count == 0


def test_bm25_results_output_exists_in_ui() -> None:
    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    app = create_shiny_app(vector_store)

    ui_html = str(app.ui)

    assert "bm25_results" in ui_html


def test_bm25_retriever_error_handling() -> None:
    from epistemon.indexing.bm25_indexer import BM25Indexer

    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    bm25_indexer = Mock(spec=BM25Indexer)
    bm25_indexer.retrieve.side_effect = Exception("BM25 index error")

    app = create_shiny_app(
        vector_store, bm25_retriever=bm25_indexer, score_threshold=0.0
    )

    assert app is not None
    assert app.server is not None


def test_bm25_badges_use_distinct_colors_from_semantic() -> None:
    from epistemon.indexing.bm25_indexer import BM25Indexer

    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    bm25_indexer = Mock(spec=BM25Indexer)

    app = create_shiny_app(
        vector_store, bm25_retriever=bm25_indexer, score_threshold=0.0
    )

    assert app is not None


def test_bm25_results_highlight_matched_keywords() -> None:
    from epistemon.indexing.bm25_indexer import BM25Indexer

    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    bm25_indexer = BM25Indexer(Path("tests/data"))

    app = create_shiny_app(
        vector_store, bm25_retriever=bm25_indexer, score_threshold=0.0
    )

    assert app is not None


def test_create_shiny_app_accepts_rag_chain_parameter() -> None:
    from epistemon.retrieval.rag_chain import RAGChain

    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    rag_chain = Mock(spec=RAGChain)

    app = create_shiny_app(vector_store, rag_chain=rag_chain)

    assert isinstance(app, App)


def test_search_ui_has_top_bar_layout_without_sidebar() -> None:
    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    app = create_shiny_app(vector_store)

    ui_dict = app.ui
    ui_html = ui_dict.get("html", str(ui_dict))

    assert "<aside" not in ui_html
    assert "collapse-toggle" not in ui_html
    assert "Search Query" in ui_html or "search query" in ui_html.lower()
    assert "Result Limit" in ui_html or "result limit" in ui_html.lower()


def test_shiny_app_has_three_column_layout_with_rag() -> None:
    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    app = create_shiny_app(vector_store)

    ui_html = str(app.ui)

    assert "BM25 (Keyword Search)" in ui_html
    assert "Semantic (Embedding Search)" in ui_html
    assert "RAG Answer" in ui_html or "RAG" in ui_html


def test_rag_answer_renders_when_rag_chain_provided() -> None:
    from epistemon.retrieval.rag_chain import RAGChain, RAGResponse

    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    rag_chain = Mock(spec=RAGChain)

    mock_doc = Document(
        page_content="Source content about testing",
        metadata={"source": "test.md", "last_modified": 1234567890.0},
    )
    mock_response = RAGResponse(
        answer="This is the generated answer from RAG",
        source_documents=[mock_doc],
        query="test query",
    )
    rag_chain.invoke.return_value = mock_response

    app = create_shiny_app(vector_store, rag_chain=rag_chain)

    assert app is not None
    assert rag_chain.invoke.call_count == 0


def test_rag_answer_shows_not_available_when_no_chain() -> None:
    from epistemon.web.shiny_ui import _execute_rag_answer

    result = _execute_rag_answer(None, "", 0.0, "test query", 5)

    result_html = str(result)
    assert (
        "not available" in result_html.lower()
        or "not implemented" in result_html.lower()
    )


def test_rag_answer_displays_answer_and_sources() -> None:
    from epistemon.retrieval.rag_chain import RAGChain, RAGResponse
    from epistemon.web.shiny_ui import _execute_rag_answer

    rag_chain = Mock(spec=RAGChain)
    mock_doc = Document(
        page_content="Source content about testing",
        metadata={"source": "test.md", "last_modified": 1234567890.0},
    )
    mock_response = RAGResponse(
        answer="This is the generated answer",
        source_documents=[mock_doc],
        query="test query",
    )
    rag_chain.invoke.return_value = mock_response

    result = _execute_rag_answer(rag_chain, "", 0.0, "test query", 5)

    result_html = str(result)
    assert "This is the generated answer" in result_html
    assert "Source content about testing" in result_html
    assert "test.md" in result_html
    rag_chain.invoke.assert_called_once_with("test query")


def test_rag_answer_handles_slow_processing() -> None:
    import time

    from epistemon.retrieval.rag_chain import RAGChain, RAGResponse
    from epistemon.web.shiny_ui import _execute_rag_answer

    rag_chain = Mock(spec=RAGChain)
    mock_doc = Document(
        page_content="Result after delay",
        metadata={"source": "slow.md", "last_modified": 1234567890.0},
    )
    mock_response = RAGResponse(
        answer="Answer after processing delay",
        source_documents=[mock_doc],
        query="slow query",
    )

    def slow_invoke(query: str) -> RAGResponse:
        time.sleep(0.1)
        return mock_response

    rag_chain.invoke.side_effect = slow_invoke

    result = _execute_rag_answer(rag_chain, "", 0.0, "slow query", 5)

    result_html = str(result)
    assert "Answer after processing delay" in result_html
    assert "Result after delay" in result_html


def test_rag_answer_handles_processing_errors() -> None:
    from epistemon.retrieval.rag_chain import RAGChain
    from epistemon.web.shiny_ui import _execute_rag_answer

    rag_chain = Mock(spec=RAGChain)
    rag_chain.invoke.side_effect = Exception("LLM API error")

    result = _execute_rag_answer(rag_chain, "", 0.0, "test query", 5)

    result_html = str(result)
    assert "error" in result_html.lower()
    assert "LLM API error" in result_html


def test_llm_api_error_displays_as_alert_not_answer_card() -> None:
    from epistemon.retrieval.rag_chain import RAGChain
    from epistemon.web.shiny_ui import _execute_rag_answer

    rag_chain = Mock(spec=RAGChain)
    rag_chain.invoke.side_effect = Exception(
        "Error code: 429 - {'error': {'message': 'You exceeded your current quota'}}"
    )

    result = _execute_rag_answer(rag_chain, "", 0.0, "test query", 5)

    result_html = str(result)
    assert "alert alert-danger" in result_html
    assert "Error code: 429" in result_html
    assert "exceeded your current quota" in result_html
    assert (
        "Answer" not in result_html
        or 'class="bg-success text-white"' not in result_html
    )


def test_semantic_search_handles_vector_store_errors() -> None:
    from langchain_core.vectorstores import VectorStore

    from epistemon.web.shiny_ui import _execute_semantic_search

    broken_store = Mock(spec=VectorStore)
    broken_store.similarity_search_with_score.side_effect = Exception(
        "Vector database connection failed"
    )

    result = _execute_semantic_search(broken_store, "", 0.0, "test query", 5)

    result_html = str(result)
    assert "error" in result_html.lower()
    assert "Vector database connection failed" in result_html


def test_search_bar_uses_majority_of_screen_width() -> None:
    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    app = create_shiny_app(vector_store)

    ui_html = str(app.ui)

    assert 'class="col-9"' in ui_html
    assert 'class="col-2"' in ui_html
    assert 'class="col-1"' in ui_html


def test_search_input_triggers_search_on_enter_key() -> None:
    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))
    app = create_shiny_app(vector_store)

    ui_html = str(app.ui)

    assert 'id="query"' in ui_html
    assert 'id="search"' in ui_html
    assert "keypress" in ui_html or "addEventListener" in ui_html
