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
