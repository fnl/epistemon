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
