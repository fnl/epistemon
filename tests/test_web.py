from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from epistemon.indexing import load_and_chunk_markdown
from epistemon.indexing.vector_store_manager import create_vector_store_manager
from epistemon.web import create_app


@pytest.fixture
def vector_store() -> VectorStore:
    test_file = Path("tests/data/sample.md")
    base_directory = Path("tests/data")
    chunks = load_and_chunk_markdown(
        test_file, chunk_size=500, chunk_overlap=100, base_directory=base_directory
    )
    store = InMemoryVectorStore(FakeEmbeddings(size=384))
    store.add_documents(chunks)
    return store


def test_create_app_accepts_vector_store(vector_store: VectorStore) -> None:
    app = create_app(vector_store, score_threshold=-1.0)
    client = TestClient(app)

    response = client.get("/search", params={"q": "LangChain", "limit": 3})

    results = response.json()["results"]
    assert len(results) > 0
    assert all("score" in result for result in results)


def test_root_serves_html(vector_store: VectorStore) -> None:
    app = create_app(vector_store)
    client = TestClient(app)

    response = client.get("/")

    assert "text/html" in response.headers["content-type"]
    assert b"Epistemon" in response.content


def test_search_returns_results(vector_store: VectorStore) -> None:
    app = create_app(vector_store, score_threshold=-1.0)
    client = TestClient(app)

    response = client.get("/search", params={"q": "LangChain", "limit": 3})

    results = response.json()["results"]
    assert len(results) > 0
    assert len(results) <= 3


def test_search_respects_limit(vector_store: VectorStore) -> None:
    app = create_app(vector_store, score_threshold=0.0)
    client = TestClient(app)

    results_1 = client.get("/search", params={"q": "LangChain", "limit": 1}).json()[
        "results"
    ]
    results_3 = client.get("/search", params={"q": "LangChain", "limit": 3}).json()[
        "results"
    ]

    assert len(results_1) == 1
    assert len(results_3) <= 3


def test_search_results_ranked_by_score(vector_store: VectorStore) -> None:
    app = create_app(vector_store, score_threshold=-1.0)
    client = TestClient(app)

    response = client.get("/search", params={"q": "LangChain", "limit": 5})

    results = response.json()["results"]
    assert len(results) > 1, "Need multiple results to verify ordering"

    scores = [result["score"] for result in results]
    is_descending = scores == sorted(scores, reverse=True)
    is_ascending = scores == sorted(scores)
    assert (
        is_descending or is_ascending
    ), f"Scores should be monotonic but got: {scores}"


def test_search_handles_empty_query(vector_store: VectorStore) -> None:
    app = create_app(vector_store)
    client = TestClient(app)

    response = client.get("/search", params={"q": "", "limit": 5})

    assert response.json()["results"] == []


def test_search_filters_by_score_threshold(vector_store: VectorStore) -> None:
    client_low = TestClient(create_app(vector_store, score_threshold=-1.0))
    client_high = TestClient(create_app(vector_store, score_threshold=1000000.0))

    results_low = client_low.get(
        "/search", params={"q": "LangChain", "limit": 5}
    ).json()["results"]
    results_high = client_high.get(
        "/search", params={"q": "LangChain", "limit": 5}
    ).json()["results"]

    assert len(results_low) > 0
    assert len(results_high) == 0


def test_search_returns_alert_when_no_matches(vector_store: VectorStore) -> None:
    app = create_app(vector_store, score_threshold=1000000.0)
    client = TestClient(app)

    response = client.get("/search", params={"q": "LangChain", "limit": 5})

    assert response.status_code == 204
    data = response.json()
    assert data["results"] == []
    assert "alert" in data


def test_search_results_include_source_link(vector_store: VectorStore) -> None:
    base_url = "http://localhost:8000/files"
    app = create_app(vector_store, base_url=base_url, score_threshold=-1.0)
    client = TestClient(app)

    response = client.get("/search", params={"q": "LangChain", "limit": 1})

    results = response.json()["results"]
    assert len(results) > 0
    result = results[0]
    assert "link" in result
    assert result["link"].startswith(base_url)
    assert "sample.md" in result["link"]


def test_files_endpoint_serves_markdown() -> None:
    from pathlib import Path

    test_data_dir = Path("tests/data")
    vector_store_fixture = InMemoryVectorStore(FakeEmbeddings(size=384))
    app = create_app(vector_store_fixture, files_directory=test_data_dir)
    client = TestClient(app)

    response = client.get("/files/sample.md")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "").lower()
    assert b"<html" in response.content.lower()
    assert b"<h1>" in response.content.lower() or b"<h2>" in response.content.lower()


def test_end_to_end_search_and_file_retrieval(vector_store: VectorStore) -> None:
    test_data_dir = Path("tests/data")
    app = create_app(
        vector_store,
        base_url="http://testserver/files",
        files_directory=test_data_dir,
        score_threshold=-1.0,
    )
    client = TestClient(app)

    search_response = client.get("/search", params={"q": "LangChain", "limit": 1})
    search_results = search_response.json()["results"]
    assert len(search_results) > 0

    result = search_results[0]
    assert "link" in result
    link = result["link"]

    file_path = link.replace("http://testserver/files/", "")
    file_response = client.get(f"/files/{file_path}")

    assert file_response.status_code == 200
    content = file_response.text
    assert "LangChain" in content or len(content) > 0


def test_search_results_include_metadata(vector_store: VectorStore) -> None:
    app = create_app(vector_store, score_threshold=-1.0)
    client = TestClient(app)

    response = client.get("/search", params={"q": "LangChain", "limit": 1})

    results = response.json()["results"]
    assert len(results) > 0
    result = results[0]
    assert "source" in result
    assert "last_modified" in result
    assert isinstance(result["last_modified"], (int, float))
    assert result["last_modified"] > 0


def test_ui_displays_metadata() -> None:
    from pathlib import Path

    html_path = Path("epistemon/web/static/index.html")
    html_content = html_path.read_text()

    assert "result.last_modified" in html_content


def test_files_endpoint_returns_list_of_indexed_files(
    vector_store: VectorStore,
) -> None:
    app = create_app(vector_store)
    client = TestClient(app)

    response = client.get("/files")

    assert response.status_code == 200
    files = response.json()["files"]
    assert isinstance(files, list)
    assert len(files) > 0
    assert any("sample.md" in file["source"] for file in files)


def test_files_endpoint_includes_all_metadata(vector_store: VectorStore) -> None:
    from datetime import datetime

    app = create_app(vector_store)
    client = TestClient(app)

    response = client.get("/files")

    files = response.json()["files"]
    assert len(files) > 0
    for file in files:
        assert "source" in file
        assert "last_modified" in file
        assert isinstance(file["source"], str)
        assert isinstance(file["last_modified"], str)
        datetime.fromisoformat(file["last_modified"])


def test_files_endpoint_handles_empty_index() -> None:
    empty_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    app = create_app(empty_store)
    client = TestClient(app)

    response = client.get("/files")

    assert response.status_code == 200
    files = response.json()["files"]
    assert isinstance(files, list)
    assert len(files) == 0


def test_files_endpoint_can_sort_by_name() -> None:
    from langchain_core.documents import Document

    store = InMemoryVectorStore(FakeEmbeddings(size=384))
    docs = [
        Document(page_content="C", metadata={"source": "zebra.md", "last_modified": 3}),
        Document(page_content="B", metadata={"source": "alpha.md", "last_modified": 2}),
        Document(page_content="A", metadata={"source": "gamma.md", "last_modified": 1}),
    ]
    store.add_documents(docs)
    app = create_app(store)
    client = TestClient(app)

    response = client.get("/files", params={"sort_by": "name"})

    files = response.json()["files"]
    assert len(files) == 3
    sources = [f["source"] for f in files]
    assert sources == ["alpha.md", "gamma.md", "zebra.md"]


def test_files_endpoint_can_sort_by_date() -> None:
    from datetime import datetime

    from langchain_core.documents import Document

    store = InMemoryVectorStore(FakeEmbeddings(size=384))
    docs = [
        Document(page_content="C", metadata={"source": "zebra.md", "last_modified": 3}),
        Document(page_content="B", metadata={"source": "alpha.md", "last_modified": 2}),
        Document(page_content="A", metadata={"source": "gamma.md", "last_modified": 1}),
    ]
    store.add_documents(docs)
    app = create_app(store)
    client = TestClient(app)

    response = client.get("/files", params={"sort_by": "date"})

    files = response.json()["files"]
    assert len(files) == 3
    modified_times = [f["last_modified"] for f in files]
    expected_times = [
        datetime.fromtimestamp(3).isoformat(),
        datetime.fromtimestamp(2).isoformat(),
        datetime.fromtimestamp(1).isoformat(),
    ]
    assert modified_times == expected_times


def test_files_endpoint_with_vector_store_manager() -> None:
    base_directory = Path("tests/data")
    store = InMemoryVectorStore(FakeEmbeddings(size=384))
    docs = [
        Document(
            page_content="Content A",
            metadata={"source": "file_a.md", "last_modified": 1234567890.0},
        ),
        Document(
            page_content="Content B",
            metadata={"source": "file_b.md", "last_modified": 1234567900.0},
        ),
    ]
    store.add_documents(docs)
    manager = create_vector_store_manager(store, base_directory)
    app = create_app(store, vector_store_manager=manager)
    client = TestClient(app)

    response = client.get("/files")

    files = response.json()["files"]
    assert len(files) == 2
    sources = {f["source"] for f in files}
    assert "file_a.md" in sources
    assert "file_b.md" in sources


def test_files_endpoint_with_manager_sorts_correctly() -> None:
    from datetime import datetime

    base_directory = Path("tests/data")
    store = InMemoryVectorStore(FakeEmbeddings(size=384))
    docs = [
        Document(
            page_content="C",
            metadata={"source": "zebra.md", "last_modified": 3.0},
        ),
        Document(
            page_content="B",
            metadata={"source": "alpha.md", "last_modified": 2.0},
        ),
    ]
    store.add_documents(docs)
    manager = create_vector_store_manager(store, base_directory)
    app = create_app(store, vector_store_manager=manager)
    client = TestClient(app)

    response = client.get("/files", params={"sort_by": "name"})
    files = response.json()["files"]
    sources = [f["source"] for f in files]
    assert sources == ["alpha.md", "zebra.md"]

    response = client.get("/files", params={"sort_by": "date"})
    files = response.json()["files"]
    modified_times = [f["last_modified"] for f in files]
    expected_times = [
        datetime.fromtimestamp(3.0).isoformat(),
        datetime.fromtimestamp(2.0).isoformat(),
    ]
    assert modified_times == expected_times


def test_files_endpoint_returns_iso_formatted_dates() -> None:
    from datetime import datetime

    store = InMemoryVectorStore(FakeEmbeddings(size=384))
    timestamp = 1234567890.5
    docs = [
        Document(
            page_content="Content",
            metadata={"source": "test.md", "last_modified": timestamp},
        ),
    ]
    store.add_documents(docs)
    app = create_app(store)
    client = TestClient(app)

    response = client.get("/files")

    files = response.json()["files"]
    assert len(files) == 1
    last_modified = files[0]["last_modified"]
    assert isinstance(last_modified, str)
    parsed_date = datetime.fromisoformat(last_modified)
    assert parsed_date == datetime.fromtimestamp(timestamp)


def test_search_endpoint_handles_missing_query_parameter(
    vector_store: VectorStore,
) -> None:
    app = create_app(vector_store)
    client = TestClient(app)

    response = client.get("/search")

    assert response.status_code == 422


def test_search_endpoint_handles_invalid_limit_parameter(
    vector_store: VectorStore,
) -> None:
    app = create_app(vector_store)
    client = TestClient(app)

    response = client.get("/search", params={"q": "test", "limit": "invalid"})

    assert response.status_code == 422


def test_search_endpoint_handles_vector_store_errors(
    vector_store: VectorStore,
) -> None:
    from unittest.mock import Mock

    broken_store = Mock(spec=VectorStore)
    broken_store.similarity_search_with_score.side_effect = Exception(
        "Vector store error"
    )
    app = create_app(broken_store)
    client = TestClient(app)

    response = client.get("/search", params={"q": "test", "limit": 5})

    assert response.status_code == 500
    assert "error" in response.json()
