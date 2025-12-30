from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from shiny import App

from epistemon.web.shiny_app import create_shiny_app


def test_create_shiny_app_returns_app_instance() -> None:
    vector_store: VectorStore = InMemoryVectorStore(FakeEmbeddings(size=384))

    app = create_shiny_app(vector_store)

    assert isinstance(app, App)
