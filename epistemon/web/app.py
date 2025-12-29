"""FastAPI application for semantic search."""

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.vectorstores import VectorStore

STATIC_DIR = Path(__file__).parent / "static"


def create_app(vector_store: VectorStore) -> FastAPI:
    app = FastAPI()

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/search")
    def search_endpoint(
        q: str = Query(..., description="Search query"),
        limit: int = Query(5, description="Maximum number of results"),
    ) -> dict[str, list[dict[str, str]]]:
        retriever = vector_store.as_retriever(search_kwargs={"k": limit})
        results = retriever.invoke(q)

        return {
            "results": [
                {"content": doc.page_content, "source": doc.metadata.get("source", "")}
                for doc in results
            ]
        }

    return app
