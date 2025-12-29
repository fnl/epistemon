"""FastAPI application for semantic search."""

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.vectorstores import VectorStoreRetriever

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    retriever: VectorStoreRetriever, score_threshold: float = 0.0
) -> FastAPI:
    app = FastAPI()

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/search", response_model=None)
    def search_endpoint(
        q: str = Query(..., description="Search query"),
        limit: int = Query(5, description="Maximum number of results"),
    ) -> dict[str, list[dict[str, str | float]] | str] | JSONResponse:
        if not q or not q.strip():
            return {"results": []}

        results_with_scores = retriever.vectorstore.similarity_search_with_score(
            q, k=limit
        )

        results = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "score": float(score),
            }
            for doc, score in results_with_scores
            if score >= score_threshold
        ]

        response: dict[str, list[dict[str, str | float]] | str] = {"results": results}

        if score_threshold > 0 and len(results) == 0:
            response["alert"] = "No results found matching the score threshold"
            return JSONResponse(content=response, status_code=204)

        return response

    return app
