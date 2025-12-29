"""FastAPI application for semantic search."""

from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.vectorstores import VectorStoreRetriever

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    retriever: VectorStoreRetriever,
    base_url: str = "",
    score_threshold: float = 0.0,
    files_directory: Path | None = None,
) -> FastAPI:
    app = FastAPI()

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    if files_directory is not None:
        app.mount(
            "/files",
            StaticFiles(directory=str(files_directory)),
            name="files",
        )

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

        results = []
        for doc, score in results_with_scores:
            if score >= score_threshold:
                source = doc.metadata.get("source", "")
                result = {
                    "content": doc.page_content,
                    "source": source,
                    "score": float(score),
                }
                if base_url and source:
                    result["link"] = f"{base_url}/{quote(source)}"
                results.append(result)

        response: dict[str, list[dict[str, str | float]] | str] = {"results": results}

        if score_threshold > 0 and len(results) == 0:
            response["alert"] = "No results found matching the score threshold"
            return JSONResponse(content=response, status_code=204)

        return response

    return app
