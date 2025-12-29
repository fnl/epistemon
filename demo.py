"""Demo script to run the web UI with sample data."""

from pathlib import Path

import uvicorn

from epistemon.config import load_config
from epistemon.indexing.indexer import index
from epistemon.vector_store_factory import create_vector_store
from epistemon.web import create_app


def main() -> None:
    print("Loading default configuration...")
    config = load_config()

    print(f"Creating vector store ({config.vector_store_type})...")
    vector_store = create_vector_store(config)

    print(f"Indexing {config.input_directory}...")
    index(config, vector_store)

    print("\nIndexing complete!")

    print("Creating retriever...")
    retriever = vector_store.as_retriever()

    app = create_app(
        retriever,
        base_url="http://localhost:8000/files",
        score_threshold=config.score_threshold,
        files_directory=Path(config.input_directory),
    )

    print("\n" + "=" * 60)
    print("Demo server starting...")
    print("Open http://localhost:8000 in your browser")
    print("\nTry searching for:")
    print("  - LangChain")
    print("  - embeddings")
    print("  - vector stores")
    print("  - semantic search")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
