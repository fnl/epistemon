"""Demo script to run the web UI with sample data."""

from pathlib import Path

import uvicorn

from epistemon.indexing import embed_and_index, load_and_chunk_markdown
from epistemon.web import create_app


def main() -> None:
    print("Loading sample markdown file...")
    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)

    print(f"Indexing {len(chunks)} chunks...")
    vector_store = embed_and_index(chunks)

    app = create_app(vector_store)

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
