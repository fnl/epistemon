"""Demo script to run the web UI with sample data."""

from pathlib import Path

import uvicorn

from epistemon.config import load_config
from epistemon.indexing import collect_markdown_files, load_and_chunk_markdown
from epistemon.vector_store_factory import create_vector_store
from epistemon.web import create_app


def main() -> None:
    print("Loading default configuration...")
    config = load_config()

    print(f"Creating vector store ({config.vector_store_type})...")
    vector_store = create_vector_store(config)

    print(f"Scanning {config.input_directory} for markdown files...")
    base_directory = Path(config.input_directory)
    markdown_files = collect_markdown_files(base_directory)

    print(f"Found {len(markdown_files)} markdown files")

    total_chunks = 0
    for file in markdown_files:
        chunks = load_and_chunk_markdown(
            file,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            base_directory=base_directory,
        )
        if chunks:
            vector_store.add_documents(chunks)
            total_chunks += len(chunks)
            print(f"  Indexed {file.name}: {len(chunks)} chunks")

    print(f"\nTotal indexed: {total_chunks} chunks from {len(markdown_files)} files")

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
