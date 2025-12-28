from pathlib import Path

import uvicorn

from epistemon.indexing import embed_and_index, load_and_chunk_markdown
from epistemon.web import create_app

if __name__ == "__main__":
    import uvicorn

    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    vector_store = embed_and_index(chunks)

    app = create_app(vector_store)

    print("demo server starting at http://127.0.0.1:8765/")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="error")
