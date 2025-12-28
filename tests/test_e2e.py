"""End-to-end tests using headless browser.

Requires Playwright browsers to be installed:
    uv run playwright install chromium
"""

import multiprocessing
import time
from pathlib import Path

import pytest
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from playwright.sync_api import expect, sync_playwright

from epistemon.indexing import load_and_chunk_markdown
from epistemon.web import create_app


def _check_playwright_installed() -> bool:
    try:
        with sync_playwright() as p:
            p.chromium.launch(headless=True).close()
        return True
    except Exception:
        return False


def run_server() -> None:
    import uvicorn

    test_file = Path("tests/data/sample.md")
    chunks = load_and_chunk_markdown(test_file, chunk_size=500, chunk_overlap=100)
    vector_store = InMemoryVectorStore(FakeEmbeddings(size=384))
    vector_store.add_documents(chunks)

    app = create_app(vector_store)

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="error")


@pytest.mark.e2e
@pytest.mark.skipif(
    not _check_playwright_installed(),
    reason="Playwright browsers not installed. Run: uv run playwright install chromium",
)
def test_search_ui_workflow() -> None:
    server_process = multiprocessing.Process(target=run_server, daemon=True)
    server_process.start()

    time.sleep(2)

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()

            page.goto("http://127.0.0.1:8765")

            expect(page.locator("h1")).to_contain_text("Epistemon")

            query_input = page.locator("#query")
            query_input.fill("LangChain")

            page.locator("button:text('Search')").click()

            page.wait_for_selector(".result", timeout=5000)

            results = page.locator(".result").all()
            assert len(results) > 0

            first_result = page.locator(".result").first
            expect(first_result.locator(".result-content")).not_to_be_empty()
            expect(first_result.locator(".result-source")).not_to_be_empty()

            browser.close()
    finally:
        server_process.terminate()
        server_process.join(timeout=5)
