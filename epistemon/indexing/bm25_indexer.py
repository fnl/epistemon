"""In-memory BM25 indexer for keyword-based search."""

import logging
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from epistemon.indexing.chunker import load_and_chunk_markdown
from epistemon.indexing.file_tracker import collect_markdown_files

logger = logging.getLogger(__name__)


class BM25Indexer:
    """In-memory BM25 indexer built from markdown files on disk."""

    def __init__(
        self, directory: Path, chunk_size: int = 500, chunk_overlap: int = 100
    ) -> None:
        self.directory = directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: list[Document] = []
        self.bm25_index: BM25Okapi | None = None
        self._build_index()

    def _load_documents_from_disk(self) -> list[Document]:
        documents = []
        markdown_files = collect_markdown_files(self.directory)

        for file_path in markdown_files:
            chunks = load_and_chunk_markdown(
                file_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                base_directory=self.directory,
            )
            documents.extend(chunks)

        return documents

    def _build_index(self) -> None:
        self.documents = self._load_documents_from_disk()

        if not self.documents:
            logger.warning("No documents found in directory for BM25 indexing")
            return

        tokenized_corpus = [doc.page_content.split() for doc in self.documents]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        logger.info("BM25 index built with %d documents", len(self.documents))

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        if not self.bm25_index or not self.documents:
            return []

        tokenized_query = query.split()
        scores = self.bm25_index.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        return [(self.documents[i], float(scores[i])) for i in top_indices]
