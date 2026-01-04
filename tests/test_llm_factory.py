"""Tests for the LLM factory module."""

from langchain_core.language_models import FakeListLLM
from langchain_openai import ChatOpenAI

from epistemon.config import Configuration
from epistemon.llm_factory import create_llm


def test_create_fake_llm() -> None:
    config = Configuration(
        input_directory="./tests/data",
        vector_store_type="inmemory",
        vector_store_path="./data/test",
        embedding_provider="fake",
        embedding_model="fake-model",
        chunk_size=500,
        chunk_overlap=200,
        search_results_limit=5,
        score_threshold=0.0,
        bm25_k1=1.5,
        bm25_b=0.75,
        bm25_top_k=5,
        hybrid_bm25_weight=0.3,
        hybrid_semantic_weight=0.7,
        llm_provider="fake",
        llm_model="fake-llm",
        llm_temperature=0.0,
        rag_enabled=True,
        rag_max_context_docs=10,
        rag_prompt_template_path="./prompts/default.txt",
    )

    llm = create_llm(config)

    assert isinstance(llm, FakeListLLM)


def test_create_openai_llm() -> None:
    config = Configuration(
        input_directory="./tests/data",
        vector_store_type="inmemory",
        vector_store_path="./data/test",
        embedding_provider="fake",
        embedding_model="fake-model",
        chunk_size=500,
        chunk_overlap=200,
        search_results_limit=5,
        score_threshold=0.0,
        bm25_k1=1.5,
        bm25_b=0.75,
        bm25_top_k=5,
        hybrid_bm25_weight=0.3,
        hybrid_semantic_weight=0.7,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_temperature=0.7,
        rag_enabled=True,
        rag_max_context_docs=10,
        rag_prompt_template_path="./prompts/default.txt",
    )

    llm = create_llm(config)

    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o-mini"
    assert llm.temperature == 0.7
