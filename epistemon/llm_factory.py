"""LLM factory for creating the appropriate LLM based on configuration."""

from typing import Any

from langchain_core.language_models import FakeListLLM
from langchain_core.language_models.base import BaseLanguageModel

from epistemon.config import Configuration


def create_llm(config: Configuration) -> BaseLanguageModel[Any]:
    if config.llm_provider == "fake":
        return FakeListLLM(responses=["Test response"])
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
