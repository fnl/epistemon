"""LLM factory for creating the appropriate LLM based on configuration."""

from typing import Any, cast

from langchain_core.language_models import FakeListLLM
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI

from epistemon.config import Configuration


def create_llm(config: Configuration) -> BaseLanguageModel[Any]:
    if config.llm_provider == "fake":
        return FakeListLLM(responses=["Test response"])
    elif config.llm_provider == "openai":
        openai_llm: BaseLanguageModel[Any] = cast(
            BaseLanguageModel[Any],
            ChatOpenAI(
                model=config.llm_model,  # type: ignore[call-arg]
                temperature=config.llm_temperature,
            ),
        )
        return openai_llm
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
