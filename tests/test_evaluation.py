"""Tests for the RAG evaluation module."""

import json
from unittest.mock import Mock

from epistemon.evaluation import (
    JudgeScore,
    RetrievalJudge,
    score_answer_faithfulness,
    score_context_relevance,
)


def test_score_context_relevance_returns_judge_score_from_valid_json() -> None:
    """score_context_relevance returns a JudgeScore parsed from LLM JSON output."""
    llm = Mock()
    payload = {"score": 0.85, "reason": "The context directly addresses the question."}
    llm.invoke.return_value = Mock(content=json.dumps(payload))

    result = score_context_relevance(llm, "What is Python?", "Python is a language.")

    assert isinstance(result, JudgeScore)
    assert result.score == 0.85
    assert result.reason == "The context directly addresses the question."


def test_score_context_relevance_returns_fallback_on_invalid_json() -> None:
    """score_context_relevance returns score=0.0 and reason='parse error' on invalid JSON."""
    llm = Mock()
    llm.invoke.return_value = Mock(content="not valid json at all")

    result = score_context_relevance(llm, "What is Python?", "Python is a language.")

    assert result.score == 0.0
    assert result.reason == "parse error"


def test_score_answer_faithfulness_returns_fallback_on_invalid_json() -> None:
    """score_answer_faithfulness returns score=0.0 and reason='parse error' on invalid JSON."""
    llm = Mock()
    llm.invoke.return_value = Mock(content="not valid json at all")

    result = score_answer_faithfulness(
        llm,
        "What is Python?",
        "Python is a language.",
        "Python is a programming language.",
    )

    assert result.score == 0.0
    assert result.reason == "parse error"


def test_score_answer_faithfulness_returns_judge_score_from_valid_json() -> None:
    """score_answer_faithfulness returns a JudgeScore parsed from LLM JSON output."""
    llm = Mock()
    payload = {"score": 0.9, "reason": "The answer is fully supported by the context."}
    llm.invoke.return_value = Mock(content=json.dumps(payload))

    result = score_answer_faithfulness(
        llm,
        "What is Python?",
        "Python is a language.",
        "Python is a programming language.",
    )

    assert isinstance(result, JudgeScore)
    assert result.score == 0.9
    assert result.reason == "The answer is fully supported by the context."


def test_retrieval_judge_delegates_score_context_relevance() -> None:
    """RetrievalJudge.score_context_relevance returns JudgeScore from its LLM."""
    llm = Mock()
    payload = {"score": 0.7, "reason": "Mostly relevant."}
    llm.invoke.return_value = Mock(content=json.dumps(payload))

    judge = RetrievalJudge(llm)
    result = judge.score_context_relevance("What is Python?", "Python is a language.")

    assert isinstance(result, JudgeScore)
    assert result.score == 0.7
    assert result.reason == "Mostly relevant."


def test_retrieval_judge_delegates_score_answer_faithfulness() -> None:
    """RetrievalJudge.score_answer_faithfulness returns JudgeScore from its LLM."""
    llm = Mock()
    payload = {"score": 0.8, "reason": "Supported by context."}
    llm.invoke.return_value = Mock(content=json.dumps(payload))

    judge = RetrievalJudge(llm)
    result = judge.score_answer_faithfulness(
        "What is Python?", "Python is a language.", "Python is a programming language."
    )

    assert isinstance(result, JudgeScore)
    assert result.score == 0.8
    assert result.reason == "Supported by context."
