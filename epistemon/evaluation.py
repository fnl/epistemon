"""RAG evaluation module with LLM-based judge scoring."""

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class JudgeScore:
    """Score and reasoning from an LLM judge."""

    score: float
    reason: str


def score_context_relevance(llm: Any, question: str, context: str) -> JudgeScore:
    """Score how relevant the context is to the question.

    Args:
        llm: LLM implementing invoke(prompt) -> response with .content attribute
        question: The user question
        context: The retrieved context to evaluate

    Returns:
        JudgeScore with a 0.0-1.0 score and reasoning, or score=0.0 on parse error
    """
    prompt = (
        "You are an evaluation judge. Score how relevant the following context is "
        "to the question on a scale from 0.0 to 1.0.\n\n"
        f"Question: {question}\n\n"
        f"Context: {context}\n\n"
        'Respond with valid JSON only: {"score": <float>, "reason": "<string>"}'
    )
    response = llm.invoke(prompt)
    try:
        parsed = json.loads(response.content)
        return JudgeScore(score=parsed["score"], reason=parsed["reason"])
    except (json.JSONDecodeError, KeyError):
        return JudgeScore(score=0.0, reason="parse error")


def score_answer_faithfulness(
    llm: Any, question: str, answer: str, context: str
) -> JudgeScore:
    """Score how faithful the answer is to the context.

    Args:
        llm: LLM implementing invoke(prompt) -> response with .content attribute
        question: The user question
        answer: The generated answer to evaluate
        context: The context the answer was generated from

    Returns:
        JudgeScore with a 0.0-1.0 score and reasoning
    """
    prompt = (
        "You are an evaluation judge. Score how faithful the following answer is "
        "to the provided context on a scale from 0.0 to 1.0.\n\n"
        f"Question: {question}\n\n"
        f"Context: {context}\n\n"
        f"Answer: {answer}\n\n"
        'Respond with valid JSON only: {"score": <float>, "reason": "<string>"}'
    )
    response = llm.invoke(prompt)
    try:
        parsed = json.loads(response.content)
        return JudgeScore(score=parsed["score"], reason=parsed["reason"])
    except (json.JSONDecodeError, KeyError):
        return JudgeScore(score=0.0, reason="parse error")
