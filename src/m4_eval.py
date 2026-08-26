"""RAG evaluation with batched LLM scoring and a deterministic fallback."""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass

from config import OPENAI_API_KEY


@dataclass
class EvaluationScore:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def _tokens(text: str) -> set[str]:
    normalised = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(char for char in normalised if not unicodedata.combining(char))
    stopwords = {"la", "va", "cua", "cho", "theo", "duoc", "mot", "nhan", "vien"}
    return {token for token in re.findall(r"[a-z0-9]+", ascii_text) if len(token) > 1 and token not in stopwords}


def _coverage(source: str, target: str) -> float:
    source_tokens = _tokens(source)
    target_tokens = _tokens(target)
    return len(source_tokens & target_tokens) / max(len(source_tokens), 1)


def _offline_score(question: str, answer: str, contexts: list[str], ground_truth: str) -> EvaluationScore:
    context = " ".join(contexts)
    faithfulness = _coverage(answer, context)
    answer_relevancy = 0.5 * _coverage(question, answer) + 0.5 * _coverage(ground_truth, answer)
    relevant_contexts = [_coverage(ground_truth, item) for item in contexts]
    context_precision = sum(score >= 0.12 for score in relevant_contexts) / max(len(contexts), 1)
    context_recall = _coverage(ground_truth, context)
    return EvaluationScore(*[round(max(0.0, min(1.0, value)), 4) for value in (
        faithfulness, answer_relevancy, context_precision, context_recall
    )])


def _llm_score_batch(batch: list[dict]) -> list[EvaluationScore]:
    from src.openai_client import chat_json

    compact = [
        {
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "contexts": [context[:2500] for context in item["contexts"]],
            "ground_truth": item["ground_truth"],
        }
        for item in batch
    ]
    response = chat_json(
        "You evaluate Vietnamese HR-policy RAG answers. Score strictly and consistently.",
        "For each item, assign four scores from 0 to 1: faithfulness (answer supported by contexts), "
        "answer_relevancy (answers the question), context_precision (retrieved contexts are focused), "
        "and context_recall (contexts cover the ground truth). Return JSON object with a 'scores' array "
        "in the same order, each containing id and the four numeric fields. Items:\n"
        + json.dumps(compact, ensure_ascii=False),
        timeout=120,
    )
    by_id = {int(item["id"]): item for item in response.get("scores", [])}
    scores = []
    for item in batch:
        raw = by_id[item["id"]]
        values = [
            max(0.0, min(1.0, float(raw[name])))
            for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        ]
        scores.append(EvaluationScore(*values))
    return scores


def evaluate_ragas(questions: list[str], answers: list[str], contexts: list[list[str]],
                   ground_truths: list[str]) -> dict:
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("questions, answers, contexts and ground_truths must have equal lengths")

    items = [
        {"id": index, "question": question, "answer": answer,
         "contexts": context, "ground_truth": ground_truth}
        for index, (question, answer, context, ground_truth)
        in enumerate(zip(questions, answers, contexts, ground_truths))
    ]
    results: list[EvaluationScore] = []
    mode = "offline_deterministic"
    use_llm = bool(OPENAI_API_KEY and os.getenv("USE_OPENAI_EVAL", "true").lower() in {"1", "true", "yes"})
    for start in range(0, len(items), 10):
        batch = items[start:start + 10]
        if use_llm:
            try:
                results.extend(_llm_score_batch(batch))
                mode = "openai_batched"
                continue
            except Exception as error:
                print(f"LLM evaluation batch failed, using deterministic fallback: {error}")
        results.extend(_offline_score(item["question"], item["answer"], item["contexts"], item["ground_truth"])
                       for item in batch)

    averages = {
        name: round(sum(getattr(result, name) for result in results) / max(len(results), 1), 4)
        for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    }
    return {"per_question": results, "averages": averages, "mode": mode}

