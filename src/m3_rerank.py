"""Fast deterministic reranker with the Day 18-compatible interface."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.m2_search import _tokens


@dataclass
class RerankResult:
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class CrossEncoderReranker:
    def rerank(self, query: str, documents: list[dict], top_k: int = 3) -> list[RerankResult]:
        query_tokens = set(_tokens(query))
        output = []
        for document in documents:
            document_tokens = set(_tokens(document["text"]))
            overlap = len(query_tokens & document_tokens) / max(len(query_tokens), 1)
            score = 0.65 * overlap + 0.35 * float(document.get("score", 0.0)) / 20
            output.append(RerankResult(document["text"], score, document.get("metadata", {})))
        return sorted(output, key=lambda result: result.score, reverse=True)[:top_k]

