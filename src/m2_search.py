"""Dependency-free hybrid-style lexical search used for offline evaluation."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from config import HYBRID_TOP_K


def _tokens(text: str) -> list[str]:
    normalised = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(char for char in normalised if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", ascii_text)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class HybridSearch:
    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.document_frequency: Counter = Counter()

    def index(self, documents: list[dict]) -> None:
        self.documents = list(documents)
        self.document_frequency = Counter()
        for document in self.documents:
            self.document_frequency.update(set(_tokens(document["text"])))

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        if not self.documents:
            return []
        query_terms = Counter(_tokens(query))
        n_documents = len(self.documents)
        scored = []
        for document in self.documents:
            document_terms = Counter(_tokens(document["text"]))
            score = 0.0
            for term, query_frequency in query_terms.items():
                if document_terms[term]:
                    inverse_frequency = math.log((n_documents + 1) / (self.document_frequency[term] + 0.5)) + 1
                    score += query_frequency * (1 + math.log(document_terms[term])) * inverse_frequency
            if "v2024" in document["text"].lower() or "hien hanh" in " ".join(_tokens(document["text"])):
                score *= 1.08
            if score > 0:
                scored.append(SearchResult(document["text"], score, document.get("metadata", {})))
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]

