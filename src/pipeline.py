"""Simple local RAG pipeline composed from the Day 18-compatible modules."""
from __future__ import annotations

from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m5_enrichment import enrich_chunks


class RAGPipeline:
    def __init__(self) -> None:
        chunks = []
        for document in load_documents():
            _, children = chunk_hierarchical(document["text"], document["metadata"])
            chunks.extend({"text": child.text, "metadata": child.metadata} for child in children)
        enriched = enrich_chunks(chunks)
        self.searcher = HybridSearch()
        self.searcher.index([
            {"text": chunk.enriched_text, "metadata": chunk.auto_metadata}
            for chunk in enriched
        ])
        self.reranker = CrossEncoderReranker()

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        candidates = self.searcher.search(question)
        documents = [
            {"text": item.text, "score": item.score, "metadata": item.metadata}
            for item in candidates
        ]
        return [item.text for item in self.reranker.rerank(question, documents, top_k)]

