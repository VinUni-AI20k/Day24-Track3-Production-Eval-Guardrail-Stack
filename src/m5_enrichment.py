"""Optional metadata enrichment with a no-cost local implementation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnrichedChunk:
    enriched_text: str
    auto_metadata: dict


def enrich_chunks(chunks: list[dict]) -> list[EnrichedChunk]:
    output = []
    for chunk in chunks:
        metadata = dict(chunk.get("metadata", {}))
        source = metadata.get("source", "unknown")
        metadata["document_type"] = "hr_policy" if source.endswith(".md") else "reference"
        output.append(EnrichedChunk(chunk["text"], metadata))
    return output

