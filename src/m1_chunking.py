"""Local document loading and hierarchical chunking for the Day 24 lab."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from config import DATA_DIR, HIERARCHICAL_CHILD_SIZE, HIERARCHICAL_PARENT_SIZE


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    documents = []
    for path in sorted(Path(data_dir).glob("*")):
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
            except (ImportError, OSError, ValueError):
                continue
        else:
            continue
        if text.strip():
            documents.append({
                "text": text,
                "metadata": {"source": path.name, "path": str(path)},
            })
    return documents


def _split_words(text: str, max_words: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        while words:
            remaining = max_words - len(current)
            current.extend(words[:remaining])
            words = words[remaining:]
            if len(current) >= max_words:
                chunks.append(" ".join(current))
                current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_hierarchical(text: str, metadata: dict | None = None,
                       parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE) -> tuple[list[Chunk], list[Chunk]]:
    metadata = dict(metadata or {})
    parent_words = max(100, parent_size // 5)
    child_words = max(40, child_size // 4)
    parents = []
    children = []
    source = metadata.get("source", "document")
    for parent_index, parent_text in enumerate(_split_words(text, parent_words)):
        parent_id = f"{source}:p{parent_index}"
        parents.append(Chunk(parent_text, {**metadata, "level": "parent"}, parent_id))
        for child_index, child_text in enumerate(_split_words(parent_text, child_words)):
            children.append(Chunk(
                child_text,
                {**metadata, "level": "child", "child_index": child_index},
                parent_id,
            ))
    return parents, children

