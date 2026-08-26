from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, re, sys, json as _json
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_API_KEY, LLM_MODEL, get_llm_client, ENRICH_WITH_LLM


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Helpers ─────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]


def _chat(system: str, user: str, max_tokens: int, json_mode: bool = False) -> str | None:
    """Gọi LLM (Groq/OpenAI-compatible). Trả None nếu không có key hoặc lỗi."""
    if not LLM_API_KEY or not ENRICH_WITH_LLM:
        return None
    try:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = get_llm_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  ⚠️  LLM call failed: {e}")
        return None


_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache",
    f"enrichment_{LLM_MODEL.replace('/', '_')}.json")
_CACHE: dict | None = None


def _cache_get(key: str):
    """Cache enrichment trên đĩa: đổi prompt/generation không phải trả tiền
    enrich lại 105 chunk (~6 phút + phí API)."""
    global _CACHE
    if _CACHE is None:
        try:
            with open(_CACHE_PATH, encoding="utf-8") as f:
                _CACHE = _json.load(f)
        except (FileNotFoundError, _json.JSONDecodeError):
            _CACHE = {}
    return _CACHE.get(key)


def _cache_put(key: str, value: dict) -> None:
    if _CACHE is None:
        return
    _CACHE[key] = value
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w", encoding="utf-8") as f:
        _json.dump(_CACHE, f, ensure_ascii=False)


def _strip_reserved(meta: dict) -> dict:
    """Metadata do LLM sinh KHÔNG được ghi đè `source` — đó là bằng chứng để
    xử lý xung đột phiên bản (v2023 vs v2024)."""
    if not isinstance(meta, dict):
        return {}
    return {k: v for k, v in meta.items() if k not in ("source", "parent_id")}


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    out = _chat("Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
                text, max_tokens=150)
    if out:
        return out

    # Extractive fallback (không cần API): lấy 2 câu đầu
    sentences = _split_sentences(text)
    if not sentences:
        return text
    return " ".join(sentences[:2])


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    out = _chat(
        f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
        "Trả về mỗi câu hỏi trên 1 dòng, không đánh số.",
        text, max_tokens=200)
    if out:
        questions = [q.strip().lstrip("0123456789.-) ") for q in out.split("\n") if q.strip()]
        if questions:
            return questions[:n_questions]

    # Extractive fallback: biến câu khẳng định thành câu hỏi thô
    sentences = [s for s in _split_sentences(text) if len(s) > 10]
    return [f"{s.rstrip('.!?')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    out = _chat(
        "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. "
        "Chỉ trả về 1 câu.",
        f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", max_tokens=80)
    if out:
        return f"{out}\n\n{text}"

    # Fallback: prepend tên tài liệu — vẫn hữu ích cho BM25 (phân biệt v2023/v2024)
    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    out = _chat(
        'Trích xuất metadata từ đoạn văn. Chỉ trả về JSON: '
        '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}',
        text, max_tokens=150, json_mode=True)
    if out:
        try:
            return _strip_reserved(_json.loads(out))
        except _json.JSONDecodeError as e:
            print(f"  ⚠️  Metadata JSON parse failed: {e}")

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    import hashlib

    ckey = hashlib.sha256(f"{source}\x00{text}".encode()).hexdigest()
    cached = _cache_get(ckey)
    if cached is not None:
        return cached

    out = _chat(
        """Phân tích đoạn văn và chỉ trả về JSON:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}""",
        f"Tài liệu: {source}\n\nĐoạn văn:\n{text}", max_tokens=400, json_mode=True)

    if out:
        try:
            data = _json.loads(out)
            data["metadata"] = _strip_reserved(data.get("metadata", {}))
            _cache_put(ckey, data)
            return data
        except _json.JSONDecodeError as e:
            print(f"  ⚠️  Enrichment JSON parse failed: {e}")

    # Fallback offline (không API key): vẫn enrich được bằng extractive + heuristic
    sentences = _split_sentences(text)
    return {
        "summary": " ".join(sentences[:2]) if sentences else text,
        "questions": [f"{s.rstrip('.!?')}?" for s in sentences[:3] if len(s) > 10],
        "context": f"Trích từ {source}." if source else "",
        "metadata": {"topic": os.path.splitext(source)[0] if source else "general",
                     "entities": [], "category": "policy", "language": "vi",
                     "enrichment": "fallback"},
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
