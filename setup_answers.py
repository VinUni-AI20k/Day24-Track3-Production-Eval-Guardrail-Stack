"""
Setup script: chạy Day 18 pipeline trên 50 câu hỏi → lưu answers_50q.json

Chạy TRƯỚC khi bắt đầu Phase A:
    python setup_answers.py

Yêu cầu:
    1. Đã copy src/ từ Day 18 (m1-m5, pipeline.py) vào thư mục này
    2. docker compose up -d  (Qdrant đang chạy trên port 6333)
    3. .env có OPENAI_API_KEY
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_day18_files() -> bool:
    required = [
        "src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
        "src/m4_eval.py",     "src/m5_enrichment.py", "src/pipeline.py",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("\n❌ Thiếu files từ Day 18. Copy chúng vào src/ trước:\n")
        for f in missing:
            print(f"   cp <Day18>/src/{os.path.basename(f)} src/")
        return False
    print(f"✓ Day 18 source files: {len(required)}/{len(required)} found")
    return True


def build_pipeline():
    from src.m1_chunking import load_documents, chunk_hierarchical
    from src.m2_search import HybridSearch
    from src.m3_rerank import CrossEncoderReranker
    from src.m5_enrichment import enrich_chunks
    from config import RERANK_TOP_K

    print("\n[1/3] Chunking + enriching documents...")
    t0 = time.time()
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id},
            })

    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)")
    else:
        print(f"  ✓ Using {len(all_chunks)} raw chunks (M5 not implemented or no API key)")

    print("\n[2/3] Indexing (BM25 + Dense)...")
    t0 = time.time()
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed {len(all_chunks)} chunks ({time.time()-t0:.1f}s)")

    print("\n[3/3] Loading reranker...")
    t0 = time.time()
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)")

    return search, reranker, RERANK_TOP_K


def run_query(q: str, search, reranker, top_k: int) -> tuple[str, list[str]]:
    from config import OPENAI_API_KEY

    results = search.search(q)
    docs    = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(q, docs, top_k=top_k)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    if OPENAI_API_KEY and contexts:
        try:
            from src.openai_client import chat_json
            context_text = "\n\n".join(contexts)
            payload = chat_json(
                "Tra loi cau hoi HR chi dua tren context. Tra ve JSON co key answer.",
                f"Context:\n{context_text}\n\nCau hoi: {q}",
            )
            return str(payload["answer"]), contexts
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}")

    return (contexts[0] if contexts else "Không tìm thấy thông tin."), contexts


def run_batch(items: list[dict], search, reranker, top_k: int) -> list[dict]:
    from config import OPENAI_API_KEY

    prepared = []
    for item in items:
        results = search.search(item["question"])
        documents = [{"text": result.text, "score": result.score, "metadata": result.metadata}
                     for result in results]
        reranked = reranker.rerank(item["question"], documents, top_k=top_k)
        contexts = [result.text for result in reranked] or [result.text for result in results[:3]]
        prepared.append({**item, "contexts": contexts})

    generated: dict[int, str] = {}
    if OPENAI_API_KEY:
        try:
            from src.openai_client import chat_json
            request_items = [
                {"id": item["id"], "question": item["question"],
                 "contexts": [context[:2500] for context in item["contexts"]]}
                for item in prepared
            ]
            payload = chat_json(
                "Ban la tro ly HR. Tra loi bang tieng Viet, chi dung context, uu tien policy hien hanh. "
                "Neu context thieu, noi ro khong tim thay. Tra ve JSON object voi key answers.",
                "Tra loi tung cau. answers la mang cac object {id, answer}, giu dung id:\n"
                + json.dumps(request_items, ensure_ascii=False),
                timeout=120,
            )
            generated = {int(row["id"]): str(row["answer"]) for row in payload.get("answers", [])}
        except Exception as error:
            print(f"  API batch failed; using extractive fallback: {error}")

    return [
        {
            "id": item["id"],
            "distribution": item["distribution"],
            "question": item["question"],
            "answer": generated.get(item["id"], item["contexts"][0] if item["contexts"] else "Không tìm thấy thông tin."),
            "contexts": item["contexts"],
            "ground_truth": item["ground_truth"],
        }
        for item in prepared
    ]


def main():
    print("=" * 60)
    print("LAB 24 SETUP — Generating answers for 50 questions")
    print("=" * 60)

    if not check_day18_files():
        sys.exit(1)

    with open("test_set_50q.json", encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"✓ Loaded {len(test_set)} questions (factual/multi_hop/adversarial)")

    try:
        search, reranker, top_k = build_pipeline()
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("→ Đảm bảo bạn đã copy src/ từ Day 18 và đã pip install -r requirements.txt")
        sys.exit(1)

    print(f"\nRunning {len(test_set)} queries...")
    answers = []
    t_start = time.time()
    batches = [test_set[index:index + 10] for index in range(0, len(test_set), 10)]
    with ThreadPoolExecutor(max_workers=min(5, len(batches))) as executor:
        futures = [executor.submit(run_batch, batch, search, reranker, top_k) for batch in batches]
        for future in as_completed(futures):
            answers.extend(future.result())
            print(f"  [{len(answers)}/{len(test_set)}] done ({time.time()-t_start:.0f}s elapsed)")
    answers.sort(key=lambda item: item["id"])

    with open("answers_50q.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(answers)} answers → answers_50q.json")
    print(f"  Total time: {time.time()-t_start:.1f}s")
    print("\n→ Bây giờ bắt đầu Phase A:")
    print("     python src/phase_a_ragas.py")


if __name__ == "__main__":
    main()
