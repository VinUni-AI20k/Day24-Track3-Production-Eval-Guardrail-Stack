from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _patch_ragas_executor() -> None:
    """[KHÁC BIỆT] Compat shim cho ragas 0.1.x trên Python 3.12+.

    `Executor.results()` gốc tạo `asyncio.as_completed(...)` KHI CHƯA có loop
    đang chạy, rồi mới gọi `asyncio.run()` — nên coroutine bind vào một loop
    khác với loop thực thi:
      - Python 3.11: get_event_loop() tự tạo loop ngầm → chỉ warning.
      - Python 3.12+: không còn fallback → RuntimeError "no current event loop";
        nếu tự set_event_loop() thì coroutine bind vào loop không bao giờ chạy
        → treo vô hạn.
    Bản thay thế dựng toàn bộ coroutine BÊN TRONG asyncio.run() → 1 loop duy nhất.
    """
    import asyncio
    from ragas import executor as _ex

    if getattr(_ex.Executor, "_py312_patched", False):
        return

    from tqdm.auto import tqdm

    def results(self):
        max_workers = (self.run_config or _ex.RunConfig()).max_workers

        async def _aresults():
            semaphore = asyncio.Semaphore(max_workers) if max_workers != -1 else None

            async def _run(coro):
                if semaphore is None:
                    return await coro
                async with semaphore:
                    return await coro

            coros = [_run(afunc(*a, **kw)) for afunc, a, kw, _ in self.jobs]
            out = []
            for future in tqdm(asyncio.as_completed(coros), desc=self.desc,
                               total=len(coros), leave=self.keep_progress_bar):
                out.append(await future)
            return out

        results = asyncio.run(_aresults())
        return [r[1] for r in sorted(results, key=lambda x: x[0])]

    _ex.Executor.results = results
    _ex.Executor._py312_patched = True


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    zeros = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    from config import (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, EMBEDDING_MODEL,
                        MODEL_DEVICE, LLM_MAX_RETRIES, LLM_TIMEOUT, LLM_PROVIDER)

    if not LLM_API_KEY:
        print("  ⚠️  RAGAS skipped: chưa có API key (GROQ_API_KEY/OPENAI_API_KEY) "
              "→ trả 0.0 cho cả 4 metric.")
        return zeros

    # RAGAS cần LLM judge + embeddings. Groq KHÔNG có embedding API → dùng
    # embedding local (bge-m3) để answer_relevancy/context_precision chạy được.
    try:
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings

        judge_llm = ChatOpenAI(model=LLM_MODEL, api_key=LLM_API_KEY,
                               base_url=LLM_BASE_URL, temperature=0,
                               max_retries=LLM_MAX_RETRIES, timeout=LLM_TIMEOUT)
        judge_emb = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs={"device": MODEL_DEVICE})

        _patch_ragas_executor()

        # [GIẢ ĐỊNH] answer_relevancy mặc định strictness=3 → gửi n=3 trong 1 request.
        # Groq trả 400 "'n' : number must be at most 1" → hạ xuống 1 (sinh 1 câu hỏi
        # thay vì 3 rồi lấy trung bình cosine). Đánh đổi: điểm nhiễu hơn một chút.
        answer_relevancy.strictness = 1

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        # Giới hạn timeout/retry để 1 câu lỗi không treo cả pipeline
        # Groq free tier chỉ 8000 TPM → phải hạ worker; OpenAI rộng hơn nhiều nên
        # song song cao hơn. Chỉnh bằng RAGAS_WORKERS nếu bị 429.
        _workers = int(os.getenv("RAGAS_WORKERS", "2" if LLM_PROVIDER == "groq" else "8"))
        run_config = RunConfig(timeout=180, max_retries=10, max_wait=60,
                               max_workers=_workers)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_llm, embeddings=judge_emb,
            run_config=run_config, raise_exceptions=False,
        )
        df = result.to_pandas()

        def _num(row, key):
            """None nếu metric không tính được (NaN do lỗi API) — KHÔNG phải 0.0."""
            try:
                v = float(row.get(key, float("nan")))
            except (TypeError, ValueError):
                return None
            return None if v != v else v  # NaN → None

        metric_names = ["faithfulness", "answer_relevancy",
                        "context_precision", "context_recall"]
        raw = [{m: _num(row, m) for m in metric_names} | {"row": row}
               for _, row in df.iterrows()]

        n_failed = sum(1 for r in raw for m in metric_names if r[m] is None)
        if n_failed:
            print(f"  ⚠️  {n_failed}/{len(raw) * len(metric_names)} metric-cell lỗi "
                  f"(rate limit/parse) — loại khỏi trung bình, không tính là 0.")

        per_question = [
            EvalResult(
                question=r["row"]["question"],
                answer=r["row"]["answer"],
                contexts=list(r["row"]["contexts"]),
                ground_truth=r["row"]["ground_truth"],
                faithfulness=r["faithfulness"] if r["faithfulness"] is not None else 0.0,
                answer_relevancy=r["answer_relevancy"] if r["answer_relevancy"] is not None else 0.0,
                context_precision=r["context_precision"] if r["context_precision"] is not None else 0.0,
                context_recall=r["context_recall"] if r["context_recall"] is not None else 0.0,
            )
            for r in raw
        ]

        def _avg(attr):
            vals = [r[attr] for r in raw if r[attr] is not None]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "faithfulness": _avg("faithfulness"),
            "answer_relevancy": _avg("answer_relevancy"),
            "context_precision": _avg("context_precision"),
            "context_recall": _avg("context_recall"),
            "per_question": per_question,
        }
    except Exception as e:
        # Thiếu dependency / lỗi API / rate limit → KHÔNG được làm vỡ pipeline
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating — answer chứa thông tin không có trong context",
                         "Tighten prompt (chỉ dựa context), lower temperature, thêm citation"),
        "context_recall": ("Missing relevant chunks — context thiếu bằng chứng cần thiết",
                           "Improve chunking (parent-child / structure) hoặc tăng trọng số BM25"),
        "context_precision": ("Too many irrelevant chunks — nhiễu đẩy bằng chứng xuống dưới",
                              "Add reranking hoặc metadata filter (version, source)"),
        "answer_relevancy": ("Answer không trả lời đúng câu hỏi",
                             "Improve prompt template, yêu cầu trả lời trực tiếp câu hỏi"),
    }
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    scored = []
    for r in eval_results:
        values = {m: getattr(r, m) for m in metric_names}
        avg = sum(values.values()) / len(metric_names)
        worst_metric = min(values, key=lambda m: values[m])
        diagnosis, fix = diagnostic_tree[worst_metric]
        scored.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "worst_metric": worst_metric,
            "score": round(values[worst_metric], 4),
            "avg_score": round(avg, 4),
            "metrics": {m: round(v, 4) for m, v in values.items()},
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    scored.sort(key=lambda d: d["avg_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
