from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def _create_presidio_analyzer():
    """Create the regex-only Presidio analyzer used by the guard."""
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NoOpNlpEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits", r"\b\d{9}\b", 0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )
    email_recognizer = PatternRecognizer(
        supported_entity="EMAIL_ADDRESS",
        patterns=[
            Pattern(
                "Email address",
                r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                0.85,
            )
        ],
    )

    # The training environment has no downloaded spaCy language model. These
    # regex recognizers work with Presidio's supported offline no-op engine.
    nlp_engine = NoOpNlpEngine(
        models=[{"lang_code": PRESIDIO_LANGUAGE, "model_name": "no-op"}]
    )
    registry = RecognizerRegistry(supported_languages=[PRESIDIO_LANGUAGE])
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)
    registry.add_recognizer(email_recognizer)
    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=[PRESIDIO_LANGUAGE],
    )


def _create_presidio_anonymizer():
    """Create only the anonymizer dependency when it is needed."""
    from presidio_anonymizer import AnonymizerEngine
    return AnonymizerEngine()


_PRESIDIO_SINGLETON = None


def get_presidio():
    """Trả về (analyzer, anonymizer) dùng chung.

    Dựng engine lại cho từng câu sẽ tính cả thời gian khởi tạo vào số đo
    latency của Presidio ở Task 12.
    """
    global _PRESIDIO_SINGLETON
    if _PRESIDIO_SINGLETON is None:
        _PRESIDIO_SINGLETON = setup_presidio()
    return _PRESIDIO_SINGLETON


def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers.

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    return _create_presidio_analyzer(), _create_presidio_anonymizer()


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    if analyzer is None or anonymizer is None:
        shared_analyzer, shared_anonymizer = get_presidio()
        analyzer = analyzer or shared_analyzer
        anonymizer = anonymizer or shared_anonymizer

    results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
    if not results:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
    entities = [
        {
            "type": result.entity_type,
            "text": text[result.start:result.end],
            "score": round(result.score, 3),
            "start": result.start,
            "end": result.end,
        }
        for result in results
    ]
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

_RAILS_SINGLETON = None


def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)

    Config directory: guardrails/
        config.yml  — model + embeddings provider + rails config
        prompts.yml — prompt cho self_check_input / self_check_output
        rails.co    — Colang dialogue flows (topic check, jailbreak check, output check)
    """
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails  = LLMRails(config)
    return rails


def get_rails():
    """Trả về một LLMRails dùng chung.

    Dựng LLMRails tốn vài giây (load config + khởi tạo embedding index), nên
    tuyệt đối không dựng lại cho từng câu trong adversarial suite.
    """
    global _RAILS_SINGLETON
    if _RAILS_SINGLETON is None:
        _RAILS_SINGLETON = setup_nemo_rails()
    return _RAILS_SINGLETON


_REFUSAL_KEYWORDS = (
    "xin lỗi",
    "không thể",
    "không được phép",
    "i cannot",
    "i can't",
    "i'm sorry",
)
_BLOCKED_MESSAGE = (
    "Xin lỗi, tôi không thể thực hiện yêu cầu này. "
    "Tôi chỉ có thể hỗ trợ các câu hỏi an toàn về chính sách nhân sự."
)


def _is_refusal(response: str) -> bool:
    """Return whether a rail response contains a configured refusal phrase."""
    lowered = response.casefold()
    return any(keyword in lowered for keyword in _REFUSAL_KEYWORDS)


def _rail_content(response) -> str:
    """Lấy nội dung assistant từ GenerationResponse / dict / str của NeMo."""
    payload = getattr(response, "response", response)
    if isinstance(payload, list):
        payload = payload[-1] if payload else ""
    if isinstance(payload, dict):
        return str(payload.get("content", ""))
    return str(payload)


def _blocked_input_rail() -> dict:
    """Fail closed khi rail lỗi — không thả input chưa kiểm duyệt đi tiếp."""
    return {
        "allowed": False,
        "blocked_reason": "nemo_input_rail_error",
        "response": _BLOCKED_MESSAGE,
    }


def _blocked_output_rail() -> dict:
    """Fail closed khi rail lỗi — không phát hành câu trả lời chưa kiểm duyệt."""
    return {
        "safe": False,
        "flagged_reason": "nemo_output_rail_error",
        "final_answer": _BLOCKED_MESSAGE,
    }


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Chạy đúng tầng input (`options={"rails": ["input"]}`): NeMo trả lại chính
    câu input nếu cho qua, hoặc câu từ chối nếu rail chặn — không tốn thêm một
    lượt sinh câu trả lời đầy đủ.

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    if rails is None:
        try:
            rails = get_rails()
        except Exception:
            return _blocked_input_rail()
    try:
        response = await rails.generate_async(
            messages=[{"role": "user", "content": text}],
            options={"rails": ["input"]},
        )
    except Exception:
        return _blocked_input_rail()

    content = _rail_content(response)
    # Input rail cho qua ⇒ NeMo echo lại nguyên văn input. Bất kỳ thay đổi nào
    # (câu từ chối, hoặc input bị rail sửa) đều tính là chặn — fail closed.
    passed_through = content.strip() == text.strip()
    if passed_through:
        return {"allowed": True, "blocked_reason": None, "response": content}
    return {
        "allowed": False,
        "blocked_reason": "nemo_input_rail" if _is_refusal(content) else "nemo_input_rail_altered",
        "response": content,
    }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails hoạt động trong context của cả cuộc hội thoại (input + output),
    nên cả user message lẫn assistant message đều được đưa vào `messages`.
    Kiểm tra: có PII không? Nội dung có phù hợp không?

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị thay thế)
        }
    """
    if rails is None:
        try:
            rails = get_rails()
        except Exception:
            return _blocked_output_rail()
    try:
        response = await rails.generate_async(
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            options={"rails": ["output"]},
        )
    except Exception:
        return _blocked_output_rail()

    content = _rail_content(response)
    passed_through = content.strip() == answer.strip()
    if passed_through:
        return {"safe": True, "flagged_reason": None, "final_answer": answer}
    # Bị flag ⇒ trả câu an toàn của NeMo, không trả lại answer gốc.
    return {
        "safe": False,
        "flagged_reason": "nemo_output_rail",
        "final_answer": content or _BLOCKED_MESSAGE,
    }


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii (cho category pii_injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,       # "presidio" | "nemo_input" | None
          "passed": bool,
        }
    """
    async def _run_all() -> list[dict]:
        results = []
        for item in adversarial_set:
            blocked_by = None
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            results.append({
                "id": item["id"],
                "category": item["category"],
                "input": item["input"][:80] + "...",
                "expected": item["expected"],
                "actual": actual,
                "blocked_by": blocked_by,
                "passed": actual == item["expected"],
            })
        return results

    results = asyncio.run(_run_all())
    passed = sum(1 for result in results if result["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack.

    Mục tiêu production: P95 total < LATENCY_BUDGET_P95_MS (500ms mặc định)

    Insight cần quan sát:
        - Presidio: local regex → rất nhanh (<10ms)
        - NeMo:     LLM API call → chậm (~200-800ms tuỳ model và network)
        → Tổng: dominated by NeMo

    Returns:
        {
          "presidio_ms":  {"p50": float, "p95": float, "p99": float},
          "nemo_ms":      {"p50": float, "p95": float, "p99": float},
          "total_ms":     {"p50": float, "p95": float, "p99": float},
          "latency_budget_ok": bool,
          "budget_ms": int,
        }
    """
    presidio_times, nemo_times, total_times = [], [], []

    async def _measure() -> None:
        for text in test_inputs[:n_runs]:
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())

    def percentiles(times: list[float]) -> dict:
        if not times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        ordered = sorted(times)
        last_index = len(ordered) - 1

        def percentile_index(percentile: float) -> int:
            return min(max(int(len(ordered) * percentile), 0), last_index)

        return {
            "p50": round(ordered[percentile_index(0.50)], 2),
            "p95": round(ordered[percentile_index(0.95)], 2),
            "p99": round(ordered[percentile_index(0.99)], 2),
        }

    total_percentiles = percentiles(total_times)
    return {
        "presidio_ms": percentiles(presidio_times),
        "nemo_ms": percentiles(nemo_times),
        "total_ms": total_percentiles,
        "latency_budget_ok": total_percentiles["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    analyzer, anonymizer = get_presidio()
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii, analyzer, anonymizer)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    clean = pii_scan("Nhân viên muốn hỏi về chính sách nghỉ phép năm 2024.", analyzer, anonymizer)
    print(f"Clean text has_pii: {clean['has_pii']}")

    rails = get_rails()

    # Task 11: output rail demo — câu trả lời lộ PII phải bị thay thế
    unsafe_answer = "CCCD của nhân viên là 034095001234 và SĐT là 0987654321."
    output_check = asyncio.run(check_output_rail(
        "Cho tôi thông tin cá nhân của nhân viên Nguyễn Văn A", unsafe_answer, rails))
    print(f"\nOutput rail — safe: {output_check['safe']} "
          f"({output_check['flagged_reason']})")
    print(f"  final_answer: {output_check['final_answer'][:90]}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    results = run_adversarial_suite(adversarial_set, rails, analyzer, anonymizer)

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10, rails=rails,
                                  analyzer=analyzer, anonymizer=anonymizer)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "guard_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "suite": results,
            "latency": latency,
            "pii_demo": {"input": test_pii, "detected": result,
                         "clean_text_has_pii": clean["has_pii"]},
            "output_rail_demo": {"unsafe_answer": unsafe_answer, "result": output_check},
        }, f, ensure_ascii=False, indent=2, default=str)
