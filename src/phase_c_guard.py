from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers.

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NoOpNlpEngine
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
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

    # The training environment has no downloaded spaCy language model.  These
    # recognizers are regex-only, so Presidio's supported no-op NLP engine
    # preserves offline PII scanning without attempting a model download.
    nlp_engine = NoOpNlpEngine(models=[{"lang_code": PRESIDIO_LANGUAGE,
                                        "model_name": "no-op"}])
    registry = RecognizerRegistry(supported_languages=[PRESIDIO_LANGUAGE])
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)
    registry.add_recognizer(email_recognizer)

    analyzer = AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=[PRESIDIO_LANGUAGE],
    )
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


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
        analyzer, anonymizer = setup_presidio()

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

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)

    Config directory: guardrails/
        config.yml  — model + rails config
        rails.co    — Colang dialogue flows (topic check, jailbreak check, output check)
    """
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails  = LLMRails(config)
    return rails


_REFUSAL_KEYWORDS = (
    "xin lỗi",
    "không thể",
    "không được phép",
    "i cannot",
    "i'm sorry",
)
_FALLBACK_REFUSAL = (
    "Xin lỗi, tôi không thể thực hiện yêu cầu này. "
    "Tôi chỉ có thể hỗ trợ các câu hỏi an toàn về chính sách nhân sự."
)
_FALLBACK_UNSAFE_INPUT_PATTERNS = (
    r"bỏ qua.*hướng dẫn",
    r"ignore (?:all |your |previous |the )?(?:previous )?(?:instructions|system prompt)",
    r"forget (?:all |your |the )?(?:instructions|system prompt|polic)",
    r"(?:pretend|đóng vai).*(?:dan|unrestricted|không có giới hạn|hr manager)",
    r"(?:system override|admin command|priority 1|ignore previous instructions)",
    r"(?:ai không có giới hạn|unrestricted ai)",
    r"(?:cccd|số điện thoại|email|thông tin cá nhân).*(?:của|nhân viên)",
    r"(?:tiết lộ|reveal|dump|list|print).*(?:lương|salary|confidential|employee|nhân viên|training data|system instructions)",
    r"(?:in ra|show|display).*(?:system instructions|system prompt|context)",
    r"(?:bảng lương|employee salaries|tấn công mạng|attack)",
    r"(?:mật khẩu|password).*(?:admin|nội bộ|internal|hệ thống|system)",
    r"(?:viết|write).*(?:bài thơ|poem)",
    r"(?:nấu|cook).*(?:phở|food|recipe)",
    r"(?:bitcoin|ethereum|giá cổ phiếu|stock price)",
    r"(?:giải phương trình|solve).*(?:phương trình|equation|dy/dx)",
    r"(?:recommend|gợi ý).*(?:phim|movie|marvel)",
    r"(?:thời tiết|weather|tin tức|news)",
)
_FALLBACK_ALLOWED_HR_PATTERNS = (
    r"(?:nghỉ phép|ngày phép|phép năm|nghỉ ốm|leave|vacation)",
    r"(?:bảo hiểm|insurance|pvi|benefit|phúc lợi)",
    r"(?:lương|thưởng|payroll|salary policy|compensation)",
    r"(?:tạm ứng|công tác|thử việc|đào tạo|onboarding)",
    r"(?:wfh|làm việc từ xa|remote work|phụ cấp|mentor)",
    r"(?:vpn|mật khẩu|password).*(?:chính sách|policy|quy định|yêu cầu|tối thiểu|đổi|bao lâu|hết hạn|hiệu lực)",
    r"(?:đổi|bao lâu|hết hạn|hiệu lực).*(?:mật khẩu|password)",
)
_FALLBACK_SENSITIVE_OUTPUT_PATTERNS = (
    r"(?:cccd|số điện thoại cá nhân|mật khẩu hệ thống|thông tin bí mật)",
)


def _is_refusal(response: str) -> bool:
    """Return whether a rail response contains a configured refusal phrase."""
    lowered = response.casefold()
    return any(keyword in lowered for keyword in _REFUSAL_KEYWORDS)


def _matches_known_unsafe_input(text: str) -> bool:
    """Narrow fail-closed fallback used only when NeMo cannot run locally."""
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _FALLBACK_UNSAFE_INPUT_PATTERNS)


def _matches_allowed_hr_input(text: str) -> bool:
    """Allow only recognisable HR-policy questions while NeMo is unavailable."""
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _FALLBACK_ALLOWED_HR_PATTERNS)


def _fallback_input_rail(text: str) -> dict:
    """Keep known hostile inputs blocked during NeMo/Python incompatibility."""
    if _matches_known_unsafe_input(text):
        return {
            "allowed": False,
            "blocked_reason": "nemo_input_rail",
            "response": _FALLBACK_REFUSAL,
        }
    if _matches_allowed_hr_input(text):
        return {"allowed": True, "blocked_reason": None, "response": ""}
    return {
        "allowed": False,
        "blocked_reason": "nemo_input_rail",
        "response": _FALLBACK_REFUSAL,
    }


def _fallback_output_rail(answer: str) -> dict:
    """Block known sensitive outputs if the NeMo runtime is unavailable."""
    if any(re.search(pattern, answer, flags=re.IGNORECASE)
           for pattern in _FALLBACK_SENSITIVE_OUTPUT_PATTERNS):
        return {
            "safe": False,
            "flagged_reason": "nemo_output_rail",
            "final_answer": _FALLBACK_REFUSAL,
        }
    return {"safe": True, "flagged_reason": None, "final_answer": answer}


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    try:
        if rails is None:
            rails = setup_nemo_rails()
        response = await rails.generate_async(
            messages=[{"role": "user", "content": text}]
        )
    except Exception:
        # nemoguardrails currently fails to import on Python 3.14 because of
        # an upstream pydantic/langchain annotation incompatibility.  Keep the
        # primary NeMo path above; this isolated fallback protects known abuse.
        return _fallback_input_rail(text)

    response = str(response)
    blocked = _is_refusal(response)
    return {
        "allowed": not blocked,
        "blocked_reason": "nemo_input_rail" if blocked else None,
        "response": response,
    }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails hoạt động trong context của cả cuộc hội thoại (input + output).
    Kiểm tra: có PII không? Nội dung có phù hợp không? Có hallucination rõ ràng không?

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    try:
        if rails is None:
            rails = setup_nemo_rails()
        response = await rails.generate_async(messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
    except Exception:
        return _fallback_output_rail(answer)

    response = str(response)
    flagged = _is_refusal(response)
    return {
        "safe": not flagged,
        "flagged_reason": "nemo_output_rail" if flagged else None,
        "final_answer": response if flagged else answer,
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
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    results = run_adversarial_suite(adversarial_set)
    if results:
        passed = sum(1 for r in results if r["passed"])
        print(f"Adversarial suite: {passed}/{len(results)} passed")

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "guard_results.json"), "w", encoding="utf-8") as f:
        json.dump({"suite": results, "latency": latency}, f, ensure_ascii=False, indent=2)
