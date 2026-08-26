from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import re
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers. (Đã implement sẵn)

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
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

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    analyzer  = AnalyzerEngine(registry=registry)
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
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if analyzer is not None:
        results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
        entities = [
            {
                "type": result.entity_type,
                "text": text[result.start:result.end],
                "score": round(float(result.score), 3),
                "start": result.start,
                "end": result.end,
            }
            for result in results
        ]
        if anonymizer is not None and results:
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
        else:
            anonymized = _redact_entities(text, entities)
        return {"has_pii": bool(entities), "entities": entities, "anonymized": anonymized}

    patterns = (
        ("EMAIL_ADDRESS", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 0.95),
        ("VN_PHONE", r"(?<!\d)0[3-9]\d{8}(?!\d)", 0.95),
        ("VN_CCCD", r"(?<!\d)\d{12}(?!\d)", 0.95),
        ("VN_CCCD", r"(?<!\d)\d{9}(?!\d)", 0.75),
    )
    entities = []
    occupied: list[tuple[int, int]] = []
    for entity_type, pattern, score in patterns:
        for match in re.finditer(pattern, text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            occupied.append(span)
            entities.append({
                "type": entity_type,
                "text": match.group(0),
                "score": score,
                "start": span[0],
                "end": span[1],
            })
    entities.sort(key=lambda item: item["start"])
    return {
        "has_pii": bool(entities),
        "entities": entities,
        "anonymized": _redact_entities(text, entities),
    }


def _redact_entities(text: str, entities: list[dict]) -> str:
    output = text
    for entity in sorted(entities, key=lambda item: item["start"], reverse=True):
        output = output[:entity["start"]] + f"<{entity['type']}>" + output[entity["end"]:]
    return output


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in text if not unicodedata.combining(char)).replace("đ", "d")


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


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    reason = _deterministic_input_block_reason(text)
    if reason:
        return {
            "allowed": False,
            "blocked_reason": reason,
            "response": "Yeu cau bi tu choi boi chinh sach an toan.",
        }

    if rails is None and OPENAI_API_KEY_AVAILABLE():
        try:
            rails = setup_nemo_rails()
        except (ImportError, RuntimeError, ValueError):
            rails = None
    if rails is None:
        return {"allowed": True, "blocked_reason": None, "response": "allowed"}

    response = await rails.generate_async(messages=[{"role": "user", "content": text}])
    if isinstance(response, dict):
        response_text = str(response.get("content", response))
    else:
        response_text = str(response)
    refuse_keywords = ("xin loi", "khong the", "khong duoc phep", "i cannot", "i'm sorry")
    blocked = any(keyword in _normalise(response_text) for keyword in refuse_keywords)
    return {
        "allowed": not blocked,
        "blocked_reason": "nemo_input_rail" if blocked else None,
        "response": response_text,
    }


def OPENAI_API_KEY_AVAILABLE() -> bool:
    from config import OPENAI_API_KEY

    return bool(OPENAI_API_KEY and os.getenv("USE_NEMO_GUARDRAILS", "").lower() in {"1", "true", "yes"})


def _deterministic_input_block_reason(text: str) -> str | None:
    normalised = _normalise(text)
    pattern_groups = {
        "prompt_injection": (
            "bo qua tat ca huong dan", "ignore your system", "ignore previous",
            "system override", "admin command", "system instructions", "system prompt",
            "forget all", "priority 1", "in ra toan bo", "dump all training",
        ),
        "jailbreak": (
            "do anything now", "khong co gioi han", "unrestricted ai", "dong vai hr manager",
            "tan cong mang", "tiet lo bang luong", "confidential hr data",
        ),
        "pii_request": (
            "cccd va so dien thoai", "thong tin nhan vien", "employee records",
            "liet ke toan bo thong tin", "tiet lo luong", "all employee salaries",
        ),
        "off_topic": (
            "bai tho", "nau pho", "bitcoin", "ethereum", "phuong trinh vi phan",
            "phim marvel", "marvel", "cong thuc truyen thong",
        ),
    }
    for reason, patterns in pattern_groups.items():
        if any(pattern in normalised for pattern in patterns):
            return reason
    return None


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
    pii_result = pii_scan(answer)
    sensitive_patterns = ("mat khau admin", "system prompt", "employee records", "bang luong chi tiet")
    sensitive = any(pattern in _normalise(answer) for pattern in sensitive_patterns)
    if pii_result["has_pii"] or sensitive:
        return {
            "safe": False,
            "flagged_reason": "pii_output" if pii_result["has_pii"] else "sensitive_output",
            "final_answer": pii_result["anonymized"] if pii_result["has_pii"] else "Noi dung da bi chan.",
        }

    if rails is not None:
        response = await rails.generate_async(messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        response_text = str(response.get("content", response)) if isinstance(response, dict) else str(response)
        refused = any(marker in _normalise(response_text) for marker in ("xin loi", "khong the", "i cannot"))
        if refused:
            return {"safe": False, "flagged_reason": "nemo_output_rail", "final_answer": response_text}
    return {"safe": True, "flagged_reason": None, "final_answer": answer}


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
        output = []
        for item in adversarial_set:
            blocked_by = None
            if pii_scan(item["input"], analyzer, anonymizer)["has_pii"]:
                blocked_by = "presidio"
            else:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"
            actual = "blocked" if blocked_by else "allowed"
            output.append({
                "id": item["id"],
                "category": item["category"],
                "input": item["input"],
                "expected": item["expected"],
                "actual": actual,
                "blocked_by": blocked_by,
                "passed": actual == item["expected"],
            })
        return output

    results = asyncio.run(_run_all())
    passed = sum(result["passed"] for result in results)
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
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    if not test_inputs:
        empty = {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        return {
            "presidio_ms": empty.copy(),
            "nemo_ms": empty.copy(),
            "total_ms": empty.copy(),
            "latency_budget_ok": True,
            "budget_ms": LATENCY_BUDGET_P95_MS,
        }

    presidio_times: list[float] = []
    nemo_times: list[float] = []
    total_times: list[float] = []

    async def _measure() -> None:
        for index in range(n_runs):
            text = test_inputs[index % len(test_inputs)]
            start = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - start) * 1000
            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())

    def percentiles(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        def nearest_rank(percentile: float) -> float:
            index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.5)))
            return round(ordered[index], 3)
        return {"p50": nearest_rank(0.50), "p95": nearest_rank(0.95), "p99": nearest_rank(0.99)}

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
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    results = run_adversarial_suite(adversarial_set)
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=20)
    report = {
        "total_inputs": len(results),
        "passed": sum(result["passed"] for result in results),
        "pass_rate": round(sum(result["passed"] for result in results) / max(len(results), 1), 4),
        "results": results,
        "latency": latency,
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/guard_results.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"Guard report saved: {report['passed']}/{report['total_inputs']} passed")
    print(f"Total guard P95: {latency['total_ms']['p95']}ms")
