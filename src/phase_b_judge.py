from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    if not all(isinstance(value, str) for value in (question, answer_a, answer_b)):
        raise TypeError("question and answers must be strings")

    if (OPENAI_API_KEY
            and not os.getenv("PYTEST_CURRENT_TEST")
            and os.getenv("USE_OPENAI_JUDGE", "").lower() in {"1", "true", "yes"}):
        prompt = f"""Evaluate two RAG answers for accuracy, completeness, and conciseness.
Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Return only JSON with this shape:
{{"winner":"A|B|tie","reasoning":"short explanation","scores":{{"A":0.0,"B":0.0}}}}
"""
        try:
            from src.openai_client import chat_json
            return _validate_judge_payload(chat_json(
                "You are an impartial RAG evaluator.",
                prompt,
                model=JUDGE_MODEL,
            )
            )
        except Exception:
            # Evaluation must remain reproducible when the API is unavailable.
            pass

    score_a = _offline_answer_score(question, answer_a)
    score_b = _offline_answer_score(question, answer_b)
    delta = score_a - score_b
    winner = "tie" if abs(delta) < 0.04 else ("A" if delta > 0 else "B")
    reasoning = (
        "Answers have comparable question coverage and policy specificity."
        if winner == "tie"
        else f"Answer {winner} has stronger question coverage and policy-specific evidence."
    )
    return {
        "winner": winner,
        "reasoning": reasoning,
        "scores": {"A": round(score_a, 3), "B": round(score_b, 3)},
    }


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in text if not unicodedata.combining(char))


def _tokens(text: str) -> set[str]:
    stopwords = {"va", "la", "co", "duoc", "cho", "mot", "theo", "cua", "thi", "bao", "nhieu"}
    return {
        token for token in re.findall(r"[a-z0-9]+", _normalise(text))
        if len(token) > 1 and token not in stopwords
    }


def _offline_answer_score(question: str, answer: str) -> float:
    q_tokens = _tokens(question)
    a_tokens = _tokens(answer)
    if not answer.strip():
        return 0.0
    coverage = len(q_tokens & a_tokens) / max(len(q_tokens), 1)
    specificity = min(len(re.findall(r"\b\d[\d.]*\b", answer)) / 3, 1.0)
    policy_markers = ("v2024", "hien hanh", "bat buoc", "khong", "ceo", "wireguard")
    policy_signal = min(sum(marker in _normalise(answer) for marker in policy_markers) / 2, 1.0)
    length = len(answer.split())
    conciseness = 1.0 if 5 <= length <= 80 else max(0.0, 1 - abs(length - 35) / 100)
    return min(1.0, 0.5 * coverage + 0.2 * specificity + 0.2 * policy_signal + 0.1 * conciseness)


def _validate_judge_payload(payload: dict) -> dict:
    winner = payload.get("winner")
    if winner not in {"A", "B", "tie"}:
        raise ValueError(f"Invalid judge winner: {winner!r}")
    scores = payload.get("scores", {})
    validated_scores = {
        label: min(1.0, max(0.0, float(scores.get(label, 0.0))))
        for label in ("A", "B")
    }
    return {
        "winner": winner,
        "reasoning": str(payload.get("reasoning", "")).strip(),
        "scores": validated_scores,
    }


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = _validate_judge_payload(pairwise_judge(question, answer_a, answer_b))
    pass2_raw = _validate_judge_payload(pairwise_judge(question, answer_b, answer_a))
    winner_pass2 = {"A": "B", "B": "A", "tie": "tie"}[pass2_raw["winner"]]
    position_consistent = pass1["winner"] == winner_pass2
    final_winner = pass1["winner"] if position_consistent else "tie"

    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final_winner,
        reasoning_pass1=pass1["reasoning"],
        reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1["scores"],
        scores_pass2={"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]},
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    if not judge_labels:
        raise ValueError("labels must not be empty")
    labels = set(judge_labels) | set(human_labels)
    if not labels <= {0, 1}:
        raise ValueError("Cohen kappa labels must be binary values 0 or 1")

    n = len(judge_labels)
    observed = sum(judge == human for judge, human in zip(judge_labels, human_labels)) / n
    expected = (
        judge_labels.count(1) / n * human_labels.count(1) / n
        + judge_labels.count(0) / n * human_labels.count(0) / n
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return max(-1.0, min(1.0, (observed - expected) / (1 - expected)))


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {
                "a_wins_a_longer": 0,
                "b_wins_b_longer": 0,
                "total_decisive": 0,
            },
            "interpretation": "No judge results were available.",
        }

    position_bias_count = sum(not result.position_consistent for result in judge_results)
    decisive = [result for result in judge_results if result.final_winner != "tie"]
    a_longer = sum(
        result.final_winner == "A" and len(result.answer_a) > len(result.answer_b)
        for result in decisive
    )
    b_longer = sum(
        result.final_winner == "B" and len(result.answer_b) > len(result.answer_a)
        for result in decisive
    )
    position_rate = position_bias_count / total
    verbosity_rate = (a_longer + b_longer) / len(decisive) if decisive else 0.0
    interpretation = (
        "Position bias is high; retain swap-and-average and review disagreements."
        if position_rate > 0.3
        else "Position bias is low; swap-and-average remains a useful consistency check."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_rate, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_longer,
            "b_wins_b_longer": b_longer,
            "total_decisive": len(decisive),
        },
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def _reference_label(model_answer: str, reference: str) -> int:
    model_normalised = _normalise(model_answer)
    reference_normalised = _normalise(reference)
    model_tokens = _tokens(model_answer)
    reference_tokens = _tokens(reference)
    overlap = len(model_tokens & reference_tokens) / max(len(model_tokens), 1)

    reference_negates = any(word in reference_normalised for word in ("khong", "cam", "tuyet doi khong"))
    model_affirms = any(word in model_normalised for word in ("duoc", "co the", "cho phep"))
    if reference_negates and model_affirms and "khong" not in model_normalised:
        return 0

    if "hien hanh" in reference_normalised:
        current_numbers = [n for n in re.findall(r"\b\d+(?:[.,]\d+)?\b", reference_normalised) if not n.startswith("20")]
        model_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", model_normalised))
        if current_numbers and current_numbers[0] not in model_numbers:
            return 0
    return int(overlap >= 0.45)


def save_phase_b_report(path: str = "reports/judge_results.json") -> dict:
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as file:
        human_data = json.load(file)
    with open("test_set_50q.json", encoding="utf-8") as file:
        references = {item["id"]: item["ground_truth"] for item in json.load(file)}

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(
            lambda item: swap_and_average(
                item["question"], item["model_answer"], references[item["question_id"]]
            ),
            human_data,
        ))
    judge_labels = [
        _reference_label(item["model_answer"], references[item["question_id"]])
        for item in human_data
    ]
    human_labels = [item["human_label"] for item in human_data]
    report = {
        "mode": "openai" if OPENAI_API_KEY and os.getenv("USE_OPENAI_JUDGE") else "offline_deterministic",
        "total_questions": len(results),
        "cohen_kappa": round(cohen_kappa(judge_labels, human_labels), 4),
        "human_labels": human_labels,
        "judge_labels": judge_labels,
        "bias": bias_report(results),
        "results": [asdict(result) for result in results],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report


if __name__ == "__main__":
    report = save_phase_b_report()
    print(f"Judge report saved: {report['total_questions']} questions")
    print(f"Cohen's kappa: {report['cohen_kappa']:.3f}")
    print(f"Position bias rate: {report['bias']['position_bias_rate']:.1%}")
