from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH, TEST_SET_PATH


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
    fallback = lambda diagnostic: {
        "winner": "tie",
        "reasoning": f"Judge unavailable: {diagnostic}",
        "scores": {"A": 0.0, "B": 0.0},
    }
    prompt = f'''Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
Trả lời JSON (chỉ JSON, không text khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}'''

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content)
        if not isinstance(payload, dict):
            raise ValueError("response is not a JSON object")
        winner = payload.get("winner")
        reasoning = payload.get("reasoning")
        scores = payload.get("scores")
        if winner not in {"A", "B", "tie"}:
            raise ValueError("winner is invalid")
        if not isinstance(reasoning, str) or not isinstance(scores, dict):
            raise ValueError("reasoning or scores has invalid type")
        if "A" not in scores or "B" not in scores:
            raise ValueError("scores must include A and B")
        normalized_scores = {
            key: min(1.0, max(0.0, float(scores[key])))
            for key in ("A", "B")
        }
        return {"winner": winner, "reasoning": reasoning, "scores": normalized_scores}
    except Exception as exc:
        return fallback(f"{type(exc).__name__}: {exc}")


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
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass1 = pass1["winner"]
    winner_pass2 = swap_map[pass2_raw["winner"]]
    position_consistent = winner_pass1 == winner_pass2
    final_winner = winner_pass1 if position_consistent else "tie"

    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=winner_pass1,
        winner_pass2=winner_pass2,
        final_winner=final_winner,
        reasoning_pass1=pass1["reasoning"],
        reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=position_consistent,
        scores_pass1=pass1["scores"],
        scores_pass2={"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]},
    )


# ─── Nhãn judge cho Cohen's κ ─────────────────────────────────────────────────

def grade_answer(question: str, answer: str, reference: str) -> dict:
    """Chấm nhị phân một câu trả lời so với ground truth.

    Đây là nhiệm vụ chấm giống hệt nhiệm vụ của người trong
    human_labels_10q.json (1 = tốt, 0 = xấu), nên nhãn trả về mới so sánh
    được với nhãn người bằng Cohen's κ. Pairwise "A hay B tốt hơn" là một
    thang khác — nó trộn độ chính xác với độ đầy đủ — nên chỉ dùng cho
    bias_report(), không dùng làm nhãn κ.

    Returns:
        {"label": 1|0, "reasoning": str}
    """
    prompt = f'''Bạn là expert kiểm định chất lượng câu trả lời RAG về chính sách nội bộ.

Câu hỏi: {question}

Đáp án đúng (ground truth):
{reference}

Câu trả lời của model:
{answer}

Chấm câu trả lời của model theo thang nhị phân:
- label = 1 (TỐT): đúng sự thật so với ground truth VÀ trả lời được ý chính của câu hỏi.
- label = 0 (XẤU): sai sự thật, mâu thuẫn với ground truth, hoặc bỏ sót ý chính.

Câu trả lời ngắn gọn nhưng đúng vẫn là TỐT. Chỉ chấm 0 khi thực sự sai hoặc thiếu ý chính.

Trả lời JSON (chỉ JSON, không text khác):
{{"label": 0 hoặc 1, "reasoning": "giải thích ngắn gọn"}}'''

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "Bạn là expert kiểm định RAG. Chỉ trả lời JSON."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.choices[0].message.content)
    label = payload.get("label")
    if label not in (0, 1):
        raise ValueError(f"grade_answer nhận label không hợp lệ: {label!r}")
    return {"label": int(label), "reasoning": str(payload.get("reasoning", ""))}


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
    if not judge_labels or len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must be equally sized and non-empty")
    if any(label not in {0, 1} for label in judge_labels + human_labels):
        raise ValueError("Cohen's kappa requires binary labels (0 or 1)")

    count = len(judge_labels)
    observed = sum(judge == human for judge, human in zip(judge_labels, human_labels)) / count
    expected = (
        judge_labels.count(1) / count * human_labels.count(1) / count
        + judge_labels.count(0) / count * human_labels.count(0) / count
    )
    if expected == 1.0:
        return 1.0 if judge_labels == human_labels else 0.0
    return (observed - expected) / (1.0 - expected)


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
    position_bias_count = sum(not result.position_consistent for result in judge_results)
    a_wins_a_longer = sum(
        result.final_winner == "A" and len(result.answer_a) > len(result.answer_b)
        for result in judge_results
    )
    b_wins_b_longer = sum(
        result.final_winner == "B" and len(result.answer_b) > len(result.answer_a)
        for result in judge_results
    )
    decisive = sum(result.final_winner != "tie" for result in judge_results)
    position_bias_rate = position_bias_count / total if total else 0.0
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive else 0.0
    interpretation = " ".join([
        "Position bias cao — nên dùng swap-and-average."
        if position_bias_rate > 0.3
        else "Position bias thấp — judge ổn định.",
        "Verbosity bias đáng lo ngại — judge gần như luôn chọn câu dài hơn."
        if verbosity_bias > 0.6
        else "Verbosity bias trong ngưỡng chấp nhận được.",
    ])
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        ground_truths = {item["id"]: item["ground_truth"] for item in json.load(f)}

    # A = câu trả lời của model, B = ground truth. Cặp thật (không phải chuỗi
    # rỗng) mới đo được position/verbosity bias có ý nghĩa.
    results = [
        swap_and_average(item["question"], item["model_answer"],
                         ground_truths[item["question_id"]])
        for item in human_data
    ]

    human_labels = [item["human_label"] for item in human_data]
    grades = [
        grade_answer(item["question"], item["model_answer"],
                     ground_truths[item["question_id"]])
        for item in human_data
    ]
    judge_labels = [g["label"] for g in grades]

    kappa = cohen_kappa(judge_labels, human_labels)
    bias = bias_report(results)
    report = {
        "human_labels": human_labels,
        "judge_labels": judge_labels,
        "judge_grades": [
            {"question_id": item["question_id"], "question": item["question"],
             "model_answer": item["model_answer"],
             "human_label": item["human_label"], "human_note": item["human_note"],
             "judge_label": grade["label"], "judge_reasoning": grade["reasoning"],
             "agree": item["human_label"] == grade["label"]}
            for item, grade in zip(human_data, grades)
        ],
        "results": [asdict(result) for result in results],
        "cohen_kappa": kappa,
        "bias": bias,
    }
    report_path = Path(__file__).resolve().parent.parent / "reports" / "judge_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Judge report written to {report_path}")
    print(f"  human labels: {human_labels}")
    print(f"  judge labels: {judge_labels}")
    print(f"  agreement:    {sum(j == h for j, h in zip(judge_labels, human_labels))}/10")
    print(f"  Cohen's κ:    {kappa:.4f}")
    print(f"  position_bias_rate: {bias['position_bias_rate']}  "
          f"verbosity_bias: {bias['verbosity_bias']}")
