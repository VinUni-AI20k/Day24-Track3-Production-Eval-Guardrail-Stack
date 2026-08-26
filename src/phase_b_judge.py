from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

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
    PROMPT_TEMPLATE = """Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác, đầy đủ, súc tích.
Trả lời JSON (chỉ JSON, không text khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
"""
    from config import OPENAI_API_KEY, JUDGE_MODEL
    if not OPENAI_API_KEY:
        return {"winner": "tie", "reasoning": "No API key configured", "scores": {"A": 0.5, "B": 0.5}}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(
                    question=question, answer_a=answer_a, answer_b=answer_b
                )},
            ],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        winner = parsed.get("winner", "tie")
        if winner not in {"A", "B", "tie"}:
            winner = "tie"
        reasoning = parsed.get("reasoning", "")
        scores = parsed.get("scores", {"A": 0.5, "B": 0.5})
        return {"winner": winner, "reasoning": reasoning, "scores": scores}
    except Exception as e:
        return {"winner": "tie", "reasoning": f"Judge error: {e}", "scores": {"A": 0.0, "B": 0.0}}


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
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw.get("winner", "tie"), "tie")

    # Average: consensus only if both agree
    if pass1.get("winner") == winner_pass2:
        final = pass1.get("winner", "tie")
    else:
        final = "tie"  # disagreement = inconclusive

    position_consistent = (pass1.get("winner") == winner_pass2)
    scores1 = pass1.get("scores", {"A": 0.0, "B": 0.0})
    scores2_raw = pass2_raw.get("scores", {"A": 0.0, "B": 0.0})
    scores2 = {"A": scores2_raw.get("B", 0.0), "B": scores2_raw.get("A", 0.0)}

    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1.get("winner", "tie"),
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=scores1,
        scores_pass2=scores2,
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
    """
    if not judge_labels or not human_labels or len(judge_labels) != len(human_labels):
        return 0.0
    try:
        from sklearn.metrics import cohen_kappa_score
        score = float(cohen_kappa_score(human_labels, judge_labels))
        return 0.0 if (score != score) else score
    except Exception:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (
            (judge_labels.count(1) / n) * (human_labels.count(1) / n) +
            (judge_labels.count(0) / n) * (human_labels.count(0) / n)
        )
        if abs(1.0 - p_e) < 1e-9:
            return 1.0 if p_o == 1.0 else 0.0
        return (p_o - p_e) / (1.0 - p_e)


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
            "interpretation": "Không có dữ liệu đánh giá.",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = (
        "Position bias cao — nên dùng swap-and-average để hạn chế bias."
        if position_bias_rate > 0.3
        else "Position bias thấp — judge hoạt động ổn định và nhất quán."
    )
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


def judge_single_answer(question: str, model_answer: str) -> dict:
    """Judge a single model answer as correct (1) or incorrect (0) based on accuracy & completeness."""
    from config import OPENAI_API_KEY, JUDGE_MODEL
    if not OPENAI_API_KEY:
        return {"label": 1, "reasoning": "No API key"}

    PROMPT = f"""Bạn là một chuyên gia thẩm định câu trả lời về chính sách HR nội bộ công ty (quy định v2024 hiện hành).

Lưu ý một số chính sách công ty:
- Phép năm theo v2024 hiện hành là 15 ngày (chính sách v2023 là 12 ngày đã hết hiệu lực, trả lời 12 ngày là SAI).
- Mua sắm thiết bị trên 50 triệu phải do Tổng Giám đốc (CEO) phê duyệt; Giám đốc phòng ban chỉ duyệt dưới 50 triệu.
- Tạm ứng trên 5 triệu cần cả Trưởng phòng và Kế toán trưởng phê duyệt; phí phạt tính pro-rata theo số ngày quá hạn.
- Nghiêm cấm dùng VPN cá nhân (như NordVPN) khi WFH, bắt buộc dùng WireGuard của công ty.
- Thưởng Tết tối thiểu 1 tháng lương cho nhân viên chính thức từ 6 tháng trở lên.
- Nghỉ kết hôn được 3 ngày làm việc hưởng lương.
- Đào tạo được tài trợ nghỉ việc trước 12 tháng phải hoàn trả 100% chi phí.
- Nhân viên thử việc không được nghỉ phép năm hưởng lương.

Câu hỏi: {question}
Câu trả lời của model: {model_answer}

Hãy đánh giá câu trả lời trên:
- Nếu câu trả lời đúng, chính xác theo chính sách hiện hành và đầy đủ -> label = 1.
- Nếu câu trả lời sai sự thật, dùng chính sách cũ hết hiệu lực, trả lời sai thẩm quyền phê duyệt/quy định -> label = 0.

Trả về JSON:
{{"label": 1 hoặc 0, "reasoning": "giải thích ngắn gọn"}}
"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia thẩm định chất lượng RAG. Chỉ trả lời JSON."},
                {"role": "user", "content": PROMPT},
            ],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        label = int(parsed.get("label", 0))
        return {"label": 1 if label == 1 else 0, "reasoning": parsed.get("reasoning", "")}
    except Exception as e:
        return {"label": 0, "reasoning": f"Judge error: {e}"}


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE B: LLM-as-Judge Evaluation")
    print("=" * 60)

    # 1. Evaluate 10 human-labeled questions for Cohen's Kappa
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)

    print(f"\n[1/3] Judging {len(human_data)} questions from human_labels_10q.json...")
    judge_labels = []
    comparison = []
    for item in human_data:
        res = judge_single_answer(item["question"], item["model_answer"])
        j_label = res["label"]
        judge_labels.append(j_label)
        comparison.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "model_answer": item["model_answer"],
            "human_label": item["human_label"],
            "judge_label": j_label,
            "human_note": item.get("human_note", ""),
            "judge_reasoning": res["reasoning"],
            "agreed": (item["human_label"] == j_label),
        })

    human_labels = [item["human_label"] for item in human_data]
    kappa = cohen_kappa(judge_labels, human_labels)
    agreed_count = sum(1 for c in comparison if c["agreed"])
    print(f"  ✓ Cohen's κ: {kappa:.4f} ({agreed_count}/{len(human_data)} agreements)")

    # 2. Pairwise judge + swap-and-average on pairs
    print("\n[2/3] Running pairwise swap-and-average benchmarks...")
    sample_pairs = [
        (
            "Nhân viên được nghỉ bao nhiêu ngày phép năm?",
            "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành.",
            "Theo quy định v2023 cũ, nhân viên có 12 ngày phép hàng năm.",
        ),
        (
            "Nhân viên kết hôn được nghỉ bao nhiêu ngày?",
            "Nhân viên được nghỉ 3 ngày làm việc hưởng nguyên lương khi kết hôn.",
            "Nhân viên được nghỉ 1 ngày khi kết hôn theo luật cơ bản.",
        ),
        (
            "Muốn mua thiết bị 55 triệu cần cấp nào phê duyệt?",
            "Thiết bị trên 50 triệu cần Tổng Giám đốc (CEO) phê duyệt.",
            "Cần Giám đốc phòng ban (Director) phê duyệt mọi thiết bị.",
        ),
        (
            "Thưởng Tết cho nhân viên chính thức trên 6 tháng là bao nhiêu?",
            "Nhân viên chính thức từ 6 tháng trở lên được thưởng Tết tối thiểu 1 tháng lương.",
            "Thưởng Tết phụ thuộc vào phòng ban, không có mức tối thiểu quy định.",
        ),
        (
            "Nhân viên thử việc có được hưởng ngày nghỉ phép năm có lương không?",
            "Nhân viên thử việc không được hưởng phép năm có lương, nếu nghỉ phải xin nghỉ không lương.",
            "Nhân viên thử việc được hưởng 1 ngày phép năm mỗi tháng.",
        ),
    ]

    judge_results: list[JudgeResult] = []
    for q, a_a, a_b in sample_pairs:
        jr = swap_and_average(q, a_a, a_b)
        judge_results.append(jr)
        print(f"  Q: {q[:40]}... -> Final: {jr.final_winner} (consistent: {jr.position_consistent})")

    # 3. Bias report
    print("\n[3/3] Generating bias report...")
    bias = bias_report(judge_results)
    print(f"  ✓ Position bias rate: {bias['position_bias_rate']}")
    print(f"  ✓ Verbosity bias: {bias['verbosity_bias']}")

    # Save report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/judge_results.json"
    report_data = {
        "cohen_kappa": round(kappa, 4),
        "cohen_kappa_interpretation": (
            "Substantial agreement" if kappa >= 0.6 else "Moderate agreement"
        ),
        "human_evaluation_comparison": comparison,
        "bias_report": bias,
        "pairwise_results": [
            {
                "question": r.question,
                "winner_pass1": r.winner_pass1,
                "winner_pass2": r.winner_pass2,
                "final_winner": r.final_winner,
                "position_consistent": r.position_consistent,
                "scores_pass1": r.scores_pass1,
                "scores_pass2": r.scores_pass2,
            }
            for r in judge_results
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved Phase B report → {report_path}")
