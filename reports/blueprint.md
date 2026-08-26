# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Lê Văn Long (2A202601711)
**Ngày:** 26/08/2026
**Nguồn số liệu:** `reports/ragas_50q.json`, `reports/judge_results.json`, `reports/guard_results.json`

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (0.31ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (1435.68ms P95)
[NeMo Input Rail — self_check_input]
    │ block if: off-topic / jailbreak / prompt injection / đòi PII người khác
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail — self_check_output]
    │ flag if:  PII in response / mật khẩu / lương cá nhân / lộ system prompt
    │ action:   replace with safe response
    ▼
User Response
```

Thứ tự này được giữ đúng trong `run_adversarial_suite()`: Presidio chạy trước
(rẻ, cục bộ, chặn được 4/20 case mà không tốn một lời gọi LLM nào), NeMo chỉ
được gọi khi Presidio cho qua.

---

## Latency Budget

*(Đo bằng `measure_p95_latency()` — Task 12, n=10 input từ `adversarial_set_20.json`)*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget | Đạt? |
|---|---|---|---|---|---|
| Presidio PII | 0.22 | **0.31** | 0.31 | <10ms | ✅ |
| NeMo Input Rail | 786.27 | **1435.68** | 1435.68 | <300ms | ❌ |
| RAG Pipeline | — | — | — | <2000ms | *không đo ở lab này* |
| NeMo Output Rail | ~1300 (đo rời, 1 lượt) | — | — | <300ms | ❌ |
| **Total Guard** | 786.40 | **1435.99** | 1435.99 | **<500ms** | **❌** |

**Budget OK?** [ ] Yes / [x] **No** — vượt 2.9× ngưỡng 500ms.

**Comment:** Bottleneck là **NeMo input rail**, chiếm **99.98%** tổng thời gian
guard (1435.68ms / 1435.99ms). Presidio nhanh hơn NeMo khoảng **4600 lần** vì
nó chỉ chạy regex cục bộ, còn mỗi lần gọi input rail là một round-trip
`gpt-4o-mini` thật qua mạng. Đây là chi phí cố định của guardrail dựa trên LLM,
không phải lỗi cấu hình — đã dùng `options={"rails": ["input"]}` để chỉ chạy
đúng tầng input, tránh tốn thêm một lượt sinh câu trả lời đầy đủ (giảm từ
~7000ms xuống ~1400ms).

Cách tối ưu để về dưới 500ms, theo thứ tự đáng làm:
1. **Cache theo hash của input** — các query lặp lại (rất phổ biến trong tra
   cứu HR) trả về ngay, không gọi LLM.
2. **Tầng lọc rẻ trước tầng đắt** — cho một bộ phân loại nhẹ (embedding +
   logistic regression, hoặc regex cho các mẫu tấn công đã biết) xử lý phần
   lớn traffic; chỉ đẩy ca không chắc chắn sang LLM rail.
3. **Chạy song song Presidio và NeMo** thay vì tuần tự (chỉ tiết kiệm 0.31ms —
   gần như vô nghĩa ở đây, nên xếp cuối).
4. **Đổi sang model rail nhỏ hơn/self-hosted** — đánh đổi trực tiếp giữa độ trễ
   và chất lượng chặn; phải đo lại pass rate trước khi đổi.

Nếu buộc phải giữ ngưỡng 500ms cứng, lựa chọn trung thực là **nâng budget cho
tuyến có LLM rail** hoặc chuyển rail sang chế độ bất đồng bộ cho các luồng
không nhạy cảm — chứ không phải tắt rail để lấy số đẹp.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75     # faithfulness 50q ≥ 0.75
    MIN_AVG_SCORE: 0.65        # avg_score 50q ≥ 0.65

- name: Judge Reliability Gate
  run: python src/phase_b_judge.py
  env:
    MIN_COHEN_KAPPA: 0.60      # κ < 0.6 ⇒ judge không đủ tin cậy để làm gate

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%); mục tiêu 18/20 (90%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total guard < 500ms  → HIỆN ĐANG FAIL (1436ms), xem mục Latency Budget
```

| Gate | Ngưỡng | Kết quả hiện tại | Trạng thái |
|---|---|---|---|
| RAGAS faithfulness (50q) | ≥ 0.75 | **0.735** | ❌ |
| RAGAS avg_score (50q) | ≥ 0.65 | 0.797 | ✅ |
| Cohen's κ | ≥ 0.60 | 1.000 | ✅ |
| Adversarial pass rate | ≥ 15/20 | **20/20** | ✅ |
| Guard P95 latency | < 500ms | **1436ms** | ❌ |

**Hai gate đang fail — và đó là kết quả trung thực, không phải lỗi cấu hình.**

Gate faithfulness trượt ngưỡng 0.75 với 0.735, và toàn bộ khoảng cách đến từ
đúng một nhóm: multi_hop chỉ đạt 0.453 trong khi factual 0.933 và adversarial
0.900. Nếu chỉ nhìn số gộp, người đọc dễ kết luận "cả pipeline hơi yếu" — thực
tế là hai phần ba test set hoàn toàn khoẻ, còn một phần ba hỏng nặng. Vì vậy
gate nên tách theo distribution:

```yaml
MIN_FAITHFULNESS_FACTUAL: 0.85     # hiện tại 0.933 ✅
MIN_FAITHFULNESS_MULTI_HOP: 0.60   # hiện tại 0.453 ❌ — mục tiêu sửa trước tiên
MIN_FAITHFULNESS_ADVERSARIAL: 0.80 # hiện tại 0.900 ✅
```

Gate latency trượt vì lý do cấu trúc (LLM rail trên đường đi đồng bộ) — xem
mục Latency Budget. Cả hai đều được giữ nguyên trạng thái fail trong báo cáo
này thay vì nới ngưỡng cho vừa số đo.

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample 50q) | < 0.70 | Page on-call; chặn deploy tiếp theo |
| RAGAS faithfulness — riêng multi_hop | < 0.40 | Điều tra prompt/retrieval cho câu suy luận nhiều bước |
| RAGAS context_recall — riêng adversarial | < 0.60 | Kiểm tra metadata filter phiên bản chính sách còn hiệu lực |
| Adversarial block rate (suite chạy hàng đêm) | < 80% | Review mẫu tấn công mới; cập nhật `prompts.yml` |
| Guard P95 latency | > 600ms | Bật cache; scale NeMo model |
| Presidio P95 latency | > 10ms | Điều tra regression ở tầng regex |
| PII detected count | spike > 10/hour | Security alert — có thể là rò rỉ dữ liệu ở phía client |
| Tỷ lệ `tie` từ LLM judge | > 30% | Judge mất ổn định; kiểm tra lại position bias |
| Cohen's κ (kiểm định hàng quý, ≥50 câu người chấm) | < 0.60 | Ngừng dùng judge làm gate cho tới khi hiệu chỉnh lại |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | **0.797** (factual 0.887 / multi_hop 0.699 / adversarial 0.814) |
| RAGAS faithfulness (50q) | **0.735** — dưới ngưỡng gate 0.75 |
| Worst metric | **faithfulness** — 17/36 failure; chỉ 0.453 ở multi_hop |
| Dominant failure distribution | **multi_hop** — 19/20 câu failure (95%) |
| Cohen's κ | **1.0000** (*almost perfect*, Landis–Koch >0.8; n=10) |
| Adversarial pass rate | **20 / 20** (Presidio chặn 4, NeMo input rail chặn 16) |
| Guard P95 latency | **1435.99 ms** (Presidio 0.31ms + NeMo 1435.68ms) |
| Latency budget (<500ms) | ❌ **không đạt** |

---

## Nhận xét & Cải tiến

> **Hoạt động tốt.** Guard stack chặn đúng 20/20 adversarial input, và điều
> đáng giá hơn con số là *cách* nó chặn: Presidio bắt gọn 4 case có CCCD/CMND/
> SĐT/email ngay tại tầng regex với 0.31ms P95 — không tốn một lời gọi LLM nào,
> đúng nguyên tắc "tầng rẻ đứng trước tầng đắt". 16 case còn lại (jailbreak,
> off-topic, prompt injection) do NeMo input rail xử lý. Output rail cũng chặn
> đúng câu trả lời chứa CCCD + SĐT và thay bằng câu an toàn thay vì để lộ.
>
> **Cần cải thiện — độ trễ.** P95 1436ms vượt 2.9× ngân sách 500ms, và bottleneck
> gần như tuyệt đối là LLM rail (99.98%). Bất kỳ guardrail nào đặt một lời gọi
> LLM trên đường đi đồng bộ đều phải chấp nhận sự thật này. Nếu deploy thật, tôi
> sẽ đặt cache theo hash input trước rail và một tầng phân loại nhẹ để LLM rail
> chỉ nhận phần traffic thật sự mơ hồ — đó là cách duy nhất giữ được chất lượng
> chặn mà vẫn về gần 500ms.
>
> **Cần cải thiện — chất lượng câu trả lời.** faithfulness của multi_hop chỉ
> 0.453: pipeline lấy đúng chunk (context_precision 0.975) nhưng LLM tự cộng
> nhẩm ra kết quả không chunk nào khẳng định. Phần tính toán phải tách khỏi phần
> sinh văn bản. Song song đó, context_recall 0.650 ở nhóm adversarial cho thấy
> corpus trộn v2023 với v2024 mà retriever không phân biệt — cần metadata filter
> theo phiên bản hiện hành ngay ở tầng Qdrant.
>
> **Ba thay đổi tôi sẽ làm trước khi đưa stack này ra production:** (1) cache +
> tầng lọc nhẹ trước NeMo rail để cứu độ trễ; (2) metadata filter theo phiên bản
> chính sách — trả lời trôi chảy bằng chính sách đã hết hiệu lực là kiểu sai
> nguy hiểm nhất ở hệ thống tra cứu nội bộ, vì nó *trông* có căn cứ; (3) tách
> CI gate faithfulness theo từng distribution, vì số gộp 0.735 đang che một điểm
> yếu 0.453 nằm gọn trong nhóm multi_hop.
>
> **Một lưu ý về việc tự đánh giá.** κ = 1.0 nghe rất đẹp nhưng chỉ trên 10 câu
> và judge được đưa sẵn ground truth. Tôi sẽ không dùng con số này để tuyên bố
> judge thay được người chấm — nó đủ để làm regression gate tự động, không đủ để
> kết luận về ca biên.
