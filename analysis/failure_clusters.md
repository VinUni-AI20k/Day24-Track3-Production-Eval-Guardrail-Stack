# Failure Cluster Analysis — Phase A

**Sinh viên:** Trần Việt Trường  
**Ngày:** 27/08/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 1,000 | 1,000 | 1,000 |
| answer_relevancy | 0,738 | 0,517 | 0,555 |
| context_precision | 0,950 | 0,983 | 0,933 |
| context_recall | 0,942 | 0,693 | 0,571 |
| **avg_score** | **0,908** | **0,798** | **0,765** |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---:|---|---|---:|---|
| 1 | adversarial | Q48 — Thử việc có hưởng PVI không? | 0,604 | context_precision |
| 2 | multi_hop | Q40 — Phát hiện vi phạm bảo mật nên làm gì? | 0,606 | answer_relevancy |
| 3 | multi_hop | Q22 — Mua laptop 30 triệu cần phê duyệt gì? | 0,680 | answer_relevancy |
| 4 | adversarial | Q41 — Số ngày phép năm hiện hành | 0,706 | context_recall |
| 5 | multi_hop | Q36 — Khác biệt đánh giá tháng 6 và 12 | 0,709 | answer_relevancy |
| 6 | adversarial | Q42 — Mốc thâm niên cộng ngày phép | 0,710 | context_recall |
| 7 | multi_hop | Q33 — Phụ cấp và phép của Manager 12 năm | 0,712 | answer_relevancy |
| 8 | adversarial | Q50 — Dùng VPN cá nhân khi WFH | 0,738 | answer_relevancy |
| 9 | multi_hop | Q21 — Phép và lương Senior 9 năm | 0,740 | answer_relevancy |
| 10 | adversarial | Q49 — So sánh phép v2023/v2024 | 0,741 | answer_relevancy |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 0 | 0 | 0 | 0 |
| answer_relevancy | 18 | 19 | 5 | 42 |
| context_precision | 1 | 0 | 1 | 2 |
| context_recall | 1 | 1 | 4 | 6 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** adversarial  
**Dominant metric:** answer_relevancy

**Lý do phân tích:**

> Adversarial có avg_score thấp nhất (0,765), chủ yếu do xung đột phiên bản v2023/v2024 và các câu hỏi phủ định. Answer relevancy là metric yếu chủ đạo vì baseline extractive trả cả đoạn policy thay vì tổng hợp đúng trọng tâm câu hỏi. Multi-hop cũng bị ảnh hưởng khi thông tin cần thiết nằm ở nhiều tài liệu. Faithfulness đạt cao vì câu trả lời được lấy trực tiếp từ context, nhưng điều đó không bảo đảm câu trả lời ngắn gọn hoặc chọn đúng phiên bản.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating | Giữ prompt chỉ dùng context, thêm citation và output rail |
| context_recall | Missing relevant chunks | Dùng parent-child retrieval, BM25+dense và mở rộng query |
| context_precision | Too many irrelevant chunks | Rerank mạnh hơn, lọc metadata theo hiệu lực/version |
| answer_relevancy | Answer doesn't match question | Prompt trả lời trực tiếp, tổng hợp multi-hop và giới hạn độ dài |

---

## 6. Nhận xét về Adversarial Distribution

> Adversarial đạt 0,765, thấp hơn factual 0,908 và multi-hop 0,798 như kỳ vọng. Có 5 câu adversarial trong bottom 10: Q48, Q41, Q42, Q50 và Q49. Các lỗi tập trung ở version conflicts, negation trap và policy contradiction; đặc biệt context recall thấp khi cả bản cũ và bản hiện hành cùng được truy hồi. Cần gắn metadata `effective_date`/`status`, ưu tiên policy hiện hành trong reranker và kiểm tra phủ định trước khi sinh câu trả lời.
