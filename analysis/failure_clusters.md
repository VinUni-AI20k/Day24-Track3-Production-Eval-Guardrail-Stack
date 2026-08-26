# Failure Cluster Analysis — Phase A

**Sinh viên:** Lê Văn Long (2A202601711)
**Ngày:** 26/08/2026
**Nguồn số liệu:** `reports/ragas_50q.json` (sinh bởi `python src/phase_a_ragas.py`)
**Judge/embedding của RAGAS:** `gpt-4o-mini` + `BAAI/bge-m3` (local)

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual (n=20) | multi_hop (n=20) | adversarial (n=10) |
|---|---|---|---|
| faithfulness | 0.933 | **0.453** | 0.900 |
| answer_relevancy | 0.777 | 0.587 | 0.755 |
| context_precision | 0.954 | 0.975 | 0.950 |
| context_recall | 0.883 | 0.779 | **0.650** |
| **avg_score** | **0.887** | **0.699** | **0.814** |

Đọc nhanh: `context_precision` cao đều ở cả 3 nhóm (0.95–0.98) → khâu rerank
lọc nhiễu tốt. Hai chỗ thủng là `faithfulness` của multi_hop (0.453) và
`context_recall` của adversarial (0.650) — hai lỗi khác bản chất, cần hai cách
sửa khác nhau.

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | multi_hop | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0 (độ dài, thời hạn đổi, MFA) | 0.125 | faithfulness |
| 2 | adversarial | Manager có thể dùng VPN cá nhân (NordVPN) khi WFH không? | 0.333 | faithfulness |
| 3 | multi_hop | Manager thâm niên 12 năm: tổng phụ cấp/tháng và số ngày phép v2024? | 0.375 | faithfulness |
| 4 | multi_hop | Senior 9 năm thâm niên: bao nhiêu ngày phép và lương khoảng nào? | 0.375 | faithfulness |
| 5 | multi_hop | So sánh quyền lợi bảo hiểm: thử việc vs chính thức | 0.375 | faithfulness |
| 6 | factual | Nam nhân viên nghỉ bao nhiêu ngày khi vợ sinh con? | 0.500 | faithfulness |
| 7 | multi_hop | Mua laptop 30 triệu: ai phê duyệt và cần gì từ CNTT? | 0.670 | context_recall |
| 8 | multi_hop | Công tác 2 ngày, khách sạn 1.500.000/đêm: thanh toán tối đa? | 0.676 | faithfulness |
| 9 | multi_hop | Thâm niên 7 năm v2024, trừ 4 ngày ốm không giấy: còn bao nhiêu? | 0.723 | faithfulness |
| 10 | multi_hop | Junior P1 lương 12 triệu vừa thử việc: lương và phụ cấp tháng đầu? | 0.732 | faithfulness |

**Phân bố bottom-10:** multi_hop 8/10, factual 1/10, adversarial 1/10.
Mỗi câu trong `reports/ragas_50q.json` đều kèm `diagnosis` + `suggested_fix`
tra từ `DIAGNOSTIC_TREE[worst_metric]`.

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu **failure** có worst_metric = row, thuộc distribution = col.
Một câu tính là failure khi metric yếu nhất của nó < 0.70 — xem
`FAILURE_THRESHOLD` trong `src/phase_a_ragas.py`.)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 2 | **14** | 1 | 17 |
| answer_relevancy | 0 | 3 | 0 | 3 |
| context_precision | 1 | 0 | 0 | 1 |
| context_recall | 5 | 2 | **8** | 15 |
| **Total failure** | 8/20 | 19/20 | 9/10 | 36/50 |
| **Failure rate** | 40% | **95%** | 90% | 72% |

> **Ghi chú về cách đếm.** Bản đầu tiên của `cluster_analysis()` đếm *mọi* câu
> chứ không chỉ câu hỏng, nên tổng mỗi cột luôn đúng bằng size của
> distribution (20/20/10) và `dominant_failure_distribution` chỉ phản ánh nhóm
> đông nhất. Đã sửa thành đếm câu thực sự failure, và so sánh giữa các nhóm
> theo **tỷ lệ** thay vì số tuyệt đối (adversarial chỉ có 10 câu, so số tuyệt
> đối với nhóm 20 câu thì luôn thiệt).

---

## 4. Dominant Failure Analysis

**Dominant distribution:** `multi_hop` (19/20 câu = 95% failure rate)
**Dominant metric:** `faithfulness` (17/36 failure, và điểm trung bình chỉ 0.453 ở multi_hop)

**Lý do phân tích:**

> Multi_hop hỏng nặng nhất vì mỗi câu đòi ghép ≥2 mảnh chính sách rồi **tính
> toán** trên đó: "Senior 9 năm thâm niên" phải lấy 15 ngày cơ bản từ mục nghỉ
> phép, cộng 3 ngày thâm niên (9÷3) từ mục khác, rồi tra khung lương ở một
> tài liệu thứ ba. `context_precision` vẫn 0.975 — nghĩa là retrieval **lấy
> đúng** các chunk cần thiết; cái hỏng nằm ở bước sinh câu trả lời. LLM ghép
> các con số lại rồi tự suy ra một kết quả không có câu nào trong context
> khẳng định, nên RAGAS chấm faithfulness rất thấp (câu #39 chỉ 0.125). Đây là
> lỗi *suy luận số học không được trích dẫn*, không phải lỗi truy hồi.
>
> Ngược lại, adversarial hỏng ở `context_recall` (0.650) chứ không phải
> faithfulness (0.900) — 8/9 câu adversarial failure có worst_metric =
> context_recall. Corpus chứa **cả v2023 lẫn v2024** của cùng một chính sách;
> ground truth chỉ tính phiên bản hiện hành, còn retriever trả về hỗn hợp hai
> phiên bản, nên phần context phủ được ground truth bị hụt (recall ≈ 0.67 ở 8
> câu). Faithfulness vẫn cao vì câu trả lời *có* bám vào context — chỉ là bám
> vào context của phiên bản đã hết hiệu lực.
>
> Factual là nhóm khoẻ nhất (avg 0.887, 40% failure) vì mỗi câu chỉ cần một
> chunk duy nhất, không phải ghép và không phải chọn phiên bản.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness (multi_hop, 0.453) | LLM tự tính toán/ghép số liệu rồi khẳng định như thể có trong tài liệu | Bắt buộc trích dẫn chunk-id cho **từng** con số trong câu trả lời; hạ `temperature` về 0; tách phép tính ra khỏi phần sinh văn bản (chain-of-thought cho ra công thức, rồi tính bằng code thay vì để LLM cộng nhẩm) |
| context_recall (adversarial, 0.650) | Corpus có nhiều phiên bản chính sách (v2023 + v2024); retriever trộn lẫn | **Metadata filter theo phiên bản hiện hành** (`policy_version == current`) ngay ở tầng Qdrant; đánh dấu tài liệu hết hiệu lực bằng `effective_to`; chỉ mở rộng sang bản cũ khi câu hỏi nói rõ "theo chính sách cũ" |
| context_precision (0.95–0.98, đạt) | Không phải điểm yếu ở lab này | Giữ nguyên rerank hiện tại (bge-reranker-v2-m3, top-3); chỉ theo dõi hồi quy |
| answer_relevancy (multi_hop, 0.587) | Câu hỏi nhiều vế, câu trả lời chỉ chạm một vế | Prompt template liệt kê từng vế của câu hỏi thành checklist và yêu cầu trả lời đủ từng vế |

---

## 6. Nhận xét về Adversarial Distribution

> **avg_score: factual 0.887 > adversarial 0.814 > multi_hop 0.699.** Bộ
> adversarial *có* làm pipeline tụt so với factual (−0.073), tức là test set
> stress-test đúng chức năng của nó. Nhưng nhóm khó nhất lại là multi_hop chứ
> không phải adversarial — chi tiết này quan trọng: câu bẫy về *phiên bản*
> hoá ra dễ hơn câu đòi *suy luận nhiều bước*.
>
> Pipeline **có** bị nhầm bởi version conflict, và dấu vết nằm ở
> `context_recall = 0.650` của nhóm adversarial. Các câu như "Nhân viên được
> nghỉ bao nhiêu ngày phép năm?" (id 41), "Bao lâu phải đổi mật khẩu?" (id 44),
> "Mật khẩu tối thiểu bao nhiêu ký tự?" (id 43) đều cố tình *không* nói rõ
> phiên bản; corpus có cả v2023 và v2024 nên retriever trả về cả hai, và
> recall so với ground truth (chỉ tính v2024) dừng ở ≈0.67.
>
> Câu adversarial duy nhất lọt bottom-10 là **id 50 — "Manager có thể dùng VPN
> cá nhân (NordVPN) khi WFH không?" (avg 0.333, faithfulness 0.00)**. Đây là
> ca xấu nhất trong ba kiểu: chính sách VPN v1.3 **cấm** VPN cá nhân, nhưng câu
> hỏi được gài chữ "để tăng bảo mật thêm" khiến model trả lời theo hướng cho
> phép — faithfulness rơi về 0 vì không một chunk nào chống lưng cho câu trả
> lời đó.
>
> **Kết luận về cách đọc số này:** faithfulness của adversarial vẫn 0.900,
> nghĩa là RAGAS không "hỏng" và guardrail không cần siết thêm ở tầng này. Tín
> hiệu thật sự là *phải thêm metadata filter theo phiên bản hiện hành* ở tầng
> retrieval. Nếu bỏ qua và chỉ đi siết prompt, pipeline sẽ vẫn trả lời trôi
> chảy bằng chính sách đã hết hiệu lực — kiểu sai nguy hiểm nhất trong hệ
> thống tra cứu chính sách nội bộ, vì câu trả lời trông hoàn toàn có căn cứ.
