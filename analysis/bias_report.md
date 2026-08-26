# LLM Judge Bias Report — Phase B

**Sinh viên:** Trần Việt Trường  
**Ngày:** 27/08/2026  
**Judge model:** deterministic offline fallback (API-ready: gpt-4o-mini)

---

## 1. Pairwise Judge Results

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---:|---|---|---|
| 1 | Nghỉ kết hôn | B | B đầy đủ hơn về việc không trừ phép năm |
| 2 | Thiết bị 55 triệu | B | B nêu đúng ngưỡng và CEO phê duyệt |
| 3 | Thưởng Tết tối thiểu | B | B bao phủ điều kiện 6 tháng và pro-rata |
| 4 | Senior 9 năm | tie | Hai câu có độ bao phủ và tính cụ thể tương đương |
| 5 | Hoàn trả khóa học | B | B nêu rõ cam kết, tỷ lệ và số tiền |

---

## 2. Swap-and-Average Results

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---:|---|---|---|---|
| 1 | B | B | B | Yes |
| 2 | B | B | B | Yes |
| 3 | B | B | B | Yes |
| 4 | tie | tie | tie | Yes |
| 5 | B | B | B | Yes |
| 6 | B | B | B | Yes |
| 7 | B | B | B | Yes |
| 8 | tie | tie | tie | Yes |
| 9 | tie | tie | tie | Yes |
| 10 | B | B | B | Yes |

**Position bias rate:** 0% (0/10 case không nhất quán)

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 6 label=1, 4 label=0)  
**Judge labels:** `[1, 1, 1, 1, 1, 1, 1, 0, 1, 0]`

| Question ID | Human Label | Judge Label | Agree? |
|---:|---:|---:|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 1 | No |
| 12 | 1 | 1 | Yes |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 1 | Yes |
| 29 | 0 | 1 | No |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 0 | Yes |

**Cohen's κ:** 0,5455  
**Interpretation:** moderate agreement

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: 0 / 7 cases
- B thắng + B dài hơn A: 7 / 7 cases  
- **Verbosity bias rate:** 100%

**Kết luận:** Judge có xu hướng chọn câu dài hơn trong toàn bộ case decisive. Một phần là do ground truth thực sự đầy đủ hơn, nhưng tỷ lệ 100% vẫn cảnh báo verbosity bias; cần tách điểm completeness khỏi conciseness và giới hạn độ dài trước khi so sánh.

---

## 5. Nhận xét chung

> κ = 0,5455 chưa vượt 0,6 nên judge chỉ đạt mức moderate và chưa đủ tin cậy để tự động quyết định release. Position bias không đáng lo trong tập này vì hai lượt swap nhất quán 100%. Swap-and-average vẫn hữu ích vì biến bất đồng thứ tự thành tie thay vì kết luận sai. Trong production nên pin model/prompt, lấy mẫu human review, theo dõi verbosity bias và chỉ dùng judge như một CI signal kết hợp với metric xác định.
