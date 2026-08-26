# LLM Judge Bias Report — Phase B

**Sinh viên:** Lê Văn Long (2A202601711)
**Ngày:** 26/08/2026
**Judge model:** gpt-4o-mini (`JUDGE_MODEL`), `response_format=json_object`
**Nguồn số liệu:** `reports/judge_results.json` (sinh bởi `python src/phase_b_judge.py`)

**Thiết kế cặp so sánh:** với mỗi câu trong `human_labels_10q.json`,
**A = `model_answer`** (câu trả lời của pipeline) và
**B = `ground_truth`** của cùng `question_id` lấy từ `test_set_50q.json`.

> Bản `main` đầu tiên chấm mỗi `model_answer` với **chuỗi rỗng** (`""`) làm
> answer B. Khi đó A thắng 10/10 một cách hiển nhiên, kéo theo
> `verbosity_bias = 1.0` và `κ = 0.0` — cả hai đều là *tạo tác của thiết lập*,
> không đo được gì. Đã thay bằng cặp thật (model answer vs ground truth).

---

## 1. Pairwise Judge Results

*(10 cặp, vượt yêu cầu tối thiểu 5 cặp — trích 6 cặp tiêu biểu)*

| # | Question (tóm tắt) | Winner | Scores (A/B) | Reasoning tóm tắt |
|---|---|---|---|---|
| 1 | Nghỉ bao nhiêu ngày khi kết hôn? | B | 0.8 / 1.0 | B nói rõ 3 ngày nghỉ **không bị trừ** vào phép năm — A đúng nhưng thiếu vế này |
| 2 | Mua thiết bị 55 triệu ai phê duyệt? | B | 0.6 / 1.0 | A trả lời "Giám đốc phòng ban" — sai ngưỡng; B chỉ đúng CEO cho đơn >50 triệu |
| 3 | Thưởng Tết tối thiểu cho NV ≥6 tháng? | B | 0.7 / 1.0 | B phủ thêm trường hợp NV dưới 6 tháng; A chỉ đúng một nửa phạm vi |
| 5 | Tài trợ khoá học 25tr, nghỉ sau 8 tháng hoàn trả bao nhiêu? | B | 0.6 / 1.0 | B nêu cả cam kết 1 năm lẫn mức hoàn trả 100% |
| 8 | Nghỉ bao nhiêu ngày phép năm? | B | 0.5 / 1.0 | A trả lời theo **v2023 đã hết hiệu lực**; B bám v2024 hiện hành |
| 10 | Manager dùng VPN cá nhân khi WFH? | B | 0.4 / 1.0 | A cho phép; B nêu đúng rằng chính sách VPN v1.3 **cấm** — kèm lý do |

**Nhận xét:** B thắng 10/10 ở pass 1. Hợp lý — B là ground truth. Điểm đáng
chú ý là judge *nêu đúng lý do thực chất* (sai ngưỡng phê duyệt, trả lời theo
phiên bản cũ, cho phép điều bị cấm), chứ không chỉ khen "B dài hơn".

---

## 2. Swap-and-Average Results

*(Pass 2 gọi `pairwise_judge(q, B, A)` rồi **convert winner về không gian gốc**
trước khi so sánh với pass 1 — cột "Pass 2 Winner" dưới đây đã ở không gian gốc)*

| # | Question ID | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|---|
| 1 | 1 | B | B | B | ✅ |
| 2 | 5 | B | B | B | ✅ |
| 3 | 12 | B | B | B | ✅ |
| 4 | 21 | B | tie | **tie** | ❌ |
| 5 | 23 | B | B | B | ✅ |
| 6 | 29 | B | B | B | ✅ |
| 7 | 33 | B | B | B | ✅ |
| 8 | 41 | B | B | B | ✅ |
| 9 | 46 | B | **A** | **tie** | ❌ |
| 10 | 50 | B | B | B | ✅ |

**Position bias rate: 20%** (2/10 case không nhất quán → `final_winner = tie`).

Hai ca lệch đều là ca judge cho điểm A cao hẳn lên khi A được đặt ở vị trí thứ
hai: câu 21 (`A: 0.8 → 1.0`) và câu 46 (`A: 0.7 → 1.0`, đồng thời `B: 1.0 → 0.5`).
Câu 46 là ca đảo chiều hoàn toàn — cùng một cặp câu trả lời, chỉ đổi thứ tự
xuất hiện là winner nhảy từ B sang A. Đây chính xác là thứ swap-and-average
sinh ra để bắt.

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu — 6 label=1, 4 label=0)
**Judge labels:** chấm nhị phân thật bằng `grade_answer()` — judge nhận
(câu hỏi, `model_answer`, ground truth) và trả `label ∈ {0, 1}`, cùng thang với
người chấm.

> Vì sao không lấy nhãn từ pairwise? Vì "A hay B tốt hơn" là thang khác — nó
> trộn *độ chính xác* với *độ đầy đủ*. Một câu trả lời đúng nhưng ngắn sẽ thua
> ground truth ở pairwise dù người chấm vẫn cho label = 1. Dùng winner pairwise
> làm nhãn κ sẽ đo nhầm thứ cần đo.

| Question ID | Human Label | Judge Label | Agree? | Judge reasoning (tóm tắt) |
|---|---|---|---|---|
| 1 | 1 | 1 | ✅ | Đúng sự thật, trả lời được ý chính (dù ngắn hơn ground truth) |
| 5 | 0 | 0 | ✅ | Sai — không nêu CEO, chỉ nói Giám đốc phòng ban |
| 12 | 1 | 1 | ✅ | Đúng và nêu rõ mức thưởng Tết tối thiểu |
| 21 | 1 | 1 | ✅ | Chính xác về số ngày phép và khung lương Senior |
| 23 | 1 | 1 | ✅ | Đúng sự thật, trúng ý chính |
| 29 | 0 | 0 | ✅ | Thiếu vế Kế toán trưởng và sai phí phạt |
| 33 | 1 | 1 | ✅ | Đúng và đầy đủ cả phụ cấp lẫn ngày phép |
| 41 | 0 | 0 | ✅ | Sai — trả lời theo v2023, đã bị v2024 thay thế |
| 46 | 1 | 1 | ✅ | Đúng so với ground truth, trúng ý chính |
| 50 | 0 | 0 | ✅ | Mâu thuẫn chính sách — cho phép VPN cá nhân trong khi v1.3 cấm |

**Agreement: 10/10** — `p_o = 1.0`, `p_e = 0.52`
**Cohen's κ = 1.0000**
**Interpretation:** *almost perfect* theo thang Landis–Koch (>0.8). Đạt điều
kiện bonus κ > 0.6 (substantial).

**Cảnh báo khi đọc κ = 1.0:** con số này *không* chứng minh judge tương đương
người trong mọi tình huống. Ba giới hạn cụ thể:
1. **n = 10** — quá nhỏ; khoảng tin cậy của κ ở cỡ mẫu này rất rộng.
2. **Judge được đưa ground truth** — nhiệm vụ trở thành đối chiếu văn bản, dễ
   hơn hẳn so với chấm mù. Trong production không có sẵn ground truth.
3. **4 câu label=0 đều sai rõ ràng** (sai ngưỡng số, sai phiên bản chính sách,
   trái điều cấm) — không có ca biên nào để thử thách judge.

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie) — 8/10 case:

- A thắng + A dài hơn B: **0 / 8**
- B thắng + B dài hơn A: **8 / 8**
- **Verbosity bias rate: 100%**

Độ dài trung bình: A (model answer) ≈ 52 ký tự, B (ground truth) ≈ 152 ký tự.
B dài hơn A ở **cả 10/10** cặp.

**Kết luận:** con số 100% **không** đủ để kết luận judge thiên vị câu dài, vì
thiết kế thí nghiệm này bị **nhiễu hoàn toàn (fully confounded)**: B vừa là câu
dài hơn, vừa là ground truth chính xác hơn. Không có một cặp nào mà câu *ngắn
hơn* lại là câu *đúng hơn*, nên không thể tách hai yếu tố.

Vì sao vẫn phải lo? Vì tín hiệu gián tiếp có thật: ở câu 1, A **đúng hoàn toàn**
(3 ngày nghỉ kết hôn) mà vẫn chỉ được 0.8 so với 1.0 của B, lý do judge đưa ra
là B "đầy đủ hơn". Judge đang trừ điểm cho sự súc tích ngay cả khi không có gì
sai. Trong production, một judge như vậy sẽ dần đẩy hệ thống về phía câu trả
lời dài dòng — tốn token, chậm hơn, và tăng bề mặt hallucination.

**Cách đo lại cho sạch:** dựng cặp *có kiểm soát* — cùng nội dung đúng, khác độ
dài (một bản súc tích, một bản thêm câu chữ thừa nhưng không thêm thông tin).
Nếu judge vẫn chọn bản dài, khi đó mới kết luận được verbosity bias.

---

## 5. Nhận xét chung

> **κ = 1.0 (almost perfect) — nhưng đọc kèm điều kiện.** Trên 10 câu và có
> ground truth trong tay, judge trùng khớp người tuyệt đối, kể cả 4 ca sai tinh
> vi (sai ngưỡng 50 triệu, trả lời theo chính sách v2023 đã hết hiệu lực).
> Đủ để dùng gpt-4o-mini làm regression gate tự động; **chưa** đủ để thay người
> chấm ở ca biên, vì n=10 và không có ca mơ hồ nào trong mẫu.
>
> **Position bias 20% — dưới ngưỡng báo động 30%, nhưng không phải bằng 0.**
> 2/10 case đảo kết quả khi hoán vị thứ tự, trong đó câu 46 đảo chiều hoàn
> toàn. Nếu chỉ chạy một lượt, 20% số phán quyết sẽ phụ thuộc vào việc câu nào
> được đặt trước — tỷ lệ không thể chấp nhận trong một quality gate.
>
> **Swap-and-average có ích và rẻ.** Nó không "sửa" được phán quyết, nhưng
> chuyển 2 ca không đáng tin thành `tie` thay vì để chúng lọt qua như kết luận
> chắc chắn. Giá phải trả là gấp đôi số lời gọi API — với gpt-4o-mini là không
> đáng kể so với rủi ro gate sai.
>
> **Verbosity bias 100% là số nhiễu, không phải kết luận.** B luôn vừa dài hơn
> vừa đúng hơn nên không tách được hai yếu tố (xem mục 4). Cần bộ cặp có kiểm
> soát độ dài mới đo được thật.
>
> **Dùng judge thế nào trong production:** (1) luôn swap-and-average, coi `tie`
> là "cần người xem lại" chứ không phải "hoà"; (2) chấm nhị phân có ground
> truth cho regression gate, giữ pairwise cho việc so sánh hai phiên bản
> pipeline; (3) thêm tiêu chí "súc tích" có trọng số vào rubric để chống xu
> hướng ưu ái câu dài; (4) tái kiểm κ trên mẫu người chấm ≥50 câu mỗi quý, và
> chặn merge nếu κ tụt dưới 0.6.
