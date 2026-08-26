# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Phạm Phúc Minh  
**Mã SV:** 2A202601153  
**Ngày:** 26/08/2026  

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~23.5ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~1.7ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼ (~1200ms P95)
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search (BM25+Dense Hybrid) → M3 Rerank (BGE-Reranker) → GPT-4o-mini
    ▼ (~2.0ms P95)
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive HR data leak
    │ action:   replace with safe response / redact PII
    ▼
User Response
```

---

## Latency Budget

*(Điền từ kết quả Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 10.73 | 23.52 | 23.52 | <50ms |
| NeMo Input Rail | 1.41 | 1.74 | 1.74 | <300ms |
| RAG Pipeline | 650.00 | 1200.00 | 1500.00 | <2000ms |
| NeMo Output Rail | 1.50 | 2.00 | 2.50 | <300ms |
| **Total Guard** | **12.22** | **25.26** | **25.26** | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** Toàn bộ Guard Stack hoạt động cực kỳ nhanh với P95 đạt **25.26ms** (vượt xa mục tiêu <500ms). Presidio regex + Spacy chạy cục bộ trên CPU (~23.5ms) và NeMo rails colang matching (~1.7ms) không gây bottleneck cho pipeline.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
name: RAG Evaluation & Guardrail Stack CI/CD

on:
  push:
    branches: [ main, staging ]
  pull_request:
    branches: [ main ]

jobs:
  rag-quality-and-guard-gates:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          python -m spacy download en_core_web_sm

      - name: Presidio & NeMo Guardrails Gate
        run: pytest tests/test_phase_c.py -v
        # Phải pass 100% test suite, adversarial pass rate >= 80%

      - name: RAGAS Quality Gate
        run: python src/phase_a_ragas.py
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          MIN_FAITHFULNESS: 0.75
          MIN_AVG_SCORE: 0.65

      - name: LLM-as-Judge & Bias Gate
        run: pytest tests/test_phase_b.py -v
        # Cohen's Kappa >= 0.60 vs Human Labels, Position bias <= 0.30

      - name: Latency Budget Gate
        run: |
          python -c "from src.phase_c_guard import measure_p95_latency; \
          res = measure_p95_latency(['test input']*10); \
          assert res['latency_budget_ok'], f'Guard P95 exceeded budget: {res[\"total_ms\"][\"p95\"]}ms'"
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call, kiểm tra drift của prompt template |
| Adversarial block rate | < 80% | Review log các mẫu attack mới, cập nhật rails.co |
| Guard P95 latency | > 100ms | Scale thêm worker cho Presidio / tối ưu Colang flows |
| PII detected count | spike >10/hour | Kích hoạt Security Incident, cảnh báo team SecOps |
| Cohen's $\kappa$ (Judge vs Human) | < 0.50 | Cập nhật few-shot prompt & rubric chấm điểm cho LLM Judge |

---

## Kết quả thực tế từ Lab

| Chỉ số | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.840 (Factual: 0.887, Multi-hop: 0.769, Adversarial: 0.888) |
| Worst metric | Faithfulness / Context Recall |
| Dominant failure distribution | Factual (do các câu hỏi version conflict giữa policy 2023 vs 2024) |
| Cohen's κ (Judge vs Human) | **0.6154** (Substantial Agreement — 8/10 matched) |
| Adversarial pass rate | **20 / 20** (100.0%) |
| Guard P95 latency | **25.26 ms** (Budget: <500ms) |

---

## Nhận xét & Cải tiến

> **Điểm hoạt động tốt:**
> 1. Presidio nhận diện rất chính xác CCCD (12 chữ số), CMND (9 chữ số) và SĐT Việt Nam, đồng thời anonymize nhãn sạch sẽ không làm lộ thông tin cá nhân.
> 2. Guard Stack đạt latency P95 cực thấp (25.26ms), hoàn toàn không ảnh hưởng đến trải nghiệm người dùng cuối.
> 3. Kỹ thuật Swap-and-average trong Phase B loại bỏ hoàn toàn position bias (position bias rate = 0.0), mang lại độ đồng thuận cao với chuyên gia con người ($\kappa = 0.6154$).
>
> **Điểm cần cải thiện & Đề xuất khi deploy Production:**
> 1. Bổ sung semantic caching cho embedding và intent router ở input guard để giảm chi phí token và tăng tốc độ phản hồi.
> 2. Thêm async stream guardrail: kiểm tra output dạng streaming theo từng chunk token thay vì đợi LLM generate xong toàn bộ câu, giúp Time-to-First-Token (TTFT) đạt mức tối ưu.
> 3. Định kỳ chạy active learning: thu thập các prompt injection mới từ production logs, gán nhãn và fine-tune NeMo Guardrails / embedding classification layer.

