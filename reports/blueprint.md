# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Trần Việt Trường  
**Ngày:** 27/08/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~0,014ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~0,014ms P95)
[NeMo Input Rail + deterministic fallback]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → answer
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   redact or replace with safe response
    ▼
User Response
```

---

## Latency Budget

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---:|---:|---:|---|
| Presidio PII | 0,010 | 0,014 | 0,025 | <10ms |
| NeMo Input Rail | 0,011 | 0,014 | 0,021 | <300ms |
| RAG Pipeline | Không đo riêng | Không đo riêng | Không đo riêng | <2000ms |
| NeMo Output Rail | Không đo riêng | Không đo riêng | Không đo riêng | <300ms |
| **Total Guard** | **0,021** | **0,026** | **0,046** | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** Guard chạy local bằng regex và rule xác định nên thấp hơn nhiều so với ngân sách. Khi bật NeMo qua API, cần đo lại vì độ trễ mạng và LLM sẽ là bottleneck chính.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

Kết quả hiện tại vượt các gate: faithfulness 1,000; avg_score 0,8353; adversarial 20/20; P95 guard 0,026ms.

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0,8353 |
| Worst metric | answer_relevancy |
| Dominant failure distribution | adversarial |
| Cohen's κ | 0,5455 (moderate) |
| Adversarial pass rate | 20 / 20 (100%) |
| Guard P95 latency | 0,026 ms |

---

## Nhận xét & Cải tiến

> Guardrail local hoạt động tốt, chặn đủ 20/20 mẫu tấn công và giữ độ trễ rất thấp. RAG pipeline đạt faithfulness cao nhưng answer_relevancy còn yếu ở câu multi-hop và adversarial vì câu trả lời extractive thường chứa thông tin thừa. Cohen's κ ở mức moderate cho thấy judge chỉ nên là tín hiệu hỗ trợ, chưa thay thế đánh giá con người. Khi triển khai production cần bật NeMo/OpenAI bằng secret manager, đo lại latency API, thêm metadata phiên bản chính sách và bắt buộc human review với các case judge không chắc chắn.
