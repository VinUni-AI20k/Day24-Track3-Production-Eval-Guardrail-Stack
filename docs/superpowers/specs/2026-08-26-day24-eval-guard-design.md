# Day 24 Eval + Guardrail Stack — Design

## Scope

Complete Lab 24 in the current repository without changing its tests. The work
implements Phase A (RAGAS 50-question evaluation), Phase B (LLM-as-Judge),
Phase C (Presidio and NeMo guardrails), and the resulting reports. Day 18's six
source modules are copied from `/home/long/project/K34-Day18-2A202601711-LeVanLong`.

## Integration and configuration

Copy `m1_chunking.py`, `m2_search.py`, `m3_rerank.py`, `m4_eval.py`,
`m5_enrichment.py`, and `pipeline.py` into `src/`. Extend the Day 24
configuration with the Day 18 LLM-provider compatibility values used by those
modules (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, retries, timeout, device,
enrichment toggle, and `get_llm_client`) while retaining the Day 24 file paths
and collection name. Secrets remain in `.env` and are never committed.

`setup_answers.py` remains the sole producer of `answers_50q.json`. Phase A
uses that file unchanged and maps the established `evaluate_ragas()` return
shape: `dict["per_question"]` contains `EvalResult` objects.

## Phase A

`group_by_distribution()` always returns the three expected groups.
`run_ragas_50q()` invokes Day 18's evaluator once for the full answer set and
maps every per-question result to `RagasResult`. `bottom_10()` sorts ascending,
adds a diagnosis and suggested fix from `DIAGNOSTIC_TREE`, and the persisted
report contains that enriched output. `cluster_analysis()` produces a complete
metric-by-distribution matrix plus a non-empty dominant-failure insight.

## Phase B

`pairwise_judge()` requests JSON from `gpt-4o-mini`, validates winner and
clamps/converts scores to floats in `[0, 1]`. `swap_and_average()` calls the
judge in original and swapped order, maps the second winner and score keys back
to original A/B space, then uses agreement only for a decisive final winner.

`cohen_kappa()` implements the binary-label formula, validates equal non-empty
input, returns `1.0` for perfect agreement, and handles degenerate expected
agreement. `bias_report()` measures inconsistency and preference for the longer
winning answer. The script judges the 10 human-label examples and writes a JSON
report using dataclass serialization.

## Phase C

`pii_scan()` uses Presidio plus the supplied Vietnamese CCCD and phone
recognizers. The guard order remains fixed: Presidio, NeMo input rail, RAG
pipeline, then NeMo output rail. NeMo calls are asynchronous. Each public
synchronous batch function owns exactly one `asyncio.run()` invocation; it
never invokes it inside an item loop. Latency percentile calculation includes
P50/P95/P99 for Presidio, NeMo, and their combined total.

If the current NeMo flow coverage misses adversarial inputs, extend only
`guardrails/rails.co` with targeted jailbreak, off-topic, and prompt-injection
patterns. Do not weaken or disable Presidio.

## Reports and verification

The phase scripts write `reports/ragas_50q.json`, `reports/judge_results.json`,
and `reports/guard_results.json`. Analysis markdown uses actual outputs rather
than invented scores. The blueprint is completed after latency and suite
measurements are available. Verification consists of phase tests, the entire
pytest suite, and `python check_lab.py`; external-model results require a
running Qdrant service, installed dependencies, and configured API key.
