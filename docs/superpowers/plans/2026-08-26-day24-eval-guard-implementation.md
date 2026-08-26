# Day 24 Eval + Guardrail Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Day 24 RAG evaluation, LLM judge, and production guardrail stack with validated outputs and reports.

**Architecture:** Copy the Day 18 RAG implementation unchanged, adding only configuration compatibility in Day 24. Phase A maps its RAGAS results into a 50-question report; Phase B uses a two-pass swapped LLM comparison; Phase C applies Presidio before asynchronous NeMo rails.

**Tech Stack:** Python, pytest, OpenAI SDK, RAGAS, Presidio, NeMo Guardrails, spaCy, Qdrant, scipy.

## Global Constraints

- Do not modify `tests/` to hide defects or commit `.env`/credentials.
- Copy exactly the six Day 18 modules from `/home/long/project/K34-Day18-2A202601711-LeVanLong/src/`.
- `setup_answers.py` is the sole producer of the unmodified Phase A input.
- Preserve guard order: Presidio, NeMo input, RAG, NeMo output.
- Batch guard functions invoke `asyncio.run()` only once, outside item loops.
- Map the swapped judge result back to original A/B before consensus.
- Extend `guardrails/rails.co` only if live adversarial results require it; retain Presidio and its language setting.
- Final gates: `pytest tests/ -v`, then `python check_lab.py`.

---

### Task 1: Integrate the Day 18 pipeline

**Files:**
- Modify: `config.py`
- Create: `src/m1_chunking.py`, `src/m2_search.py`, `src/m3_rerank.py`, `src/m4_eval.py`, `src/m5_enrichment.py`, `src/pipeline.py`
- Test: `check_lab.py`

**Interfaces:**
- Produces `evaluate_ragas(questions, answers, contexts, ground_truths) -> dict` and `run_query(query, search, reranker) -> tuple[str, list[str]]`.

- [ ] **Step 1: establish the failing baseline**

Run `python check_lab.py`. Confirm that it reports the six absent Day 18 files; do not change the checker.

- [ ] **Step 2: copy source files**

Run the following commands:

```bash
cp /home/long/project/K34-Day18-2A202601711-LeVanLong/src/m1_chunking.py src/
cp /home/long/project/K34-Day18-2A202601711-LeVanLong/src/m2_search.py src/
cp /home/long/project/K34-Day18-2A202601711-LeVanLong/src/m3_rerank.py src/
cp /home/long/project/K34-Day18-2A202601711-LeVanLong/src/m4_eval.py src/
cp /home/long/project/K34-Day18-2A202601711-LeVanLong/src/m5_enrichment.py src/
cp /home/long/project/K34-Day18-2A202601711-LeVanLong/src/pipeline.py src/
```

- [ ] **Step 3: add Day 18 configuration compatibility**

Keep every Day 24 path and collection setting. Add the Day 18 provider values: `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_MAX_RETRIES`, `LLM_TIMEOUT`, `ENRICH_WITH_LLM`, `MODEL_DEVICE`, and `get_llm_client()`.

```python
def get_llm_client():
    from openai import OpenAI
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
                  max_retries=LLM_MAX_RETRIES, timeout=LLM_TIMEOUT)
```

- [ ] **Step 4: verify imports**

Run `python -c "from src.m4_eval import evaluate_ragas; from src.pipeline import run_query; print('imports ok')"`; expect `imports ok`.

- [ ] **Step 5: commit the unit**

Run `git add config.py src/m1_chunking.py src/m2_search.py src/m3_rerank.py src/m4_eval.py src/m5_enrichment.py src/pipeline.py` then `git commit -m "feat: integrate Day 18 RAG pipeline"`.

### Task 2: Implement Phase A evaluation

**Files:**
- Modify: `src/phase_a_ragas.py`
- Test: `tests/test_phase_a.py`

**Interfaces:**
- Consumes `evaluate_ragas(...)["per_question"]`, whose entries are Day 18 `EvalResult` objects.
- Produces `group_by_distribution()`, `run_ragas_50q()`, `bottom_10()`, and `cluster_analysis()`.

- [ ] **Step 1: establish focused test failures**

Run `pytest tests/test_phase_a.py -v`; expect failures due to the empty group, ranking, and cluster returns.

- [ ] **Step 2: implement grouping, ranking, and failure matrix**

Return a dictionary initialized as `{"factual": [], "multi_hop": [], "adversarial": []}` and append every input by its distribution. Sort results by `avg_score`, select at most ten, and attach `diagnosis` and `suggested_fix` from `DIAGNOSTIC_TREE[result.worst_metric]`. Initialize all twelve metric/distribution cells to zero, increment each result's cell, calculate dominant row and column totals, and return a non-empty insight containing the chosen fix.

- [ ] **Step 3: implement RAGAS mapping**

Build the four arrays from every answer record, call the evaluator once, and map each returned `EvalResult` plus its aligned answer into `RagasResult`. When `per_question` is non-empty but its count differs from answers, raise `RuntimeError("RAGAS per_question count does not match answers")`.

- [ ] **Step 4: persist enriched bottom-ten rows**

Have `save_phase_a_report()` set `"bottom_10": bottom_10(results)` so the JSON includes all eight expected fields.

- [ ] **Step 5: verify and commit**

Run `pytest tests/test_phase_a.py -v`; expect all passing. Then run `git add src/phase_a_ragas.py && git commit -m "feat: add RAGAS distribution analysis"`.

### Task 3: Implement Phase B judging and agreement

**Files:**
- Modify: `src/phase_b_judge.py`
- Test: `tests/test_phase_b.py`

**Interfaces:**
- Produces `pairwise_judge(...) -> dict`, `swap_and_average(...) -> JudgeResult`, `cohen_kappa(labels, labels) -> float`, and `bias_report(results) -> dict`.

- [ ] **Step 1: establish focused test failures**

Run `pytest tests/test_phase_b.py -v`; expect the perfect-agreement κ test to fail.

- [ ] **Step 2: implement the validated pairwise call**

Call `OpenAI(api_key=OPENAI_API_KEY).chat.completions.create` with `JUDGE_MODEL` and JSON-object response format. Validate winner against `{"A", "B", "tie"}` and normalize scores with `min(1.0, max(0.0, float(value)))`. On API or schema failure, return a tie with non-empty diagnostic reasoning and two zero scores.

- [ ] **Step 3: implement swap-and-average**

Call pass one with `(A, B)`, then pass two with `(B, A)`. Convert the latter with `{"A": "B", "B": "A", "tie": "tie"}` and exchange its score keys. Set final winner only when converted pass-two winner equals pass one; otherwise final winner is `tie` and position consistency is false.

- [ ] **Step 4: implement κ and bias statistics**

Require equally sized non-empty binary lists. Calculate observed agreement, expected agreement, and `(observed - expected) / (1 - expected)`. Return `1.0` for identical labels when expected agreement is one; return `0.0` for other undefined degenerate cases. Compute position bias from inconsistent results and verbosity bias from decisive cases where the winning text is longer.

- [ ] **Step 5: emit the real judge report**

In the script entrypoint, judge all ten human-label entries against an empty baseline, label a model answer good only when it wins consensus, calculate κ and bias, and write `reports/judge_results.json` containing labels, results serialized by `dataclasses.asdict`, κ, and bias.

- [ ] **Step 6: verify and commit**

Run `pytest tests/test_phase_b.py -v`; expect all passing even without a usable API because API errors yield a structurally valid tie. Then run `git add src/phase_b_judge.py && git commit -m "feat: add swapped LLM judge evaluation"`.

### Task 4: Implement Phase C guardrails and latency

**Files:**
- Modify: `src/phase_c_guard.py`
- Test: `tests/test_phase_c.py`

**Interfaces:**
- Produces `pii_scan()`, async `check_input_rail()`, async `check_output_rail()`, `run_adversarial_suite()`, and `measure_p95_latency()`.

- [ ] **Step 1: establish focused test failures**

Run `pytest tests/test_phase_c.py -v`; expect CCCD/phone detection and suite-count assertions to fail.

- [ ] **Step 2: implement Presidio scanning**

Lazily create an analyzer/anonymizer if either was omitted. Call `analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)`. Each entity contains entity type, source substring, rounded score, start, and end. When results exist, anonymize with `anonymizer.anonymize`; otherwise return original text with an empty entity list.

- [ ] **Step 3: implement NeMo adapters**

Invoke `rails.generate_async` with user-only messages for input and both user/assistant messages for output. Treat `xin lỗi`, `không thể`, `không được phép`, `i cannot`, and `i'm sorry` case-insensitively as refusals. A flagged output returns the rail response; an unflagged output returns the original answer.

- [ ] **Step 4: implement one-loop suite and one-loop latency measurement**

Define an inner async coroutine for each batch function, loop inside it, and invoke `asyncio.run()` exactly once after it is defined. The suite calls `pii_scan` first, calls NeMo only if no PII is found, and produces `id`, `category`, `input`, `expected`, `actual`, `blocked_by`, and `passed`. Latency uses `time.perf_counter()` around each layer and a sorted, clamped percentile helper for `p50`, `p95`, and `p99`.

- [ ] **Step 5: write output and verify**

At script end, write `{"suite": results, "latency": latency}` to `reports/guard_results.json`. Run `pytest tests/test_phase_c.py -v`; expect all tests passing. Then run `git add src/phase_c_guard.py && git commit -m "feat: add production guardrail stack"`.

### Task 5: Generate reports, tune rails, and document results

**Files:**
- Modify: `guardrails/rails.co`, `reports/blueprint.md`, `analysis/failure_clusters.md`, `analysis/bias_report.md`
- Create: `answers_50q.json`, `reports/ragas_50q.json`, `reports/judge_results.json`, `reports/guard_results.json`
- Test: phase scripts and all phase tests

**Interfaces:**
- Consumes the completed phase modules, Qdrant, installed dependencies, spaCy language model, and secrets from untracked `.env`.
- Produces complete reports based on actual execution.

- [ ] **Step 1: generate and validate the answer input**

Run `curl --fail http://localhost:6333/readyz && python setup_answers.py`. Then run `python -c "import json; d=json.load(open('answers_50q.json')); assert len(d)==50; assert all({'id','distribution','question','answer','contexts','ground_truth'} <= x.keys() for x in d)"`.

- [ ] **Step 2: generate Phase A and B reports**

Run `python src/phase_a_ragas.py && python src/phase_b_judge.py`. From their JSON values, write `analysis/failure_clusters.md` with distribution metrics, all ten worst rows, and version-conflict analysis. Write `analysis/bias_report.md` with at least five pairwise rows, ten human/judge labels, κ interpretation, position bias, and verbosity bias.

- [ ] **Step 3: tune only failed live rail categories**

Run `python src/phase_c_guard.py`. If fewer than 15/20 pass, inspect the failed categories in its JSON and add those exact attack utterances to their relevant existing Colang input block. Rerun until at least 15/20 pass; do not weaken the normal HR flow.

- [ ] **Step 4: complete the blueprint from actual values**

Replace `[Họ Tên]`, the date, and every `?` in `reports/blueprint.md`. Use actual latency, RAGAS, κ, and suite numbers from the three JSON reports. Add a 3–5 sentence deployment assessment describing the bottleneck and monitoring response.

- [ ] **Step 5: verify outputs and commit**

Run `pytest tests/test_phase_a.py tests/test_phase_b.py tests/test_phase_c.py -v` and search the three phase files for remaining implementation-marker comments; expect all tests pass and no search results. Then add relevant rail/report/analysis files and commit with `git commit -m "docs: add Day 24 evaluation reports"`.

### Task 6: Run the final lab gate

**Files:**
- Verify: all source and report deliverables
- Test: `tests/`, `check_lab.py`

- [ ] **Step 1: full regression**

Run `pytest tests/ -v`; expect every collected test to pass.

- [ ] **Step 2: submission check**

Run `python check_lab.py`; expect every check to pass and its final output to include `Sẵn sàng nộp bài!`.

- [ ] **Step 3: inspect final state**

Run `git status --short && git diff --check HEAD`. Confirm no secrets are staged and whitespace validation has no output.

## Plan self-review

- Spec coverage: Tasks 1–4 cover pipeline compatibility and every required phase function; Tasks 5–6 cover reports, analysis, blueprint, and both final gates.
- Placeholder scan: every task includes a concrete change and a runnable verification command.
- Interface consistency: the RAGAS `per_question` contract, `JudgeResult`, guard suite records, and latency keys match their producers and consumers.
