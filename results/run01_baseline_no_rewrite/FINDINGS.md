# Benchmark Run 01 — Baseline (reflection loop *without* query reformulation)

**Date:** 2 August 2026
**Corpus:** 400 parallel EN/AR document pairs (800 PDFs) from the UN Parallel Corpus, indexed as 400 EN + 400 AR chunks in pgvector
**Evaluation set:** 65 items — 33 EN, 32 AR, 5 out-of-scope refusal probes
**Judge:** qwen2.5:7b-instruct (local, via Ollama) + nomic-embed-text; 260 evaluations per system
**Hardware:** NVIDIA RTX 5070, 12 GB VRAM — fully on-premise, no external API

---

## 1. Headline results

| Metric | Naive RAG | Agentic (Critic) | Δ (agentic − naive) |
|---|---|---|---|
| Faithfulness | 0.2853 | 0.2925 | +0.0072 |
| Answer Relevancy | 0.2432 | 0.2201 | **−0.0231** |
| Context Precision | 0.7800 | 0.7834 | +0.0034 |
| Context Recall | 0.8949 | 0.8949 | **0.0000** |
| Avg latency (s) | 7.61 | 10.36 | **+36 %** |

Out-of-scope refusals: naive **4/5**, agentic **3/5**.

**Conclusion:** in this configuration the agentic pipeline is statistically
indistinguishable from naive RAG on retrieval and answer quality, while costing
36 % more latency.

## 2. Per-language breakdown (in-scope only, n = 30 each)

| System | Lang | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|---|
| Agentic | EN | 0.2231 | 0.2977 | 0.8741 | 0.9333 |
| Agentic | AR | 0.3024 | 0.1792 | 0.8247 | 0.8722 |
| Naive | EN | 0.1762 | 0.3005 | 0.8657 | 0.9333 |
| Naive | AR | 0.3358 | 0.2141 | 0.8274 | 0.8722 |

Retrieval quality is consistently **lower for Arabic** (context recall 0.872 vs
0.933; precision 0.825 vs 0.870), quantifying the cross-lingual gap this project
set out to study.

## 3. Root cause of the null result — *the reflection loop is a no-op*

Context Recall is identical to four decimal places for both systems, in both
languages. Direct inspection confirms why:

> **65 / 65 questions retrieved byte-identical context sets** in the agentic and
> naive pipelines.

`retrieve_node` embeds `state["query"]` — the original, unmodified question — on
every pass. When the Critic judges the context insufficient and the LangGraph
conditional edge routes back to `retrieve`, the node re-executes the **same
vector search with the same query vector** and therefore returns the **same
passages**. The loop consumes an extra Critic + retrieval cycle and returns to an
identical state.

The generated answers do differ (61 / 65) — but that is generation-time sampling
variance, not better evidence.

**Finding:** *a Reflexion-style reflection loop cannot improve retrieval unless the
retry changes the retrieval input.* Verification alone detects a bad context; it
cannot repair one.

## 4. Implication and the fix

The loop needs a **query-reformulation step**: when the Critic rejects the
context, an LLM rewrites the query (term expansion, multi-hop decomposition,
entity/keyword emphasis) and the *rewritten* query drives the next retrieval.
This is implemented in `src/agents/nodes.py::rewrite_node` for Run 02.

Run 01 is retained as the **controlled baseline** that isolates the contribution
of query reformulation: Run 01 = verification only; Run 02 = verification +
reformulation.

## 5. Secondary observations

- **Absolute faithfulness is low (~0.29) for both systems.** The golden ground
  truths are deliberately terse (e.g. "On 29 July 1993."), which penalises longer
  generated answers under Ragas' statement-level scoring. Worth investigating
  separately; it affects both systems equally so the comparison stays valid.
- **Arabic answer relevancy trails English** (0.179 vs 0.298 agentic), consistent
  with the retrieval gap above.
- The 5-item out-of-scope sample is too small to draw conclusions from the 4/5
  vs 3/5 refusal difference.

## 6. Methodological notes (benchmark validity)

Two defects were found and fixed *before* this run; both had silently emptied
metrics in earlier attempts:

1. **Concurrency.** Ragas defaults to 16 concurrent workers, but a single local
   GPU serves one request at a time, so most requests queued past the 180 s
   timeout — 162 TimeoutErrors, leaving faithfulness and context_recall entirely
   unscored. Fixed with `max_workers=2`, `timeout=900`.
2. **Judge capability.** llama3:8b could not hold Ragas' NLI structured-output
   format (0/65 faithfulness scored) and ran at 61 s per evaluation. Replaced by
   qwen2.5:7b-instruct with JSON-constrained decoding: all four metrics populate
   at 2.6 s per evaluation.

Judge context is capped at the top 4 passages (faithfulness is
O(statements × contexts)), applied identically to both systems.
