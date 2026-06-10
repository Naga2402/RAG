# JISR — جسر
### Jais-Integrated Self-correcting Retrieval
**An Agentic Multi-Lingual (Arabic ↔ English) RAG Framework for Enterprise Document Management Systems**

> M.Tech Dissertation — VN Sanjay (2024AA05123), BITS Pilani WILP AIML
> Carried out at VICISOFT Technologies Pvt Ltd, Hyderabad

---

## What this is
JISR is a **self-correcting, cross-lingual RAG** layer for enterprise DMS. It moves beyond
"naive" one-shot RAG by using a **LangGraph agentic loop** with a Critic/Reflection node that
verifies retrieved context *before* answering, and a **semantic router** that switches between
English and Arabic vector indices to cut translation error and latency. All inference runs
**locally** on a 12 GB RTX 5070 using INT4/GGUF quantized bilingual LLMs (Jais-13B, Llama-3).

## Pillars
1. **Agentic Reflection Loop** — self-correcting Critic node (LangGraph)
2. **Semantic Cross-Lingual Router** — EN/AR index routing by intent
3. **Local Bilingual Inference** — quantized Jais-13B / Llama-3 in 12 GB VRAM
4. **Comparative Benchmark** — Agentic RAG vs Naive RAG via Ragas

## Architecture
```
Query (EN/AR) ─► [LangGraph Orchestrator]
                  1. Decompose ─► 2. Route ─► 3. Retrieve (hybrid) ─►
                  4. CRITIC (relevant? loop back if weak) ─► 5. Generate
                       │                 │                        │
                 EN/AR indices    pgvector + BGE-M3 + BM25   Jais-13B / Llama-3 (GGUF)
                                          │
                                  Ragas eval: Agentic vs Naive
```

## Repository
```bash
git clone https://github.com/Naga2402/RAG.git
cd RAG
```

## Quick start
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 0. Try the pipeline immediately on the bundled bilingual samples
#    (data/samples/ — no OCR binaries needed, plain-text fixtures)
python -m src.ingestion.run --input data/samples
python scripts/inspect_chunks.py                    # sanity-check language tagging

# 1. Start Postgres + pgvector and Ollama (see config/settings.yaml)
ollama pull jais:13b          # or a GGUF via llama.cpp
# 2. Ingest your real documents (place them in data/raw/, which is gitignored)
python -m src.ingestion.run --input data/raw
# 3. Build index
python -m src.indexing.build
# 4. Ask
python -m src.agents.run --query "ما هي شروط الدفع في العقد؟"
# 5. Benchmark (Agentic vs Naive)
python -m src.eval.run --golden data/golden/sample_golden.jsonl
```

## Environment doctor
```bash
python scripts/check_env.py   # verifies deps, GPU/CUDA, OCR binaries, Postgres, Ollama
```
For the RTX 5070 (Blackwell), install the CUDA 12.8 torch build, not the default CPU wheel:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
> Note: `src/eval/_compat.py` ships a small shim so ragas (which expects an older
> langchain) imports cleanly on the langchain 1.x stack used by LangGraph. No
> separate environment needed.

## Sample data
`data/samples/` ships 6 committed bilingual fixtures (EN+AR invoice, service
agreement, and pump manual) with **parallel content**, so cross-lingual retrieval
and the EN/AR router can be validated before real documents arrive. Real corpora
go in `data/raw/` (gitignored — never committed).

## Repo layout
| Path | Purpose |
|------|---------|
| `src/ingestion/` | OCR (ara+eng), layout-aware chunking, language detection |
| `src/indexing/`  | BGE-M3 embeddings + BM25 hybrid index on pgvector |
| `src/agents/`    | LangGraph nodes: router, retriever, critic, generator |
| `src/inference/` | Local quantized LLM loading |
| `src/baseline/`  | Naive RAG pipeline (comparison baseline) |
| `src/eval/`      | Ragas harness + VRAM/latency profiling |
| `data/golden/`   | Golden Q&A eval set (critical path) |
| `results/`       | Benchmark tables/charts for the dissertation |

## Status
See `PLAN.md` for the milestone roadmap aligned to the BITS submission timeline.
