# JISR — Development Plan & Milestone Roadmap

Aligned to the BITS dissertation timeline (16 weeks, May–Aug). Current position: ~Week 4.

| Milestone | Window | Deliverable | Status |
|-----------|--------|-------------|--------|
| **M0 — Setup** | Wk 1–2 | Repo scaffold, Ollama + pgvector running, golden dataset started | 🟡 in progress |
| **M1 — Ingestion** | Wk 3–5 | OCR (ara+eng) + layout chunker + language tagging → `data/processed` | ⬜ |
| **M2 — Indexing** | Wk 6–8 | BGE-M3 + BM25 hybrid index in pgvector; separate EN/AR collections | 🟡 working on samples |
| **M3 — Agentic core** | Wk 9–10 | LangGraph reflection loop + semantic router; naive baseline in parallel | 🟡 loop works EN+AR (Llama-3 critic); naive baseline next |
| **M4 — Benchmark** | Wk 11–13 | Ragas: Agentic vs Naive; VRAM/latency profiling on RTX 5070 | ⬜ |
| **M5 — Report** | Wk 14–16 | Dissertation chapters, formatting, plagiarism check, final submission | ⬜ |

## Critical path
The **golden dataset** (`data/golden/`) gates everything measurable. Build 50 EN + 50 AR
question/ground-truth/source triples early, drawn from the real company docs.

## Immediate next steps
1. [ ] `pip install -r requirements.txt` in a fresh venv
2. [ ] Install Tesseract with Arabic (`ara`) language pack + Poppler (for `pdf2image`)
3. [ ] Start Postgres, enable `CREATE EXTENSION vector;`, create `jisr` DB
4. [ ] `ollama pull jais:13b` and `ollama pull llama3:8b`; confirm both run under 12 GB
5. [ ] Drop company docs into `data/raw/`, run `python -m src.ingestion.run`
6. [ ] Expand `data/golden/sample_golden.jsonl` to 100 entries

## Design decisions (locked)
- Vector store: **pgvector** (production/enterprise story)
- Embeddings: **BGE-M3** (one model for EN+AR, dense+sparse)
- Router: **per-language indices** so routing is a real retrieval decision, not just a prompt
- Critic loop capped at `max_reflection_loops` to bound latency for fair benchmarking

## Environment status (run `python scripts/check_env.py`)
As of first setup on the RTX 5070 machine:
- ✅ Python deps installed in `.venv`; smoke tests pass (4/4)
- ✅ Ingestion verified on `data/samples` → 14 chunks, 7 EN / 7 AR
- ✅ Postgres reachable on `localhost:5432`
- ✅ **torch CUDA 12.8** installed (`2.11.0+cu128`); `torch.cuda` sees the
      **RTX 5070 (12226 MB)**
- ✅ **Ollama** + models: `llama3:8b` and Jais-adapted-13b (HF GGUF) pulled
- ✅ **Dockerized pgvector** (pgvector 0.8.2, PG17) on 5433; healthy
- ✅ **Hybrid index built** from samples: 7 EN + 7 AR chunks; bilingual hybrid
      retrieval verified end-to-end (`scripts/verify_index.py`)
- ⚠️ **Windows OpenMP**: torch must load before psycopg or the process segfaults
      (duplicate libiomp5md.dll). Indexing calls `embeddings.warmup()` first; the
      agent path is naturally ordered (embed before DB).
- ✅ **ragas imports** on the langchain 1.x stack via `src/eval/_compat.py` shim
      (stubs the one removed `langchain_community.chat_models.vertexai` module;
      JISR never uses Vertex AI). A separate 0.2-era venv was rejected: langchain
      0.2.x pins numpy < 2.0, which has no Python 3.13 wheels.
- ⚠️ Ollama not running — install from ollama.com, then `ollama pull jais:13b llama3:8b`
- ⚠️ Tesseract (+`ara`) and Poppler not on PATH — only needed for PDF/image OCR

## Local LLM VRAM (RTX 5070, 12 GB) — measured via scripts/verify_models.py
| Model | Quant | Total @4096 ctx | GPU placement |
|-------|-------|-----------------|---------------|
| Llama-3 8B (`llama3:8b`) | Q4 | 5.01 GB | **100% GPU** (no offload) |
| Jais-adapted-13B chat (HF GGUF) | Q4_K_M | 11 GB | **88% GPU / 12% CPU** |

Jais at 4096 context slightly exceeds the usable VRAM budget (Ollama reserves
~1 GB headroom), so 12% offloads to CPU. This is a genuine VRAM-constraint result
for the thesis. Levers to get Jais fully on GPU: lower `context_window` (e.g. 2048),
or a smaller quant (Q4_K_S / Q3_K_M).

## M3 status — agentic loop runs end-to-end (EN + AR)
- ✅ Jais chat template fixed: `models/Modelfile.jais` -> `jais-chat` (config `ar`).
      Arabic answers are now fluent, grounded, and cited.
- ✅ Router → hybrid retrieve → critic → generate works for both languages.
- ✅ Critic switched from JSON to a YES/NO verdict (robust for Jais).
- ✅ Generation system prompt is language-specific (Arabic prompt enforces Arabic).
- ✅ **Critic = Llama-3 for both languages** (config `inference.critic_model`).
      Now judges correctly: relevant AR context -> YES in 1 pass (no wasted loops);
      out-of-scope EN query -> NO -> faithful refusal instead of hallucinating.
      Trade-off (documented): an Arabic query swaps Llama-3 (critic) -> Jais (generate).

## Open items to revisit
- [ ] Measure the critic model-swap latency cost on Arabic queries (VRAM/latency chapter)
- [ ] Tune Jais `context_window`/quant to remove the 12% CPU offload (VRAM chapter)
- [ ] Tesseract (+ara) and Poppler are installed; validate Arabic OCR on a real
      scanned PDF when the company docs arrive
- [ ] Decide reranker on/off for the final benchmark (report both)
