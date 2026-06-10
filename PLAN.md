# JISR — Development Plan & Milestone Roadmap

Aligned to the BITS dissertation timeline (16 weeks, May–Aug). Current position: ~Week 4.

| Milestone | Window | Deliverable | Status |
|-----------|--------|-------------|--------|
| **M0 — Setup** | Wk 1–2 | Repo scaffold, Ollama + pgvector running, golden dataset started | 🟡 in progress |
| **M1 — Ingestion** | Wk 3–5 | OCR (ara+eng) + layout chunker + language tagging → `data/processed` | ⬜ |
| **M2 — Indexing** | Wk 6–8 | BGE-M3 + BM25 hybrid index in pgvector; separate EN/AR collections | ⬜ |
| **M3 — Agentic core** | Wk 9–10 | LangGraph reflection loop + semantic router; naive baseline in parallel | ⬜ |
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

## Open items to revisit
- [ ] Confirm Jais-13B q4_K_M actually fits alongside embeddings on 12 GB (else Jais-7B/offload)
- [ ] Decide reranker on/off for the final benchmark (report both)
