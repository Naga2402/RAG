"""JISR web demo backend.

A thin FastAPI layer over the live pipeline: it serves the single-page UI at `/`
and exposes the agentic and naive RAG paths under `/api`. The whole thing runs
against the real stack (Ollama + pgvector), so the web page is a genuine live
demo, not a mock.

Run:
    uvicorn serve:app --reload --port 8000
Then open http://localhost:8000
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load torch/BGE-M3 before psycopg touches its native libs (Windows OpenMP order).
from src.indexing.embeddings import warmup

warmup()

from src.agents.graph import answer as agentic_answer  # noqa: E402
from src.baseline.naive_rag import naive_answer          # noqa: E402
from src.config import CFG, ROOT                          # noqa: E402

app = FastAPI(title="JISR — Agentic Cross-Lingual RAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WEB = ROOT / "web"


class AskRequest(BaseModel):
    query: str
    mode: str = "agentic"   # "agentic" | "naive"


def _fmt_contexts(contexts):
    out = []
    for c in contexts or []:
        out.append({
            "source": c.get("source", "?"),
            "chunk_idx": c.get("chunk_idx", c.get("index", 0)),
            "score": round(float(c.get("score", 0.0)), 3),
            "snippet": (c.get("content", "") or "").replace("\n", " ")[:220],
        })
    return out


@app.get("/api/health")
def health():
    info = {"ok": True, "models": CFG["inference"]["models"],
            "critic_model": CFG["inference"].get("critic_model"),
            "db": f"{CFG['vector_store']['pgvector']['host']}:{CFG['vector_store']['pgvector']['port']}"}
    return info


def _run(mode: str, query: str) -> dict:
    t0 = time.perf_counter()
    if mode == "naive":
        r = naive_answer(query)
        payload = {"mode": "naive", "lang": r["lang"], "loops": None,
                   "critic": None, "answer": r["answer"],
                   "contexts": _fmt_contexts(r["contexts"])}
    else:
        st = agentic_answer(query)
        payload = {"mode": "agentic", "lang": st.get("lang"),
                   "loops": st.get("loops"), "critic": (st.get("critique") or "").strip(),
                   "answer": st.get("answer", ""),
                   "contexts": _fmt_contexts(st.get("contexts"))}
    payload["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
    return payload


@app.post("/api/ask")
def ask(req: AskRequest):
    return _run(req.mode, req.query)


@app.post("/api/compare")
def compare(req: AskRequest):
    """Run BOTH pipelines on the same query so the UI can show the difference the
    agentic Critic makes. Agentic first (so the Arabic model stays warm for naive)."""
    agentic = _run("agentic", req.query)
    naive = _run("naive", req.query)
    return {"query": req.query, "agentic": agentic, "naive": naive}


@app.get("/api/golden")
def golden(n: int = 6):
    """Sample the REAL golden evaluation set so the UI benchmark runs against the
    same data as the dissertation benchmark (not hard-coded demo questions).
    Prefers short, checkable answers and keeps the EN/AR mix plus a refusal probe."""
    path = ROOT / CFG["eval"]["golden_set"]
    if not path.exists():
        return {"items": [], "error": "golden set not built yet"}
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    def short(r):
        return len(r.get("ground_truth", "")) < 60 and len(r.get("question", "")) < 110

    ins = [r for r in rows if r.get("type") == "in_scope" and short(r)]
    oos = [r for r in rows if r.get("type") == "out_of_scope"]
    rng = random.Random(13)
    en = [r for r in ins if r["lang"] == "en"]; rng.shuffle(en)
    ar = [r for r in ins if r["lang"] == "ar"]; rng.shuffle(ar)
    half = max(1, (n - 1) // 2)
    picked = en[:half] + ar[:half] + oos[:1]
    return {"total_golden": len(rows), "items": [
        {"id": r["id"], "lang": r["lang"], "type": r["type"],
         "question": r["question"], "ground_truth": r["ground_truth"],
         "source": r.get("source")} for r in picked]}


@app.get("/")
def index():
    return FileResponse(str(WEB / "index.html"))


if WEB.exists():
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
