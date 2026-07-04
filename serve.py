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


@app.post("/api/ask")
def ask(req: AskRequest):
    t0 = time.perf_counter()
    if req.mode == "naive":
        r = naive_answer(req.query)
        payload = {"mode": "naive", "lang": r["lang"], "loops": None,
                   "critic": None, "answer": r["answer"],
                   "contexts": _fmt_contexts(r["contexts"])}
    else:
        st = agentic_answer(req.query)
        payload = {"mode": "agentic", "lang": st.get("lang"),
                   "loops": st.get("loops"), "critic": (st.get("critique") or "").strip(),
                   "answer": st.get("answer", ""),
                   "contexts": _fmt_contexts(st.get("contexts"))}
    payload["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
    return payload


@app.get("/")
def index():
    return FileResponse(str(WEB / "index.html"))


if WEB.exists():
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
