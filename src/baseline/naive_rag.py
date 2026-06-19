"""Naive (one-shot) RAG baseline — the control group for the benchmark.

No router (always uses query language), no critic, no reflection loop:
retrieve once, stuff context, generate. This is exactly what JISR's agentic
pipeline is measured against with Ragas.
"""
from __future__ import annotations

from src.agents.nodes import _ANSWER_SYS  # same generation prompt as the agent
from src.agents.router import route
from src.indexing import vector_store as vs
from src.indexing.embeddings import embed_query, warmup
from src.inference.llm import generate

_conn = None


def _db():
    global _conn
    if _conn is None:
        warmup()           # load torch/BGE-M3 before psycopg (OpenMP ordering)
        _conn = vs.connect()
    return _conn


def naive_answer(query: str) -> dict:
    lang = route(query)
    emb = embed_query(query)           # embed BEFORE connecting (OpenMP ordering)
    contexts = vs.hybrid_search(_db(), lang, emb)
    ctx = "\n---\n".join(c["content"] for c in contexts)
    # Identical generation to the agent so the ONLY difference is the critic loop.
    ans = generate(f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:",
                   lang=lang, system=_ANSWER_SYS[lang])
    return {"query": query, "lang": lang, "answer": ans, "contexts": contexts}
