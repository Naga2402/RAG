"""Naive (one-shot) RAG baseline — the control group for the benchmark.

No router (always uses query language), no critic, no reflection loop:
retrieve once, stuff context, generate. This is exactly what JISR's agentic
pipeline is measured against with Ragas.
"""
from __future__ import annotations

from src.agents.router import route
from src.indexing import vector_store as vs
from src.indexing.embeddings import embed_query
from src.inference.llm import generate

_conn = None


def _db():
    global _conn
    if _conn is None:
        _conn = vs.connect()
    return _conn


_SYS = "Answer using only the provided context. Reply in the question's language."


def naive_answer(query: str) -> dict:
    lang = route(query)
    contexts = vs.hybrid_search(_db(), lang, embed_query(query))
    ctx = "\n---\n".join(c["content"] for c in contexts)
    ans = generate(f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:", lang=lang, system=_SYS)
    return {"query": query, "lang": lang, "answer": ans, "contexts": contexts}
