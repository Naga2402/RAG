"""LangGraph nodes for the JISR agentic loop.

Flow: decompose -> route -> retrieve -> critic -(loop)-> generate
The Critic node is the heart of the self-correction (Reflexion-style): it judges
whether retrieved context is sufficient, and if not, forces another retrieval
pass (bounded by max_reflection_loops) before the model is allowed to answer.
"""
from __future__ import annotations

import json

from src.agents.router import route
from src.agents.state import AgentState
from src.config import CFG
from src.indexing import vector_store as vs
from src.indexing.embeddings import embed_query
from src.inference.llm import generate

_A = CFG["agent"]

# Lazy single connection for the graph run.
_conn = None


def _db():
    global _conn
    if _conn is None:
        _conn = vs.connect()
    return _conn


def decompose_node(state: AgentState) -> AgentState:
    """Break a complex query into sub-queries (no-op for simple ones)."""
    q = state["query"]
    state["sub_queries"] = [q]   # TODO: LLM-driven decomposition for multi-hop
    state.setdefault("loops", 0)
    return state


def route_node(state: AgentState) -> AgentState:
    state["lang"] = route(state["query"])
    return state


def retrieve_node(state: AgentState) -> AgentState:
    lang = state["lang"]
    emb = embed_query(state["query"])
    state["contexts"] = vs.hybrid_search(_db(), lang, emb)
    return state


_CRITIC_SYS = (
    "You are a strict retrieval critic. Given a question and retrieved context, "
    "answer ONLY with JSON: {\"relevant\": true|false, \"reason\": \"...\"}. "
    "Mark relevant=false if the context does not contain enough information to "
    "answer faithfully."
)


def critic_node(state: AgentState) -> AgentState:
    ctx = "\n---\n".join(c["content"] for c in state.get("contexts", []))
    prompt = f"Question: {state['query']}\n\nContext:\n{ctx}\n\nVerdict JSON:"
    raw = generate(prompt, lang=state["lang"], system=_CRITIC_SYS)
    try:
        verdict = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        state["relevant"] = bool(verdict.get("relevant"))
        state["critique"] = verdict.get("reason", "")
    except Exception:
        state["relevant"] = True   # fail-open to avoid infinite loops
        state["critique"] = "unparseable critic output; proceeding"
    state["loops"] = state.get("loops", 0) + 1
    return state


def should_retry(state: AgentState) -> str:
    """Conditional edge: loop back to retrieve, or move on to generate."""
    if not state.get("relevant") and state["loops"] < _A["max_reflection_loops"]:
        return "retry"
    return "generate"


_ANSWER_SYS = (
    "Answer the question using ONLY the provided context. Cite the source. "
    "If the context is insufficient, say so plainly. Reply in the question's language."
)


def generate_node(state: AgentState) -> AgentState:
    ctx = "\n---\n".join(
        f"[{c['source']}#{c['chunk_idx']}] {c['content']}" for c in state.get("contexts", [])
    )
    prompt = f"Context:\n{ctx}\n\nQuestion: {state['query']}\n\nAnswer:"
    state["answer"] = generate(prompt, lang=state["lang"], system=_ANSWER_SYS)
    return state
