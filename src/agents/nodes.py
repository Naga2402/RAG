"""LangGraph nodes for the JISR agentic loop.

Flow: decompose -> route -> retrieve -> critic -(loop)-> generate
The Critic node is the heart of the self-correction (Reflexion-style): it judges
whether retrieved context is sufficient, and if not, forces another retrieval
pass (bounded by max_reflection_loops) before the model is allowed to answer.
"""
from __future__ import annotations

import re

from src.agents.router import route
from src.agents.state import AgentState
from src.config import CFG
from src.indexing import vector_store as vs
from src.indexing.embeddings import embed_query
from src.inference.llm import generate

_A = CFG["agent"]
_CRITIC_MODEL = CFG["inference"].get("critic_model")  # one judge for both languages

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
    state["search_query"] = q    # what retrieval actually uses; rewritten on retry
    return state


def route_node(state: AgentState) -> AgentState:
    state["lang"] = route(state["query"])
    return state


def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve using `search_query`, which the rewrite node updates between
    reflection loops. Using the original query here would make every retry
    return identical passages (measured: 65/65 identical in run 01)."""
    lang = state["lang"]
    emb = embed_query(state.get("search_query") or state["query"])
    state["contexts"] = vs.hybrid_search(_db(), lang, emb)
    return state


_REWRITE_SYS = {
    "en": ("You rewrite search queries for a document retrieval system. The previous "
           "search did not find enough information. Rewrite the question as a better "
           "search query: keep the key entities, add likely synonyms and domain terms, "
           "and drop conversational filler. Reply with ONLY the rewritten query."),
    "ar": ("أنت تعيد صياغة استعلامات البحث لنظام استرجاع المستندات. لم يعثر البحث السابق "
           "على معلومات كافية. أعد صياغة السؤال كاستعلام بحث أفضل: احتفظ بالكيانات الأساسية، "
           "وأضف المرادفات والمصطلحات المتخصصة المحتملة. أجب بالاستعلام الجديد فقط."),
}


def rewrite_node(state: AgentState) -> AgentState:
    """Reformulate the query after the Critic rejects the retrieved context.

    This is what makes the reflection loop meaningful: verification alone can
    detect a bad context but cannot repair one — the retry must change the
    retrieval input.
    """
    lang = state.get("lang", "en")
    prev = state.get("search_query") or state["query"]
    prompt = (f"Original question: {state['query']}\n"
              f"Previous search query: {prev}\n"
              f"Critic feedback: context was insufficient.\n\n"
              f"Rewritten search query:")
    try:
        new_q = generate(prompt, lang=lang, system=_REWRITE_SYS.get(lang, _REWRITE_SYS["en"]),
                         model=_CRITIC_MODEL).strip().strip('"').split("\n")[0]
    except Exception:  # noqa: BLE001
        new_q = ""
    # Strip template placeholders like "[industry/sector]" that the model
    # sometimes emits, and any leading label it prefixes.
    new_q = re.sub(r"\[[^\]]*\]", " ", new_q)
    new_q = re.sub(r"^\s*(rewritten\s+)?(search\s+)?query\s*:\s*", "", new_q, flags=re.I)
    new_q = re.sub(r"\s{2,}", " ", new_q).strip()
    # Fall back to the previous query if the rewrite is empty or degenerate.
    state["search_query"] = new_q if len(new_q) >= 8 else prev
    state["rewrites"] = state.get("rewrites", []) + [state["search_query"]]
    return state


# A single-word YES/NO verdict is far more robust than JSON across both models
# (Jais in particular is unreliable at structured JSON) and both languages.
_CRITIC_SYS = (
    "You are a strict retrieval critic. Decide whether the provided context "
    "contains enough information to answer the question faithfully. "
    "Reply with exactly one word: YES if it is sufficient, or NO if it is not. "
    "أجب بكلمة واحدة فقط: YES أو NO."
)


def _verdict_relevant(raw: str) -> bool | None:
    """Parse a YES/NO verdict (EN or AR). None means unparseable -> fail open."""
    t = raw.strip().lower()
    if t.startswith(("yes", "نعم")) or t.split()[:1] == ["yes"]:
        return True
    if t.startswith(("no", "لا", "كلا")):
        return False
    if "نعم" in t and "لا" not in t:
        return True
    if "yes" in t and "no" not in t:
        return True
    if "no" in t.split() or "لا" in t.split():
        return False
    return None


def critic_node(state: AgentState) -> AgentState:
    ctx = "\n---\n".join(c["content"] for c in state.get("contexts", []))
    prompt = (f"Question: {state['query']}\n\nContext:\n{ctx}\n\n"
              f"Is the context sufficient? Answer YES or NO:")
    raw = generate(prompt, lang=state["lang"], system=_CRITIC_SYS,
                   model=_CRITIC_MODEL)
    verdict = _verdict_relevant(raw)
    state["relevant"] = True if verdict is None else verdict  # fail open
    state["critique"] = raw.strip()[:200] or "(empty verdict; proceeding)"
    state["loops"] = state.get("loops", 0) + 1
    return state


def should_retry(state: AgentState) -> str:
    """Conditional edge: loop back to retrieve, or move on to generate."""
    if not state.get("relevant") and state["loops"] < _A["max_reflection_loops"]:
        return "retry"
    return "generate"


# Language-specific answer instructions. A bare "reply in the question's language"
# hint is not reliable for Jais, so the Arabic prompt is written in Arabic and
# explicitly demands an Arabic answer.
_ANSWER_SYS = {
    "en": ("Answer the question using ONLY the provided context, and cite the "
           "source in brackets. If the context is insufficient, say so plainly. "
           "Answer in English."),
    "ar": ("أجب عن السؤال بالاعتماد فقط على السياق المُعطى، واذكر المصدر بين قوسين. "
           "إذا كان السياق غير كافٍ فاذكر ذلك بوضوح. يجب أن تكون الإجابة باللغة العربية."),
}


def generate_node(state: AgentState) -> AgentState:
    ctx = "\n---\n".join(
        f"[{c['source']}#{c['chunk_idx']}] {c['content']}" for c in state.get("contexts", [])
    )
    lang = state["lang"]
    prompt = f"Context:\n{ctx}\n\nQuestion: {state['query']}\n\nAnswer:"
    state["answer"] = generate(prompt, lang=lang, system=_ANSWER_SYS[lang])
    return state
