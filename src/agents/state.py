"""Shared state passed between LangGraph nodes."""
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    query: str              # original user query
    lang: str               # 'en' | 'ar' (routing decision)
    sub_queries: list[str]  # from decomposition node
    contexts: list[dict]    # retrieved chunks
    relevant: bool          # critic verdict
    critique: str           # critic reasoning (for transparency / logs)
    loops: int              # reflection loop counter
    answer: str             # final generated answer
