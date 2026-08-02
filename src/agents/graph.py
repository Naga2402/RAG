r"""Assembles the JISR LangGraph: the agentic reflection loop.

    decompose -> route -> retrieve -> critic --(retry)--> rewrite -> retrieve
                                            \--(generate)--> generate -> END

The retry edge passes through `rewrite`, which reformulates the search query.
Looping straight back to `retrieve` would re-run the identical vector search and
return identical passages (measured in run 01: 65/65 identical context sets), so
verification alone produced no retrieval gain.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, StateGraph

from src.agents import nodes
from src.agents.state import AgentState


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("decompose", nodes.decompose_node)
    g.add_node("route", nodes.route_node)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("critic", nodes.critic_node)
    g.add_node("rewrite", nodes.rewrite_node)
    g.add_node("generate", nodes.generate_node)

    g.set_entry_point("decompose")
    g.add_edge("decompose", "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "critic")
    g.add_conditional_edges(
        "critic", nodes.should_retry,
        {"retry": "rewrite", "generate": "generate"},
    )
    g.add_edge("rewrite", "retrieve")   # reformulate, THEN search again
    g.add_edge("generate", END)
    return g.compile()


def answer(query: str) -> AgentState:
    return build_graph().invoke({"query": query})
