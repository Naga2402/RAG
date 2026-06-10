"""Assembles the JISR LangGraph: the agentic reflection loop.

    decompose -> route -> retrieve -> critic --(retry)--> retrieve
                                            \--(generate)--> generate -> END
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
    g.add_node("generate", nodes.generate_node)

    g.set_entry_point("decompose")
    g.add_edge("decompose", "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "critic")
    g.add_conditional_edges(
        "critic", nodes.should_retry,
        {"retry": "retrieve", "generate": "generate"},
    )
    g.add_edge("generate", END)
    return g.compile()


def answer(query: str) -> AgentState:
    return build_graph().invoke({"query": query})
