"""Compatibility shim for ragas on the langchain 1.x stack.

ragas 0.4.3 unconditionally does, in ragas/llms/base.py:

    from langchain_community.chat_models.vertexai import ChatVertexAI

That submodule was removed in langchain-community >= 0.3, but JISR must run
langchain 1.x (required by LangGraph 1.x). Downgrading langchain is not an option
on Python 3.13 either, because langchain 0.2.x pins numpy < 2.0, which has no
3.13 wheels. JISR never uses Vertex AI, so we register a stub module exposing a
placeholder ChatVertexAI *before* ragas is imported.

Import this module first in anything that imports ragas (see src/eval/run.py).
"""
from __future__ import annotations

import sys
import types

_MOD = "langchain_community.chat_models.vertexai"


def install() -> None:
    """Register the stub only if the real module is genuinely absent."""
    if _MOD in sys.modules:
        return
    try:
        __import__(_MOD)
        return  # real module present (older langchain) — leave it alone
    except ModuleNotFoundError:
        pass

    stub = types.ModuleType(_MOD)

    class ChatVertexAI:  # noqa: D401 - placeholder, never instantiated in JISR
        """Stub: Vertex AI is not used by JISR. Present only so ragas imports."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ChatVertexAI is a compatibility stub and is not available in JISR."
            )

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[_MOD] = stub


install()
