"""Local quantized inference via Ollama. Routes Arabic prompts to Jais-13B and
English to Llama-3, both INT4/GGUF so they fit the 12 GB RTX 5070."""
from __future__ import annotations

import ollama

from src.config import CFG

_I = CFG["inference"]


def _model_for(lang: str) -> str:
    return _I["models"].get(lang, _I["models"]["en"])


def generate(prompt: str, lang: str = "en", system: str | None = None,
             model: str | None = None) -> str:
    """Generate a completion. `model` overrides language routing (used by the
    Critic node, which always judges with the configured critic model)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(
        model=model or _model_for(lang),
        messages=messages,
        options={
            "temperature": _I["temperature"],
            "num_ctx": _I["context_window"],
        },
    )
    return resp["message"]["content"].strip()
