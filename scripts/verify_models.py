"""Confirm both local LLMs load on the RTX 5070 GPU and fit in 12 GB VRAM.

Loads each model one at a time (router uses one per query), runs a tiny bilingual
generation, then reports total size vs the portion resident in VRAM. 100% on GPU
means no CPU offload — the goal for the 12 GB constraint.
"""
from __future__ import annotations

import time

import ollama

MODELS = {
    "EN (Llama-3)": "llama3:8b",
    "AR (Jais-adapted-13b)": "hf.co/Solshine/jais-adapted-13b-chat-Q4_K_M-GGUF",
}
PROMPTS = {"EN (Llama-3)": "Reply with exactly: OK",
           "AR (Jais-adapted-13b)": "أجب بكلمة واحدة: نعم"}


def gb(n: int) -> float:
    return n / 1e9


for label, model in MODELS.items():
    t0 = time.time()
    resp = ollama.generate(model=model, prompt=PROMPTS[label],
                           options={"num_predict": 8, "temperature": 0})
    dt = time.time() - t0

    info = next((m for m in ollama.ps()["models"] if m["model"] == model), None)
    print(f"\n### {label}")
    print(f"  sample reply : {resp['response'].strip()[:40]!r}")
    print(f"  load+gen     : {dt:.1f}s")
    if info:
        total, vram = info["size"], info["size_vram"]
        pct = 100 * vram / total if total else 0
        where = "GPU (no CPU offload)" if pct >= 99 else f"{pct:.0f}% GPU / rest on CPU"
        print(f"  total size   : {gb(total):.2f} GB")
        print(f"  in VRAM      : {gb(vram):.2f} GB  -> {where}")
        print(f"  fits 12 GB   : {'YES' if gb(vram) < 12 and pct >= 99 else 'CHECK'}")
    # free VRAM before loading the next model
    ollama.generate(model=model, prompt="", keep_alive=0)

print("\nNote: models load one at a time (router picks per query), so peak VRAM is "
      "a single model, not the sum.")
