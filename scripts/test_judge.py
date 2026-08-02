# -*- coding: utf-8 -*-
"""Isolate the Ragas judge: can the configured local model actually score all
four metrics? Uses tiny synthetic samples, so it takes ~1 minute instead of
re-running the whole pipeline.

    python scripts/test_judge.py                    # uses eval.judge_model
    python scripts/test_judge.py --model llama3:8b  # compare judges
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.eval import _compat  # noqa: F401  (shim before ragas)

from src.config import CFG

SAMPLES = {
    "question": [
        "When did the ONUSAL Human Rights Division sign a cooperation agreement?",
        "What is the total estimated requirement for the programme?",
    ],
    "answer": [
        "The Division signed the cooperation agreement on 29 July 1993.",
        "The total estimated requirement is $7,291,700.",
    ],
    "contexts": [
        ["The ONUSAL Human Rights Division signed a cooperation agreement with the "
         "Office of the Counsel for the Defence of Human Rights on 29 July 1993, "
         "establishing joint monitoring arrangements."],
        ["The Secretary-General reports that the total estimated requirements arising "
         "out of resolutions and decisions adopted amount to $7,291,700 for the "
         "biennium."],
    ],
    "ground_truth": ["On 29 July 1993.", "$7,291,700."],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    from datasets import Dataset
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)
    from ragas.run_config import RunConfig

    model = args.model or CFG["eval"].get("judge_model") or "llama3:8b"
    print(f"Judge model: {model}")

    llm = LangchainLLMWrapper(ChatOllama(model=model, temperature=0.0, format="json",
                                         num_ctx=CFG["inference"]["context_window"]))
    emb = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=CFG["eval"].get("judge_embeddings", "nomic-embed-text")))

    t0 = time.perf_counter()
    res = evaluate(
        Dataset.from_dict(SAMPLES),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm, embeddings=emb, raise_exceptions=False,
        run_config=RunConfig(timeout=600, max_workers=2, max_retries=3),
    )
    dt = time.perf_counter() - t0

    import pandas as pd
    df = res.to_pandas()
    print(f"\nScored {len(df)} samples in {dt:.0f}s  ({dt/ (len(df)*4):.1f}s per evaluation)\n")
    ok = True
    for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        if m in df.columns:
            s = pd.to_numeric(df[m], errors="coerce")
            filled = int(s.notna().sum())
            status = "OK " if filled == len(df) else "FAIL"
            if filled != len(df):
                ok = False
            print(f"  [{status}] {m:<20} {filled}/{len(df)} filled   mean={s.mean():.3f}")
        else:
            ok = False
            print(f"  [FAIL] {m:<20} column missing")
    print("\nVERDICT:", "all four metrics usable ✅" if ok else "some metrics unusable ❌")


if __name__ == "__main__":
    main()
