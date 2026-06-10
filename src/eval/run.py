"""Ragas benchmark: Agentic (JISR) vs Naive RAG over the golden set.

Reports Faithfulness, Answer Relevancy, Context Precision, Context Recall for
both systems and writes a comparison table to results/.

Usage:
    python -m src.eval.run --golden data/golden/sample_golden.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.agents.graph import answer as agentic_answer
from src.baseline.naive_rag import naive_answer
from src.config import CFG, ROOT


def _load_golden(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _collect(system: str, golden: list[dict]) -> "Dataset":
    from datasets import Dataset
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for g in golden:
        if system == "agentic":
            st = agentic_answer(g["question"])
            ans, ctx = st.get("answer", ""), st.get("contexts", [])
        else:
            out = naive_answer(g["question"])
            ans, ctx = out["answer"], out["contexts"]
        rows["question"].append(g["question"])
        rows["answer"].append(ans)
        rows["contexts"].append([c["content"] for c in ctx])
        rows["ground_truth"].append(g["ground_truth"])
    return Dataset.from_dict(rows)


def _evaluate(dataset):
    from ragas import evaluate
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)
    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=CFG["eval"]["golden_set"])
    args = ap.parse_args()

    golden = _load_golden(ROOT / args.golden)
    results = {}
    for system in CFG["eval"]["baselines"]:
        print(f"\n=== Running {system} over {len(golden)} questions ===")
        scores = _evaluate(_collect(system, golden))
        results[system] = scores

    df = pd.DataFrame({s: dict(v) for s, v in results.items()}).T
    out = ROOT / CFG["eval"]["results_dir"] / "benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print("\n=== JISR vs Naive ===")
    print(df.to_string())
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
