"""Ragas benchmark: Agentic (JISR) vs Naive RAG over the golden set.

Everything runs LOCALLY — the Ragas judge itself is a local Ollama model, so no
data leaves the machine and there is no API cost, consistent with the project's
on-premise premise.

Reports Faithfulness, Answer Relevancy, Context Precision and Context Recall for
both systems, plus per-question latency, and writes results/benchmark.csv.

Usage:
    python -m src.eval.run                     # full golden set
    python -m src.eval.run --limit 20          # quick validation run
    python -m src.eval.run --lang ar           # single language
"""
from __future__ import annotations

from src.eval import _compat  # noqa: F401  -- installs ragas shim BEFORE ragas import

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.config import CFG, ROOT
from src.indexing.embeddings import warmup

GOLDEN_DEFAULT = "data/golden/golden.jsonl"


def _load_golden(path: Path, limit: int | None, lang: str | None) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if lang:
        rows = [r for r in rows if r.get("lang") == lang]
    if limit:
        # keep the language mix and always retain the out-of-scope probes
        oos = [r for r in rows if r.get("type") == "out_of_scope"]
        ins = [r for r in rows if r.get("type") != "out_of_scope"]
        en = [r for r in ins if r["lang"] == "en"][: max(1, limit // 2)]
        ar = [r for r in ins if r["lang"] == "ar"][: max(1, limit // 2)]
        rows = en + ar + oos
    # Group by language. Ollama holds ONE model resident, so alternating EN/AR
    # forces a 5-11 GB VRAM swap per question; grouping makes each model load once.
    rows.sort(key=lambda r: (r.get("lang") != "en", r.get("type") == "out_of_scope"))
    return rows


def _local_judge():
    """Ragas judge + embeddings, both served locally by Ollama."""
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    judge = CFG["inference"].get("critic_model") or CFG["inference"]["models"]["en"]
    llm = ChatOllama(model=judge, temperature=0.0,
                     num_ctx=CFG["inference"]["context_window"])
    emb = OllamaEmbeddings(model=CFG["eval"].get("judge_embeddings", "nomic-embed-text"))
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb), judge


def _collect(system: str, golden: list[dict]) -> tuple[dict, list[float]]:
    """Run one pipeline over the golden set, returning Ragas columns + latencies."""
    from src.agents.graph import answer as agentic_answer
    from src.baseline.naive_rag import naive_answer

    cols = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    lats: list[float] = []
    for i, g in enumerate(golden, 1):
        t0 = time.perf_counter()
        try:
            if system == "agentic":
                st = agentic_answer(g["question"])
                ans, ctx = st.get("answer", ""), st.get("contexts", []) or []
            else:
                out = naive_answer(g["question"])
                ans, ctx = out["answer"], out["contexts"] or []
        except Exception as e:  # noqa: BLE001
            print(f"    ! {g['id']}: {e}")
            ans, ctx = "", []
        lats.append(time.perf_counter() - t0)
        cols["question"].append(g["question"])
        cols["answer"].append(ans)
        cols["contexts"].append([c.get("content", "") for c in ctx])
        cols["ground_truth"].append(g["ground_truth"])
        if i % 10 == 0 or i == len(golden):
            print(f"    {system}: {i}/{len(golden)}  (avg {sum(lats)/len(lats):.1f}s)")
    return cols, lats


def _evaluate(cols: dict, llm, emb):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)

    ds = Dataset.from_dict(cols)
    return evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm, embeddings=emb, raise_exceptions=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=GOLDEN_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="subset size for a quick run")
    ap.add_argument("--lang", choices=["en", "ar"], default=None)
    ap.add_argument("--skip-ragas", action="store_true",
                    help="only run the pipelines and record latency/answers")
    args = ap.parse_args()

    warmup()  # torch before psycopg (Windows OpenMP ordering)
    golden = _load_golden(ROOT / args.golden, args.limit, args.lang)
    print(f"Golden items: {len(golden)}  "
          f"(EN {sum(1 for g in golden if g['lang']=='en')}, "
          f"AR {sum(1 for g in golden if g['lang']=='ar')}, "
          f"out-of-scope {sum(1 for g in golden if g.get('type')=='out_of_scope')})")

    runs = {}
    for system in CFG["eval"]["baselines"]:
        print(f"\n=== Running {system} ===")
        cols, lats = _collect(system, golden)
        runs[system] = (cols, lats)

    out_dir = ROOT / CFG["eval"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # persist raw answers for inspection / the dissertation appendix
    with open(out_dir / "raw_answers.json", "w", encoding="utf-8") as fh:
        json.dump({s: {"cols": c, "latencies": l} for s, (c, l) in runs.items()},
                  fh, ensure_ascii=False, indent=1)

    rows = {}
    if not args.skip_ragas:
        llm, emb, judge = _local_judge()
        print(f"\nRagas judge (local): {judge}")
        for system, (cols, lats) in runs.items():
            print(f"  scoring {system}…")
            score = _evaluate(cols, llm, emb)
            # ragas 0.4.x: read per-sample scores via to_pandas() and average the
            # metric columns (the result object is not dict()-convertible).
            sdf = score.to_pandas()
            d = {}
            for m in ("faithfulness", "answer_relevancy",
                      "context_precision", "context_recall"):
                if m in sdf.columns:
                    d[m] = round(float(sdf[m].astype(float).mean(skipna=True)), 4)
            d["avg_latency_s"] = round(sum(lats) / len(lats), 2)
            rows[system] = d
            sdf.to_csv(out_dir / f"per_question_{system}.csv", index=False,
                       encoding="utf-8-sig")
    else:
        for system, (cols, lats) in runs.items():
            rows[system] = {"avg_latency_s": round(sum(lats) / len(lats), 2)}

    df = pd.DataFrame(rows).T
    out = out_dir / "benchmark.csv"
    df.to_csv(out)
    print("\n=== JISR: Agentic vs Naive ===")
    print(df.to_string())
    print(f"\nSaved -> {out}\nRaw answers -> {out_dir / 'raw_answers.json'}")


if __name__ == "__main__":
    main()
