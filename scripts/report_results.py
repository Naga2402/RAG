# -*- coding: utf-8 -*-
"""Turn the benchmark output into dissertation-ready tables.

Reads results/benchmark.csv, results/per_question_*.csv and raw_answers.json and
emits:
  * a headline Agentic-vs-Naive metric table (markdown + LaTeX)
  * a per-language breakdown (EN vs AR) — the cross-lingual claim
  * refusal behaviour on the out-of-scope probes (the Critic's contribution)

Usage:
    python scripts/report_results.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
NICE = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision", "context_recall": "Context Recall",
        "avg_latency_s": "Avg latency (s)"}

REFUSAL = ("cannot", "can't", "not available", "does not", "doesn't", "no information",
           "unable", "not mention", "not provide", "not contain", "insufficient",
           "لا يوجد", "لا يمكن", "لا تتوفر", "غير متوف", "غير كاف", "لا يحتوي")


def is_refusal(a: str) -> bool:
    a = (a or "").lower()
    return any(k in a for k in REFUSAL)


def headline() -> pd.DataFrame | None:
    p = RES / "benchmark.csv"
    if not p.exists():
        print("! results/benchmark.csv not found — run the benchmark first.")
        return None
    df = pd.read_csv(p, index_col=0)
    print("\n=== Headline: Agentic vs Naive ===\n")
    show = df.rename(columns=NICE)
    print(show.round(4).to_string())

    if {"agentic", "naive"} <= set(df.index):
        print("\n--- delta (agentic - naive) ---")
        for m in METRICS:
            if m in df.columns:
                d = df.loc["agentic", m] - df.loc["naive", m]
                arrow = "▲" if d > 0 else ("▼" if d < 0 else "=")
                print(f"  {NICE[m]:<20} {d:+.4f}  {arrow}")
    return df


def per_language() -> None:
    """Split the per-question scores by language using the golden set."""
    golden_path = ROOT / "data/golden/golden.jsonl"
    if not golden_path.exists():
        return
    golden = {json.loads(l)["question"]: json.loads(l)
              for l in open(golden_path, encoding="utf-8") if l.strip()}

    rows = []
    for system in ("agentic", "naive"):
        f = RES / f"per_question_{system}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        qcol = "user_input" if "user_input" in df.columns else "question"
        if qcol not in df.columns:
            continue
        df["lang"] = df[qcol].map(lambda q: (golden.get(q) or {}).get("lang", "?"))
        df["type"] = df[qcol].map(lambda q: (golden.get(q) or {}).get("type", "?"))
        for lang in ("en", "ar"):
            sub = df[(df["lang"] == lang) & (df["type"] == "in_scope")]
            if sub.empty:
                continue
            r = {"system": system, "lang": lang.upper(), "n": len(sub)}
            for m in METRICS:
                if m in sub.columns:
                    r[NICE[m]] = round(float(pd.to_numeric(sub[m], errors="coerce").mean()), 4)
            rows.append(r)
    if rows:
        print("\n=== Per-language breakdown (in-scope only) ===\n")
        print(pd.DataFrame(rows).to_string(index=False))


def refusals() -> None:
    p = RES / "raw_answers.json"
    gp = ROOT / "data/golden/golden.jsonl"
    if not (p.exists() and gp.exists()):
        return
    raw = json.load(open(p, encoding="utf-8"))
    golden = [json.loads(l) for l in open(gp, encoding="utf-8") if l.strip()]
    oos_q = {g["question"] for g in golden if g.get("type") == "out_of_scope"}

    print("\n=== Out-of-scope behaviour (the Critic's contribution) ===\n")
    for system, v in raw.items():
        cols = v["cols"]
        idx = [i for i, q in enumerate(cols["question"]) if q in oos_q]
        if not idx:
            continue
        good = sum(1 for i in idx if is_refusal(cols["answer"][i]))
        print(f"  {system:<8} correctly declined {good}/{len(idx)} out-of-scope questions")


def as_markdown(df: pd.DataFrame) -> None:
    if df is None:
        return
    out = RES / "benchmark_table.md"
    cols = [c for c in ["faithfulness", "answer_relevancy", "context_precision",
                        "context_recall", "avg_latency_s"] if c in df.columns]
    lines = ["| System | " + " | ".join(NICE[c] for c in cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for sysname in df.index:
        lines.append("| " + sysname.title() + " | " +
                     " | ".join(f"{df.loc[sysname, c]:.4f}" for c in cols) + " |")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown table -> {out}")


if __name__ == "__main__":
    df = headline()
    per_language()
    refusals()
    as_markdown(df)
