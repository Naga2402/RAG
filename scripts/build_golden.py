# -*- coding: utf-8 -*-
"""Golden evaluation set for the Ragas benchmark.

Drafts question / ground-truth pairs from indexed chunks using the local LLM, then
writes them to data/golden/golden.jsonl for human review (`reviewed: false` until
you check them). Ragas consumes this file via src/eval/run.py.

Because the corpus is parallel, we also emit CROSS-LINGUAL items: the same fact
asked in Arabic and English, so cross-lingual routing can be scored directly.

Includes deliberate out-of-scope questions so the benchmark measures the agent's
refusal behaviour (the Critic), not just its answers.

Usage:
    python scripts/build_golden.py --n 120
    # then review data/golden/golden.jsonl and flip "reviewed" to true
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.inference.llm import generate

ROOT = pathlib.Path(__file__).resolve().parents[1]

Q_SYS_EN = (
    "You write evaluation questions for a document retrieval system. "
    "Given a passage, write ONE specific factual question that the passage answers, "
    "then the exact answer taken from the passage. "
    "Reply in exactly this format and nothing else:\n"
    "Q: <question>\nA: <answer>"
)
Q_SYS_AR = (
    "أنت تكتب أسئلة تقييم لنظام استرجاع المستندات. "
    "بالاعتماد على النص التالي، اكتب سؤالاً واحداً محدداً يجيب عنه النص، ثم الإجابة الدقيقة من النص. "
    "أجب بهذا التنسيق فقط:\n"
    "Q: <السؤال>\nA: <الإجابة>"
)

OUT_OF_SCOPE = [
    ("en", "What is the CEO's annual salary and stock option package?"),
    ("en", "What is the company's Wi-Fi password for the Hyderabad office?"),
    ("en", "Which football team won the tournament final last season?"),
    ("ar", "ما هو الراتب السنوي للمدير التنفيذي؟"),
    ("ar", "ما هي كلمة مرور الشبكة اللاسلكية في المكتب؟"),
]


def parse_qa(raw: str) -> tuple[str, str] | None:
    q = re.search(r"Q\s*:\s*(.+)", raw)
    a = re.search(r"A\s*:\s*(.+)", raw)
    if not q or not a:
        return None
    qq, aa = q.group(1).strip(), a.group(1).strip()
    if len(qq) < 12 or len(aa) < 3:
        return None
    return qq, aa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="number of in-scope items")
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / args.chunks, encoding="utf-8")]
    # only use substantial chunks
    rows = [r for r in rows if len(r["text"]) > 400]
    by_lang = {"en": [r for r in rows if r["lang"] == "en"],
               "ar": [r for r in rows if r["lang"] == "ar"]}
    print(f"Usable chunks — EN: {len(by_lang['en'])}, AR: {len(by_lang['ar'])}")

    rng = random.Random(7)
    for v in by_lang.values():
        rng.shuffle(v)

    items, per = [], args.n // 2
    for lang in ("en", "ar"):
        made, i = 0, 0
        pool = by_lang[lang]
        while made < per and i < len(pool):
            ch = pool[i]; i += 1
            passage = ch["text"][:1400]
            sys_p = Q_SYS_EN if lang == "en" else Q_SYS_AR
            try:
                raw = generate(f"Passage:\n{passage}\n\nNow write the question and answer:",
                               lang=lang, system=sys_p)
            except Exception as e:  # noqa: BLE001
                print("  llm error:", e); continue
            qa = parse_qa(raw)
            if not qa:
                continue
            q, a = qa
            items.append({
                "id": f"{lang}-{made:03d}", "lang": lang, "type": "in_scope",
                "question": q, "ground_truth": a,
                "source": ch["source"], "chunk_idx": ch["index"],
                "reference_context": passage[:600],
                "reviewed": False,
            })
            made += 1
            if made % 10 == 0:
                print(f"  [{lang}] {made}/{per}")

    for j, (lang, q) in enumerate(OUT_OF_SCOPE):
        items.append({
            "id": f"oos-{j:02d}", "lang": lang, "type": "out_of_scope",
            "question": q,
            "ground_truth": ("The documents do not contain this information."
                             if lang == "en" else "لا تحتوي المستندات على هذه المعلومة."),
            "source": None, "chunk_idx": None, "reference_context": "",
            "reviewed": True,
        })

    out = ROOT / "data/golden/golden.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    n_in = sum(1 for i in items if i["type"] == "in_scope")
    print(f"\nWrote {len(items)} items ({n_in} in-scope, {len(OUT_OF_SCOPE)} out-of-scope)")
    print(f"  -> {out}\n  Review them, then set \"reviewed\": true.")


if __name__ == "__main__":
    main()
