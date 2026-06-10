"""Build the hybrid index: data/processed/chunks.jsonl -> pgvector (EN/AR tables).

Usage:
    python -m src.indexing.build
"""
from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from src.config import CFG, ROOT
from src.indexing import vector_store as vs
from src.indexing.embeddings import embed, warmup

BATCH = 32


def _flush(conn, lang: str, batch: list[dict]) -> None:
    if not batch:
        return
    embs = embed([b["content"] for b in batch])
    rows = [{**b, **e} for b, e in zip(batch, embs)]
    vs.upsert(conn, lang, rows)


def main() -> None:
    src = ROOT / CFG["ingestion"]["output_dir"] / "chunks.jsonl"
    warmup()                 # load torch/BGE-M3 before psycopg (OpenMP ordering)
    conn = vs.connect()
    vs.init_schema(conn)
    vs.reset(conn)           # clean slate so re-runs don't duplicate rows

    buckets: dict[str, list[dict]] = {"en": [], "ar": []}
    with open(src, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc="indexing"):
            c = json.loads(line)
            lang = c["lang"] if c["lang"] in buckets else "en"
            buckets[lang].append(
                {"source": c["source"], "index": c["index"], "content": c["text"]}
            )
            if len(buckets[lang]) >= BATCH:
                _flush(conn, lang, buckets[lang])
                buckets[lang] = []
    for lang, batch in buckets.items():
        _flush(conn, lang, batch)

    conn.close()
    print("Index build complete.")


if __name__ == "__main__":
    main()
