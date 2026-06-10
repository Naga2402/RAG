"""Ingestion entrypoint: data/raw/* -> OCR -> chunk -> data/processed/chunks.jsonl

Usage:
    python -m src.ingestion.run --input data/raw
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from src.config import CFG, ROOT
from src.ingestion.chunker import chunk_text
from src.ingestion.ocr import IMAGE_EXT, TEXT_EXT, extract_text

SUPPORTED = {".pdf"} | IMAGE_EXT | TEXT_EXT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=CFG["ingestion"]["input_dir"])
    ap.add_argument("--output", default=CFG["ingestion"]["output_dir"])
    args = ap.parse_args()

    in_dir = ROOT / args.input
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "chunks.jsonl"

    files = [p for p in in_dir.rglob("*") if p.suffix.lower() in SUPPORTED]
    print(f"Found {len(files)} documents in {in_dir}")

    n_chunks = 0
    with open(out_file, "w", encoding="utf-8") as fh:
        for f in tqdm(files, desc="ingesting"):
            try:
                text = extract_text(f)
            except Exception as e:  # noqa: BLE001
                print(f"  ! skip {f.name}: {e}")
                continue
            for ch in chunk_text(text, source=str(f.relative_to(in_dir))):
                fh.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")
                n_chunks += 1

    print(f"Wrote {n_chunks} chunks -> {out_file}")


if __name__ == "__main__":
    main()
