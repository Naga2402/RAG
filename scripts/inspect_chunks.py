"""Quick sanity check on ingested chunks: language distribution + samples."""
import collections
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/chunks.jsonl"
rows = [json.loads(line) for line in open(path, encoding="utf-8")]

langs = collections.Counter(r["lang"] for r in rows)
print("total chunks:", len(rows))
print("by language :", dict(langs))
print("sources     :", len({r["source"] for r in rows}), "documents")
en = next((r["text"][:60] for r in rows if r["lang"] == "en"), "(none)")
ar = next((r["text"][:60] for r in rows if r["lang"] == "ar"), "(none)")
print("sample EN   :", en.replace(chr(10), " "))
print("sample AR   :", ar.replace(chr(10), " "))
