"""End-to-end check of the pgvector hybrid index: row counts per language table,
plus a bilingual hybrid retrieval to confirm the EN/AR routing + search works."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.indexing.embeddings import embed_query, warmup
from src.indexing import vector_store as vs
from src.config import CFG

warmup()  # torch before psycopg (OpenMP ordering)
conn = vs.connect()

print("=== row counts ===")
for lang, table in CFG["vector_store"]["pgvector"]["tables"].items():
    n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    print(f"  {lang}: {n} chunks in {table}")

queries = {
    "en": "What are the payment terms in the service contract?",
    "ar": "ما هي مدة ضمان المضخة الصناعية؟",   # warranty period of the pump
}
for lang, q in queries.items():
    print(f"\n=== [{lang}] {q}")
    hits = vs.hybrid_search(conn, lang, embed_query(q), top_k=2)
    for h in hits:
        snippet = h["content"].replace("\n", " ")[:70]
        print(f"  {h['score']:.3f}  {h['source']}#{h['chunk_idx']}  {snippet}")

conn.close()
