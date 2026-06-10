"""pgvector store with per-language collections (chunks_en / chunks_ar).

Separate tables per language are what make the cross-lingual router a real
retrieval decision rather than a prompt trick. Hybrid scoring combines dense
cosine similarity with sparse (BM25-style) lexical overlap.
"""
from __future__ import annotations

import json

import psycopg
from pgvector.psycopg import register_vector

from src.config import CFG

_V = CFG["vector_store"]["pgvector"]
_H = CFG["vector_store"]["hybrid"]
_DIM = 1024  # BGE-M3 dense dimension


def connect() -> psycopg.Connection:
    conn = psycopg.connect(
        host=_V["host"], port=_V["port"], dbname=_V["dbname"],
        user=_V["user"], password=_V["password"],
    )
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    for table in _V["tables"].values():
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id        BIGSERIAL PRIMARY KEY,
                source    TEXT,
                chunk_idx INT,
                content   TEXT,
                sparse    JSONB,
                embedding VECTOR({_DIM})
            )
        """)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_emb_idx ON {table} "
            f"USING hnsw (embedding vector_cosine_ops)"
        )
    conn.commit()


def reset(conn) -> None:
    """Truncate both language tables for a clean rebuild."""
    for table in _V["tables"].values():
        conn.execute(f"TRUNCATE {table} RESTART IDENTITY")
    conn.commit()


def upsert(conn, lang: str, rows: list[dict]) -> None:
    table = _V["tables"][lang]
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO {table} (source, chunk_idx, content, sparse, embedding) "
            f"VALUES (%s, %s, %s, %s, %s)",
            [(r["source"], r["index"], r["content"],
              json.dumps(r["sparse"]), r["dense"]) for r in rows],
        )
    conn.commit()


def _sparse_score(q: dict[int, float], d: dict) -> float:
    d = {int(k): v for k, v in (d or {}).items()}
    return sum(w * d.get(t, 0.0) for t, w in q.items())


def hybrid_search(conn, lang: str, query_emb: dict, top_k: int | None = None) -> list[dict]:
    """Dense ANN candidates re-scored with weighted dense+sparse fusion."""
    table = _V["tables"][lang]
    k = top_k or _H["top_k"]
    pool = max(k * 4, 20)
    with conn.cursor() as cur:
        # Cast the bound list to `vector`; without context psycopg sends it as
        # double precision[], which the <=> operator does not accept.
        cur.execute(
            f"SELECT id, source, chunk_idx, content, sparse, "
            f"1 - (embedding <=> %s::vector) AS dense_sim "
            f"FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s",
            (query_emb["dense"], query_emb["dense"], pool),
        )
        rows = cur.fetchall()

    q_sparse = query_emb.get("sparse", {})
    scored = []
    for _id, source, idx, content, sparse, dense_sim in rows:
        s = _sparse_score(q_sparse, sparse)
        score = _H["dense_weight"] * float(dense_sim) + _H["sparse_weight"] * s
        scored.append({"id": _id, "source": source, "chunk_idx": idx,
                       "content": content, "score": score})
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:k]
