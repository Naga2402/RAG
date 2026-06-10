"""BGE-M3 embedding wrapper. BGE-M3 is multilingual and returns BOTH dense and
sparse (lexical) vectors in one pass — the foundation for hybrid EN/AR retrieval
and the cross-lingual router."""
from __future__ import annotations

from functools import lru_cache

from src.config import CFG

_E = CFG["embeddings"]


@lru_cache(maxsize=1)
def _model():
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel(_E["model"], use_fp16=(_E["device"] == "cuda"))


def warmup() -> None:
    """Force-load the embedding model (and torch's OpenMP runtime).

    On Windows, loading torch AFTER psycopg's native libs triggers a duplicate
    libiomp5md.dll crash (segfault). Call this before opening a DB connection so
    torch's OpenMP initializes first. Safe to call multiple times (cached).
    """
    _model()


def embed(texts: list[str], dense: bool = True, sparse: bool = True) -> list[dict]:
    """Return list of {'dense': [...], 'sparse': {token_id: weight}} per text."""
    out = _model().encode(
        texts,
        return_dense=dense,
        return_sparse=sparse,
        return_colbert_vecs=False,
    )
    results = []
    for i in range(len(texts)):
        item = {}
        if dense:
            item["dense"] = out["dense_vecs"][i].tolist()
        if sparse:
            item["sparse"] = {int(k): float(v) for k, v in out["lexical_weights"][i].items()}
        results.append(item)
    return results


def embed_query(text: str) -> dict:
    return embed([text])[0]
