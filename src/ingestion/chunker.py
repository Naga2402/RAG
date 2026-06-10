"""Layout-aware chunking. Splits on structural boundaries (blank lines /
headings) first, then packs into ~chunk_size windows with overlap. Keeps
Arabic and English chunks separable by tagging language per chunk."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import CFG
from src.ingestion.language import detect_language

_C = CFG["ingestion"]["chunking"]


@dataclass
class Chunk:
    text: str
    lang: str
    source: str
    index: int
    meta: dict = field(default_factory=dict)


def _segments(text: str) -> list[str]:
    # structural split: paragraphs / blank-line separated blocks
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return blocks or [text.strip()]


def chunk_text(text: str, source: str) -> list[Chunk]:
    size, overlap = _C["chunk_size"], _C["chunk_overlap"]
    chunks: list[Chunk] = []
    buf = ""
    idx = 0
    for seg in _segments(text):
        if len(buf) + len(seg) + 1 <= size:
            buf = f"{buf}\n{seg}".strip()
        else:
            if buf:
                chunks.append(Chunk(buf, detect_language(buf), source, idx))
                idx += 1
                buf = (buf[-overlap:] + "\n" + seg).strip()
            else:
                buf = seg
    if buf:
        chunks.append(Chunk(buf, detect_language(buf), source, idx))
    return chunks
