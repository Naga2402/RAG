"""Arabic text normalisation for the ingestion pipeline.

Many real-world Arabic PDFs store glyphs as **Arabic Presentation Forms**
(U+FB50–FDFF, U+FE70–FEFF) rather than the canonical letters. Extracted text then
*looks* correct but uses different codepoints, which silently breaks embedding,
matching and evaluation. We normalise every chunk so the index only ever holds
canonical Arabic.

Also strips tatweel and (optionally) diacritics, which vary between OCR output and
digital text and would otherwise cause spurious retrieval misses.
"""
from __future__ import annotations

import re
import unicodedata

# Harakat / tashkeel (short vowels, shadda, sukun, tanween)
_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_PRESENTATION = re.compile(r"[ﭐ-﷿ﹰ-﻿]")


def has_presentation_forms(text: str) -> bool:
    """True if the text contains Arabic presentation-form codepoints."""
    return bool(_PRESENTATION.search(text or ""))


def normalize_arabic(text: str, strip_diacritics: bool = True) -> str:
    """Canonicalise Arabic text extracted from PDFs or OCR.

    NFKC maps presentation forms back to base letters; we then unify the
    alef/ya/ta-marbuta variants that OCR and typography use interchangeably.
    """
    if not text:
        return text
    out = unicodedata.normalize("NFKC", text)
    if strip_diacritics:
        out = _DIACRITICS.sub("", out)
    # unify common orthographic variants
    out = re.sub(r"[آأإٱ]", "ا", out)  # آ أ إ ٱ -> ا
    out = out.replace("ى", "ي")                        # ى -> ي
    out = out.replace("ة", "ه")                        # ة -> ه
    out = re.sub(r"[ \t ]+", " ", out)
    return out.strip()


def normalize_text(text: str, lang: str | None = None) -> str:
    """Entry point used by the chunker. Only touches Arabic content."""
    if lang == "en":
        return text
    if lang == "ar" or has_presentation_forms(text) or re.search(r"[؀-ۿ]", text or ""):
        return normalize_arabic(text)
    return text
