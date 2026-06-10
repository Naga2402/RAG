"""Language detection — tags each chunk/doc as 'en' or 'ar' so the router and
per-language indices stay clean. Arabic Unicode range gives a fast, reliable
signal; langdetect is the fallback for mixed/Latin text."""
from __future__ import annotations

from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # deterministic output

_AR_RANGE = (0x0600, 0x06FF)


def arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    ar = sum(1 for c in text if _AR_RANGE[0] <= ord(c) <= _AR_RANGE[1])
    letters = sum(1 for c in text if c.isalpha())
    return ar / letters if letters else 0.0


def detect_language(text: str) -> str:
    """Return 'ar' or 'en'. Defaults to 'en' on ambiguity."""
    if arabic_ratio(text) > 0.15:
        return "ar"
    try:
        return "ar" if detect(text) == "ar" else "en"
    except Exception:
        return "en"
