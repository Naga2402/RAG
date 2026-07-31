"""Language detection — tags each chunk/doc as 'en' or 'ar' so the router and
per-language indices stay clean. Arabic Unicode range gives a fast, reliable
signal; langdetect is the fallback for mixed/Latin text."""
from __future__ import annotations

from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # deterministic output

# Base Arabic block, plus the Presentation Forms blocks that PDF text layers
# frequently use — without these, Arabic extracted from PDFs is misread as English.
_AR_RANGES = (
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0xFB50, 0xFDFF),   # Presentation Forms-A
    (0xFE70, 0xFEFF),   # Presentation Forms-B
)


def _is_arabic_char(c: str) -> bool:
    o = ord(c)
    return any(lo <= o <= hi for lo, hi in _AR_RANGES)


def arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    ar = sum(1 for c in text if _is_arabic_char(c))
    letters = sum(1 for c in text if c.isalpha() or _is_arabic_char(c))
    return ar / letters if letters else 0.0


def detect_language(text: str) -> str:
    """Return 'ar' or 'en'. Defaults to 'en' on ambiguity."""
    if arabic_ratio(text) > 0.15:
        return "ar"
    try:
        return "ar" if detect(text) == "ar" else "en"
    except Exception:
        return "en"
