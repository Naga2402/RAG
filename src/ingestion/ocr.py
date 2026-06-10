"""OCR for heterogeneous EN/AR documents (invoices, contracts, manuals).

Uses Tesseract with the combined 'eng+ara' model so Arabic RTL text is captured.
PDFs are rasterised page-by-page; native-text PDFs short-circuit OCR via PyMuPDF.
Plain-text/markdown (common DMS exports) are read directly — no OCR needed.

OCR-only dependencies (pytesseract, pdf2image, Tesseract + 'ara' traineddata,
Poppler) are imported lazily, so text ingestion works before they're installed.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.config import CFG, ROOT

_OCR = CFG["ingestion"]["ocr"]

TEXT_EXT = {".txt", ".md"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Point Tesseract at the project-local tessdata (eng+ara) if configured.
_tessdata = _OCR.get("tessdata_dir")
if _tessdata:
    _abs = (ROOT / _tessdata).resolve()
    if _abs.is_dir():
        os.environ.setdefault("TESSDATA_PREFIX", str(_abs))

# Poppler bin for pdf2image: config value, else POPPLER_PATH env, else PATH.
_POPPLER = _OCR.get("poppler_path") or os.getenv("POPPLER_PATH") or None


def _ocr_image(img) -> str:
    import pytesseract  # lazy: needs Tesseract binary
    return pytesseract.image_to_string(img, lang=_OCR["languages"])


def extract_text(path: str | Path) -> str:
    """Return full text for a text file, PDF, or image. Tries native PDF text
    first, falls back to OCR when a page has no embedded text layer."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in TEXT_EXT:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix in IMAGE_EXT:
        from PIL import Image
        return _ocr_image(Image.open(path))

    if suffix == ".pdf":
        import fitz  # PyMuPDF, lazy
        from pdf2image import convert_from_path  # lazy: needs Poppler
        parts: list[str] = []
        doc = fitz.open(path)
        needs_ocr_pages = []
        for i, page in enumerate(doc):
            native = page.get_text().strip()
            if len(native) > 40:           # has a real text layer
                parts.append(native)
            else:
                needs_ocr_pages.append(i)
        if needs_ocr_pages:
            images = convert_from_path(
                str(path), dpi=_OCR["dpi"], poppler_path=_POPPLER
            )
            for i in needs_ocr_pages:
                parts.append(_ocr_image(images[i]))
        return "\n".join(parts)

    raise ValueError(f"Unsupported file type: {path.suffix}")
