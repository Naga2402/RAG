# -*- coding: utf-8 -*-
"""Corpus B — 'scanned' bilingual documents that force the OCR path.

Takes a subset of Corpus A and rasterises each page at scan-like DPI, optionally
adding slight rotation and noise, then rebuilds an IMAGE-ONLY PDF. With no text
layer, the ingestion pipeline must fall back to Tesseract OCR — exercising the
Arabic (RTL) OCR route end to end, which is what a real DMS receives from a
document scanner.

Usage:
    python scripts/build_scanned.py --n 60 --out data/scanned
"""
from __future__ import annotations

import argparse
import io
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import fitz
from PIL import Image, ImageEnhance, ImageFilter

ROOT = pathlib.Path(__file__).resolve().parents[1]


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    """Mimic a real scanner: tiny skew, slight blur, contrast shift, light noise."""
    if rng.random() < 0.7:
        img = img.rotate(rng.uniform(-0.7, 0.7), resample=Image.BICUBIC,
                         fillcolor="white", expand=False)
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.5)))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.92, 1.08))
    return img


def scan_pdf(src: pathlib.Path, dst: pathlib.Path, dpi: int, rng: random.Random) -> None:
    doc = fitz.open(src)
    out = fitz.open()
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        img = degrade(img, rng)
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=82)
        rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
        np = out.new_page(width=page.rect.width, height=page.rect.height)
        np.insert_image(rect, stream=buf.getvalue())   # image only -> no text layer
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst); out.close(); doc.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="pairs to scan (x2 files)")
    ap.add_argument("--src", default="data/raw")
    ap.add_argument("--out", default="data/scanned")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    rng = random.Random(11)
    src = ROOT / args.src
    en = sorted((src / "en").glob("*.pdf"))[: args.n]
    ar = sorted((src / "ar").glob("*.pdf"))[: args.n]
    print(f"Scanning {len(en)} EN + {len(ar)} AR documents at {args.dpi} DPI…")

    for i, p in enumerate(en + ar):
        lang = "en" if p.parent.name == "en" else "ar"
        scan_pdf(p, ROOT / args.out / lang / p.name, args.dpi, rng)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(en)+len(ar)}")
    print(f"Done -> {ROOT / args.out}  (image-only PDFs; ingestion will use OCR)")


if __name__ == "__main__":
    main()
