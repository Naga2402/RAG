# -*- coding: utf-8 -*-
"""Corpus A — parallel bilingual (Arabic/English) document corpus.

Streams the UN Parallel Corpus (Helsinki-NLP/un_pc, ar-en), groups aligned
sentence pairs into coherent multi-section documents, and renders each document
as a PDF in BOTH languages. Because every Arabic document is an exact twin of an
English one, cross-lingual retrieval can be evaluated precisely: ask the same
question in either language and check that the router returns the corresponding
document.

Arabic is shaped with arabic-reshaper + python-bidi for correct visual rendering;
the ingestion pipeline canonicalises the extracted presentation forms via
src/ingestion/arabic.py.

Usage:
    python scripts/build_corpus.py --docs 400 --out data/raw
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONT_REG, FONT_BOLD = "DocFont", "DocFontB"
pdfmetrics.registerFont(TTFont(FONT_REG, "C:/Windows/Fonts/arial.ttf"))
pdfmetrics.registerFont(TTFont(FONT_BOLD, "C:/Windows/Fonts/arialbd.ttf"))

AR_RE = re.compile(r"[؀-ۿ]")
W, H = A4
MARGIN = 20 * mm

SUBJECTS_EN = [
    "Report of the Secretary-General", "Resolution adopted by the General Assembly",
    "Note by the Secretariat", "Report of the Advisory Committee",
    "Programme Budget Implications", "Report of the Working Group",
    "Administrative and Budgetary Matters", "Report on Contract Compliance",
    "Technical Cooperation Agreement", "Procurement and Supply Report",
]
SUBJECTS_AR = [
    "تقرير الأمين العام", "قرار اتخذته الجمعية العامة",
    "مذكرة من الأمانة العامة", "تقرير اللجنة الاستشارية",
    "الآثار المترتبة في الميزانية البرنامجية", "تقرير الفريق العامل",
    "المسائل الإدارية ومسائل الميزانية", "تقرير عن الامتثال للعقود",
    "اتفاق التعاون التقني", "تقرير المشتريات والإمدادات",
]


import html as _html


def clean(s: str) -> str:
    """UN source text carries HTML entities and stray spacing."""
    s = _html.unescape(_html.unescape(s or ""))
    s = s.replace("&quot;", '"').replace("&amp;", "&")
    return re.sub(r"\s+", " ", s).strip()


def good_pair(en: str, ar: str) -> bool:
    """Keep only substantive, genuinely bilingual sentence pairs."""
    if not en or not ar:
        return False
    en, ar = en.strip(), ar.strip()
    if not (60 <= len(en) <= 400 and 50 <= len(ar) <= 400):
        return False
    if en == ar:                       # document codes, numerals
        return False
    if not AR_RE.search(ar):           # AR side must actually be Arabic
        return False
    if AR_RE.search(en):               # EN side must not be
        return False
    if len(en.split()) < 8:
        return False
    return True


def wrap(text: str, font: str, size: float, max_w: float) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if pdfmetrics.stringWidth(trial, font, size) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def shape(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))


def render_pdf(path: pathlib.Path, doc_id: str, title: str, date: str,
               sections: list[tuple[str, list[str]]], rtl: bool) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    max_w = W - 2 * MARGIN
    y = H - MARGIN

    def line(txt, font=FONT_REG, size=10.5, gap=5.2, bold=False):
        nonlocal y
        f = FONT_BOLD if bold else font
        for ln in wrap(txt, f, size, max_w):
            if y < MARGIN + 30:
                c.showPage(); y = H - MARGIN
            c.setFont(f, size)
            out = shape(ln) if rtl else ln
            if rtl:
                c.drawRightString(W - MARGIN, y, out)
            else:
                c.drawString(MARGIN, y, out)
            y -= (size + gap)

    # header block
    c.setFont(FONT_BOLD, 9)
    hdr = shape("الأمم المتحدة") if rtl else "UNITED NATIONS"
    (c.drawRightString(W - MARGIN, y, hdr) if rtl else c.drawString(MARGIN, y, hdr))
    y -= 16
    line(f"{'الوثيقة' if rtl else 'Document'}: {doc_id}", size=9.5, gap=3)
    line(f"{'التاريخ' if rtl else 'Date'}: {date}", size=9.5, gap=3)
    y -= 8
    line(title, size=13.5, bold=True, gap=9)
    y -= 4

    for head, paras in sections:
        line(head, size=11.5, bold=True, gap=6)
        for p in paras:
            line(p, size=10.5, gap=5.2)
        y -= 6
    c.save()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=400, help="number of PARALLEL pairs")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--sents", type=int, default=18, help="sentence pairs per document")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    out_en = root / args.out / "en"; out_ar = root / args.out / "ar"
    out_en.mkdir(parents=True, exist_ok=True); out_ar.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    print("Streaming UN parallel corpus (ar-en)…")
    ds = load_dataset("Helsinki-NLP/un_pc", "ar-en", split="train", streaming=True)

    needed = args.docs * args.sents
    pairs, seen = [], set()
    for row in ds:
        t = row["translation"]
        en, ar = clean(t.get("en")), clean(t.get("ar"))
        if not good_pair(en, ar) or en in seen:
            continue
        seen.add(en); pairs.append((en, ar))
        if len(pairs) >= needed:
            break
        if len(pairs) % 2000 == 0:
            print(f"  collected {len(pairs)}/{needed}")
    print(f"Collected {len(pairs)} usable sentence pairs.")

    rng = random.Random(42)
    manifest = []
    n_docs = min(args.docs, len(pairs) // args.sents)
    for i in range(n_docs):
        block = pairs[i * args.sents:(i + 1) * args.sents]
        if not block:
            break
        doc_id = f"JISR/{2020 + i % 6}/{1000 + i}"
        si = i % len(SUBJECTS_EN)
        date = f"{rng.randint(1,28):02d} {rng.choice(['January','March','June','September','November'])} {2020 + i % 6}"
        date_ar = f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{2020 + i % 6}"

        # split the block into 3 sections
        per = max(1, len(block) // 3)
        sec_en, sec_ar = [], []
        heads_en = ["I.  Background", "II.  Findings and Observations", "III.  Recommendations"]
        heads_ar = ["أولاً - معلومات أساسية", "ثانياً - النتائج والملاحظات", "ثالثاً - التوصيات"]
        for s in range(3):
            chunk = block[s * per:(s + 1) * per] if s < 2 else block[2 * per:]
            if not chunk:
                continue
            sec_en.append((heads_en[s], [p[0] for p in chunk]))
            sec_ar.append((heads_ar[s], [p[1] for p in chunk]))

        stem = f"doc_{i:04d}"
        pen = out_en / f"{stem}_en.pdf"; par = out_ar / f"{stem}_ar.pdf"
        render_pdf(pen, doc_id, SUBJECTS_EN[si], date, sec_en, rtl=False)
        render_pdf(par, doc_id, SUBJECTS_AR[si], date_ar, sec_ar, rtl=True)
        manifest.append({"pair_id": stem, "doc_id": doc_id,
                         "en_file": str(pen.relative_to(root)), "ar_file": str(par.relative_to(root)),
                         "title_en": SUBJECTS_EN[si], "title_ar": SUBJECTS_AR[si],
                         "sentences": [{"en": a, "ar": b} for a, b in block]})
        if (i + 1) % 50 == 0:
            print(f"  rendered {i+1}/{n_docs} pairs")

    mpath = root / args.out / "manifest.jsonl"
    with open(mpath, "w", encoding="utf-8") as fh:
        for m in manifest:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"\nDone: {len(manifest)} parallel pairs = {len(manifest)*2} PDFs")
    print(f"  EN -> {out_en}\n  AR -> {out_ar}\n  manifest -> {mpath}")


if __name__ == "__main__":
    main()
