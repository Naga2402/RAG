# -*- coding: utf-8 -*-
"""CRITICAL pre-flight test: Arabic must render to PDF *and* extract back as the
original logical text. Naive reshaping stores presentation forms in visual order,
which extracts as mojibake and would silently poison the whole RAG index.

Compares two approaches and reports which round-trips cleanly.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import fitz  # PyMuPDF

AR = "يلتزم العميل بسداد جميع الفواتير خلال ثلاثين يوماً من تاريخ الفاتورة."
EN = "The Client shall pay all invoices within thirty days of the invoice date."
OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def norm(s):
    return "".join(s.split())


def report(name, extracted):
    ok = norm(AR) in norm(extracted)
    print(f"\n--- {name} ---")
    print("  extracted :", extracted.strip()[:90].replace("\n", " "))
    print("  round-trip:", "PASS ✅" if ok else "FAIL ❌ (extraction != original)")
    return ok


# ---------- Approach 1: PyMuPDF insert_htmlbox (native shaping, logical storage)
def approach_htmlbox():
    p = OUT / "ar_test_htmlbox.pdf"
    doc = fitz.open()
    page = doc.new_page()
    html = f"""<div style="font-family:Arial;font-size:14px;direction:rtl;text-align:right">{AR}</div>
               <div style="font-family:Arial;font-size:14px">{EN}</div>"""
    page.insert_htmlbox(fitz.Rect(50, 50, 545, 300), html)
    doc.save(p); doc.close()
    txt = fitz.open(p)[0].get_text()
    return report("PyMuPDF insert_htmlbox", txt), p


# ---------- Approach 2: reportlab + arabic_reshaper + bidi (classic route)
def approach_reshaper():
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import arabic_reshaper
    from bidi.algorithm import get_display

    pdfmetrics.registerFont(TTFont("ArialU", "C:/Windows/Fonts/arial.ttf"))
    p = OUT / "ar_test_reshaper.pdf"
    c = canvas.Canvas(str(p))
    c.setFont("ArialU", 14)
    shaped = get_display(arabic_reshaper.reshape(AR))
    c.drawRightString(545, 750, shaped)
    c.drawString(50, 720, EN)
    c.save()
    txt = fitz.open(p)[0].get_text()
    return report("reportlab + reshaper + bidi", txt), p


if __name__ == "__main__":
    print("Original Arabic:", AR)
    results = {}
    for fn in (approach_htmlbox, approach_reshaper):
        try:
            ok, path = fn()
            results[fn.__name__] = (ok, path)
        except Exception as e:
            print(f"\n--- {fn.__name__} --- ERROR: {e}")
            results[fn.__name__] = (False, None)
    print("\n==== VERDICT ====")
    for k, (ok, path) in results.items():
        print(f"  {k}: {'USABLE' if ok else 'unusable'}  {path or ''}")
