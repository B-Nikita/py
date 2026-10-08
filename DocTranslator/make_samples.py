"""Generate test inputs for doc_translate.py.

samples/complex_layout.pdf  - digital PDF: title, 2-column body, inline bold/italic,
                              coloured text, shaded table, vector logo, footer.
samples/scanned_page.png    - the same page rasterised at 200 DPI with noise,
                              i.e. what a scanner / phone photo would give you.
"""
import os

import numpy as np
import pymupdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "samples")

BODY = [
    "Large language models have changed how organisations handle multilingual "
    "documents. Instead of sending every contract to an agency, teams can now "
    "produce a <b>first-pass translation</b> in minutes and route only the "
    "sensitive sections to human reviewers.",
    "The hard part is not the translation itself but the <i>layout</i>. Invoices, "
    "brochures and scientific papers mix columns, tables, captions and footnotes, "
    "and readers expect the translated file to look exactly like the original.",
    "Our pipeline keeps every vector graphic, image and background in place and "
    "replaces only the text, so tables keep their borders and shading.",
    "Text expansion is the main risk: German or French output is often thirty "
    "percent longer than English. The renderer first grows each box into free "
    "space, then shrinks the font within a safe limit, and finally asks the model "
    "for a shorter phrasing when nothing else works.",
]


def build_pdf(path):
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Title"], fontSize=22,
                           textColor=colors.HexColor("#1F3A93"), alignment=0)
    sub = ParagraphStyle("s", parent=styles["Normal"], fontSize=11,
                         textColor=colors.HexColor("#555555"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13,
                        textColor=colors.HexColor("#C0392B"))
    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=10,
                          leading=13, alignment=4)
    cell = ParagraphStyle("c", parent=styles["BodyText"], fontSize=9, leading=11)
    cell_h = ParagraphStyle("ch", parent=cell, textColor=colors.white,
                            fontName="Helvetica-Bold")

    W, H = A4
    m = 18 * mm
    gap = 8 * mm
    col_w = (W - 2 * m - gap) / 2
    top_h = 38 * mm
    frames = [
        Frame(m, H - m - top_h, W - 2 * m, top_h, id="top"),
        Frame(m, 120 * mm, col_w, H - 2 * m - top_h - 120 * mm + m, id="c1"),
        Frame(m + col_w + gap, 120 * mm, col_w, H - 2 * m - top_h - 120 * mm + m, id="c2"),
        Frame(m, m + 8 * mm, W - 2 * m, 120 * mm - m - 8 * mm, id="bottom"),
    ]

    def decorate(canv, _doc):
        canv.saveState()
        canv.setFillColor(colors.HexColor("#1F3A93"))
        canv.circle(W - m - 12 * mm, H - m - 10 * mm, 9 * mm, fill=1, stroke=0)
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica-Bold", 14)
        canv.drawCentredString(W - m - 12 * mm, H - m - 12 * mm, "AI")
        canv.setStrokeColor(colors.HexColor("#BBBBBB"))
        canv.line(m, m + 6 * mm, W - m, m + 6 * mm)
        canv.setFillColor(colors.HexColor("#777777"))
        canv.setFont("Helvetica", 8)
        canv.drawString(m, m, "Confidential - internal use only")
        canv.drawRightString(W - m, m, "Page 1 of 1")
        canv.restoreState()

    doc = BaseDocTemplate(path, pagesize=A4, leftMargin=m, rightMargin=m,
                          topMargin=m, bottomMargin=m)
    doc.addPageTemplates([PageTemplate(id="p", frames=frames, onPage=decorate)])

    story = [
        Paragraph("Translating Documents Without Losing Layout", title),
        Paragraph("A practical guide for operations teams - 2026 edition", sub),
        FrameBreak(),
        Paragraph("Why it matters", h2),
        Paragraph(BODY[0], body), Spacer(1, 6),
        Paragraph(BODY[1], body),
        FrameBreak(),
        Paragraph("How the pipeline works", h2),
        Paragraph(BODY[2], body), Spacer(1, 6),
        Paragraph(BODY[3], body),
        FrameBreak(),
        Paragraph("Quarterly results by region", h2),
    ]
    rows = [
        ["Region", "Revenue", "Growth", "Comment"],
        ["North America", "$4.2M", "+12%", "Strong demand from healthcare customers"],
        ["Europe", "$3.1M", "+8%", "New office opened in Berlin"],
        ["Asia Pacific", "$2.7M", "+21%", "Fastest growing market this year"],
        ["Latin America", "$0.9M", "-3%", "Currency headwinds offset new deals"],
    ]
    data = [[Paragraph(c, cell_h) for c in rows[0]]] + \
           [[Paragraph(c, cell) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=[38 * mm, 25 * mm, 20 * mm, W - 2 * m - 83 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A93")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2FB")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [t, Spacer(1, 8),
              Paragraph("Note: figures are unaudited and rounded to one decimal place.",
                        ParagraphStyle("n", parent=cell, textColor=colors.HexColor("#777777"),
                                       fontName="Helvetica-Oblique"))]
    doc.build(story)


def build_scan(pdf_path, png_path, dpi=200):
    page = pymupdf.open(pdf_path)[0]
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
    rng = np.random.default_rng(0)
    noisy = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    pymupdf.Pixmap(pymupdf.csRGB, pix.w, pix.h, noisy.tobytes(), False).save(png_path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pdf = os.path.join(OUT, "complex_layout.pdf")
    build_pdf(pdf)
    build_scan(pdf, os.path.join(OUT, "scanned_page.png"))
    print("wrote", pdf, "and scanned_page.png")
