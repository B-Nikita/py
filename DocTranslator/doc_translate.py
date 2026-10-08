#!/usr/bin/env python3
"""
doc_translate.py - translate PDFs and images with an LLM while keeping the original layout.

How the layout is kept
----------------------
Converting a document to Markdown and building a new file from it always loses the layout,
whichever parser you use. This script edits the original page instead:

    original page ──► text blocks with geometry ──► LLM translation ──► text swapped in place

  * Digital PDFs:  text, fonts, sizes, colours and boxes come from the PDF itself (PyMuPDF),
                   so no OCR is needed. Only the glyphs of translated blocks are removed
                   (redaction with images and vector graphics kept), so tables, shading, logos,
                   charts and backgrounds are unchanged.
  * Scans/images:  the page is rendered at 300 DPI and OCR'd (Tesseract, or LlamaParse for text
                   and headings), table cells are found from ruling lines (OpenCV), only text
                   pixels are erased (mask + inpainting), and the translation is drawn on top.
                   If you use Claude, it also gets the page image and corrects the OCR in the
                   same call that translates it.

Text fitting (per block, in this order)
  1. original box, original font size
  2. grow downwards into free space (stops at the next block, rule, image or cell border)
  3. grow sideways into free space (for left, right or centred text)
  4. shrink the font, but not below --min-scale (default 0.70)
  5. ask the LLM for a shorter translation within a character budget, then try 1-4 again
  6. last resort: shrink without a floor, and record the block as "overflow" in the report

Usage
  pip install pymupdf anthropic opencv-python-headless numpy pytesseract   # + apt install tesseract-ocr
  export ANTHROPIC_API_KEY=...
  python doc_translate.py input.pdf  -t German
  python doc_translate.py scan.png   -t "Brazilian Portuguese" --ocr-lang por
  python doc_translate.py input.pdf  -t French --translator pseudo --debug   # offline dry run
  python doc_translate.py scan.pdf   -t Spanish --ocr llamaparse            # needs LLAMA_CLOUD_API_KEY
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
import pymupdf

# ----------------------------------------------------------------------------- data model


@dataclass
class Line:
    rect: pymupdf.Rect
    html: str            # text with <b>/<i> markup for inline style changes
    plain: str
    size: float          # font size in pt
    color: tuple         # (r, g, b) 0..1
    bold: bool
    italic: bool
    family: str          # sans-serif | serif | monospace
    erase_rects: list    # rects whose text is removed (span boxes for PDFs, line box for scans)
    group: tuple         # lines may only merge into a paragraph within the same group
    baseline: float = 0.0


@dataclass
class Block:
    id: str
    rect: pymupdf.Rect
    lines: list
    html: str
    plain: str
    size: float
    color: tuple
    bold: bool
    italic: bool
    family: str
    align: str = "left"
    line_height: float = 1.2
    cell: pymupdf.Rect | None = None
    baseline0: float = 0.0   # first and last baseline: translated text is anchored to these
    baseline1: float = 0.0
    translate: bool = True
    translation: str | None = None
    # filled in by the renderer
    scale: float = 1.0
    condensed: bool = False
    overflow: bool = False


@dataclass
class PageJob:
    index: int
    scanned: bool
    blocks: list = field(default_factory=list)
    image: np.ndarray | None = None      # scans: page image (RGB)
    px_per_pt: float = 1.0
    grid_mask: np.ndarray | None = None
    obstacles: list = field(default_factory=list)   # non-text things text must not grow into


HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
RTL_LANGS = {"arabic", "hebrew", "persian", "farsi", "urdu", "pashto", "yiddish"}


def rgb_from_int(c: int) -> tuple:
    return ((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255


def hex_color(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(max(0, min(1, v)) * 255)) for v in rgb)


def union(rects) -> pymupdf.Rect:
    r = pymupdf.Rect(rects[0])
    for x in rects[1:]:
        r |= x
    return r


def v_overlap(a, b) -> float:
    return max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def h_overlap(a, b) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


# ----------------------------------------------------------------------------- digital PDF extraction

def font_family(font: str, flags: int) -> str:
    f = font.lower()
    if flags & 8 or any(k in f for k in ("mono", "courier", "consol")):
        return "monospace"
    if flags & 4 or any(k in f for k in ("times", "serif", "georgia", "garamond", "roman", "minion", "cambria")):
        if "sans" not in f:
            return "serif"
    return "sans-serif"


def is_bold(font: str, flags: int) -> bool:
    return bool(flags & 16) or bool(re.search(r"bold|black|heavy|semibold|demi", font, re.I))


def is_italic(font: str, flags: int) -> bool:
    return bool(flags & 2) or bool(re.search(r"italic|oblique", font, re.I))


def extract_digital(page: pymupdf.Page) -> tuple[list[Line], list]:
    """Lines with style from the PDF text layer, grouped by table cell or PyMuPDF block."""
    cells = []
    try:
        for tab in page.find_tables().tables:
            cells += [pymupdf.Rect(c) for c in tab.cells if c]
    except Exception:
        pass

    raw = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP)
    lines: list[Line] = []
    for b in raw["blocks"]:
        if b.get("type") != 0:
            continue
        for ln in b["lines"]:
            dx, dy = ln["dir"]
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans or abs(dy) > 0.05 or dx < 0:   # rotated text is left untouched
                continue
            # dominant style = style of most characters
            weight = {}
            for s in spans:
                k = (round(s["size"], 1), s["color"], is_bold(s["font"], s["flags"]),
                     is_italic(s["font"], s["flags"]), font_family(s["font"], s["flags"]))
                weight[k] = weight.get(k, 0) + len(s["text"].strip())
            size, color, bold, italic, family = max(weight, key=weight.get)
            parts, plain = [], []
            for s in ln["spans"]:
                t = s["text"]
                ph = html.escape(t)
                if t.strip():
                    if is_bold(s["font"], s["flags"]) and not bold:
                        ph = f"<b>{ph}</b>"
                    if is_italic(s["font"], s["flags"]) and not italic:
                        ph = f"<i>{ph}</i>"
                parts.append(ph)
                plain.append(t)
            rect = union([pymupdf.Rect(s["bbox"]) for s in spans])
            center = pymupdf.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
            cell_i = next((i for i, c in enumerate(cells) if center in c), None)
            group = ("cell", cell_i) if cell_i is not None else ("flow",)
            lines.append(Line(rect, re.sub(r"</b><b>|</i><i>", "", "".join(parts)).strip(),
                              "".join(plain).strip(), size, rgb_from_int(color), bold, italic,
                              family, [pymupdf.Rect(s["bbox"]) for s in spans], group,
                              baseline=max(s["origin"][1] for s in spans)))
    return lines, cells


def page_obstacles(page: pymupdf.Page) -> list:
    """Images and vector drawings: text may not grow over them."""
    obs = []
    for info in page.get_image_info():
        obs.append(pymupdf.Rect(info["bbox"]))
    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"])
        if r.width < page.rect.width * 0.95 or r.height < page.rect.height * 0.95:   # skip page backgrounds
            obs.append(r)
    return obs


def is_scanned_page(page: pymupdf.Page) -> bool:
    area = abs(page.rect)
    cover = sum(abs(pymupdf.Rect(i["bbox"]) & page.rect) for i in page.get_image_info()) / area
    if cover < 0.6:
        return False
    chars = invisible = 0
    for span in page.get_texttrace():
        n = len(span["chars"])
        chars += n
        if span["type"] == 3 or span.get("opacity", 1) == 0:
            invisible += n
    return chars < 30 or invisible / max(chars, 1) > 0.8    # no text, or only an OCR layer


# ----------------------------------------------------------------------------- OCR extraction (scans / images)

def render_page(page: pymupdf.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, alpha=False, colorspace=pymupdf.csRGB)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3).copy()


def detect_grid(gray: np.ndarray) -> tuple[list, np.ndarray]:
    """Table cells from horizontal and vertical ruling lines. Returns (cells_px, line_mask)."""
    h, w = gray.shape
    binv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    hor = cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 30), 1)))
    ver = cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 60))))
    grid = cv2.dilate(hor | ver, np.ones((3, 3), np.uint8))
    cells = []
    keep = np.zeros_like(grid)      # only real rules: tall letter stems also survive the opening
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grid, 8)
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if (bw >= w * 0.1 and bh < 15) or (bh >= h * 0.05 and bw < 15):     # stand-alone rule
            keep[labels == i] = 255
        if bw < w * 0.1 or bh < 15:
            continue
        sub_h, sub_v = hor[y:y + bh, x:x + bw] > 0, ver[y:y + bh, x:x + bw] > 0
        # a rule must span most of the table; shaded bands (thick) contribute both edges
        rows = _line_positions(sub_h.sum(axis=1) > 0.6 * bw)
        cols = _line_positions(sub_v.sum(axis=0) > 0.6 * bh)
        if len(rows) < 2:
            continue
        if len(cols) < 2:                       # only row rules: each row is a cell
            cols = [0, bw - 1]
        for r0, r1 in zip(rows, rows[1:]):
            for c0, c1 in zip(cols, cols[1:]):
                if r1 - r0 > 8 and c1 - c0 > 8:
                    cells.append((x + c0, y + r0, x + c1, y + r1))
        keep[labels == i] = 255
    return cells, keep


def _line_positions(on: np.ndarray) -> list:
    pos, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            pos += [(i + j - 1) // 2] if j - i <= 8 else [i, j - 1]
            i = j
        else:
            i += 1
    return pos


def text_style_from_pixels(img: np.ndarray, box, size_px: float) -> tuple[tuple, float]:
    """(foreground colour, stroke width / font size) of the text inside a pixel box."""
    x0, y0, x1, y1 = [int(v) for v in box]
    crop = img[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return (0, 0, 0), 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = m == 0 if (m == 0).sum() < (m > 0).sum() else m > 0      # text is the minority class
    if fg.sum() < 5:
        return (0, 0, 0), 0.0
    color = tuple(np.median(crop[fg], axis=0) / 255)
    # mean stroke width = ink area / half the outline length (robust at low resolution)
    contours, _ = cv2.findContours(fg.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    stroke = 2 * fg.sum() / max(1.0, sum(cv2.arcLength(c, True) for c in contours))
    return color, stroke / max(1.0, size_px)


XHTML = "{http://www.w3.org/1999/xhtml}"
LINE_CLASSES = {"ocr_line", "ocr_header", "ocr_caption", "ocr_textfloat"}
BORDER_GLYPHS = {"|", "||", "I", "l", "[", "]", "!", "_", "—", "-"}


def _hocr_title(title: str) -> dict:
    out = {}
    for part in title.split(";"):
        k, *v = part.split()
        out[k] = [float(x) for x in v if re.match(r"^-?[\d.]+$", x)]
    return out


def ocr_tesseract(img: np.ndarray, lang: str, cells_px: list, scale: float,
                  grid: np.ndarray | None = None, region=None) -> list[Line]:
    """Tesseract hOCR: word boxes plus per-line baseline and font height (x_size).
    Words inside a table cell are grouped by cell; glyphs that are really table rules are dropped."""
    import xml.etree.ElementTree as ET
    import pytesseract
    up = 2.0 if img.shape[1] < 1800 else 1.0                 # small images: upscale for accuracy
    src = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up > 1 else img
    root = ET.fromstring(pytesseract.image_to_pdf_or_hocr(src, lang=lang, extension="hocr",
                                                          config="--oem 1 --psm 3"))
    words, meta = {}, {}
    for li, line in enumerate(root.iter(XHTML + "span")):
        if line.get("class") not in LINE_CLASSES:
            continue
        t = _hocr_title(line.get("title", ""))
        lb = t["bbox"]
        slope, off = (t.get("baseline") or [0, 0])[:2]
        size_px = (t.get("x_size") or [lb[3] - lb[1]])[0] / up
        for w in line.iter(XHTML + "span"):
            if w.get("class") != "ocrx_word":
                continue
            text = "".join(w.itertext()).strip()
            wt = _hocr_title(w.get("title", ""))
            if not text or (wt.get("x_wconf") or [100])[0] < 20:
                continue
            x0, y0, x1, y1 = (v / up for v in wt["bbox"])
            if region is not None and not region.contains(pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)):
                continue
            if grid is not None and text in BORDER_GLYPHS:
                g = grid[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
                if g.size and (g > 0).mean() > 0.25:
                    continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            cell = next((ci for ci, c in enumerate(cells_px) if c[0] <= cx <= c[2] and c[1] <= cy <= c[3]), None)
            words.setdefault((cell, li), []).append((x0, y0, x1, y1, text))
            meta[li] = ((lb[3] + off + slope * (cx * up - lb[0])) / up, size_px)
    return _words_to_lines(img, words, meta, scale)


def _words_to_lines(img, words: dict, meta: dict, scale: float) -> list[Line]:
    lines = []
    for (cell, li), ws in words.items():
        baseline_px, size_px = meta[li]
        ws.sort(key=lambda w: w[0])
        # split where Tesseract joined words across a wide gap (e.g. two columns without rules)
        runs, cur = [], [ws[0]]
        for w in ws[1:]:
            if w[0] - cur[-1][2] > 1.5 * size_px:
                runs.append(cur)
                cur = [w]
            else:
                cur.append(w)
        runs.append(cur)
        for run in runs:
            box = (min(w[0] for w in run), min(w[1] for w in run), max(w[2] for w in run), max(w[3] for w in run))
            text = " ".join(w[4] for w in run)
            color, stroke = text_style_from_pixels(img, box, size_px)
            group = ("cell", cell) if cell is not None else ("flow",)
            lines.append(Line(pymupdf.Rect(*[v / scale for v in box]), html.escape(text), text,
                              size_px / scale / 0.96, color, stroke > BOLD_STROKE, False, "sans-serif",
                              [pymupdf.Rect(*box)], group, baseline=baseline_px / scale))
    return lines


BOLD_STROKE = 0.105    # stroke width / font size: regular text ~0.08-0.09, bold ~0.12


class LlamaParseOCR:
    """LlamaParse (LlamaCloud) REST API, JSON result with item bounding boxes.

    LlamaParse returns *block-level* boxes (a paragraph, a heading, a whole table). Text and
    headings come from LlamaParse; table geometry still comes from the ruling-line grid + Tesseract,
    because a single box for a whole table cannot place text back into its cells.
    """
    BASE = os.environ.get("LLAMA_CLOUD_BASE_URL", "https://api.cloud.llamaindex.ai")

    def __init__(self, path: str):
        import requests
        self.req = requests
        self.key = os.environ.get("LLAMA_CLOUD_API_KEY")
        if not self.key:
            sys.exit("--ocr llamaparse needs LLAMA_CLOUD_API_KEY")
        self.pages = self._parse(path)

    def _parse(self, path):
        hdr = {"Authorization": f"Bearer {self.key}"}
        with open(path, "rb") as f:
            r = self.req.post(f"{self.BASE}/api/v1/parsing/upload", headers=hdr,
                              files={"file": (os.path.basename(path), f)},
                              data={"premium_mode": "true"}, timeout=300)
        r.raise_for_status()
        job = r.json()["id"]
        for _ in range(600):
            st = self.req.get(f"{self.BASE}/api/v1/parsing/job/{job}", headers=hdr, timeout=60).json()
            if st.get("status") == "SUCCESS":
                break
            if st.get("status") in ("ERROR", "CANCELED"):
                sys.exit(f"LlamaParse job failed: {st}")
            time.sleep(2)
        res = self.req.get(f"{self.BASE}/api/v1/parsing/job/{job}/result/json", headers=hdr, timeout=300)
        res.raise_for_status()
        return {p.get("page", i + 1) - 1: p for i, p in enumerate(res.json()["pages"])}

    def lines(self, index: int, img: np.ndarray, page_rect: pymupdf.Rect, scale: float,
              cells_px: list, lang: str, grid: np.ndarray) -> list[Line]:
        p, self.grid = self.pages.get(index, {}), grid
        k = page_rect.width / float(p.get("width") or page_rect.width)   # LlamaParse units -> pt
        out, table_regions = [], []
        for n, it in enumerate(p.get("items", [])):
            bb = it.get("bBox") or it.get("bbox")
            if not bb:
                continue
            r = pymupdf.Rect(bb["x"] * k, bb["y"] * k, (bb["x"] + bb["w"]) * k, (bb["y"] + bb["h"]) * k)
            if it.get("type") == "table":
                table_regions.append(r)
                continue
            text = (it.get("value") or it.get("md") or "").strip().lstrip("#").strip()
            if not text:
                continue
            nlines = max(1, round(r.height / 14))   # LlamaParse gives no line boxes: assume ~14pt pitch
            px = [v * scale for v in r]
            size = r.height / max(1, nlines) / 1.25
            color, stroke = text_style_from_pixels(img, px, size * scale)
            out.append(Line(r, html.escape(text), text, size, color, it.get("type") == "heading" or stroke > BOLD_STROKE,
                            False, "sans-serif", [pymupdf.Rect(*px)], ("llama", n), baseline=r.y0 + 0.95 * size))
        for reg in table_regions:   # tables: cell geometry from grid + Tesseract words
            out += ocr_tesseract(img, lang, cells_px, scale, self.grid, region=pymupdf.Rect(*[v * scale for v in reg]))
        return out


# ----------------------------------------------------------------------------- lines -> blocks

def same_color(a, b) -> bool:
    return max(abs(x - y) for x, y in zip(a, b)) < 0.15


def lines_to_blocks(lines: list[Line], cells: list, page_no: int, page_rect, ocr=False) -> list[Block]:
    """Merge lines into paragraphs (same group, stacked, similar size), detect alignment.
    OCR measurements are noisy, so size/colour/bold tolerances are looser for scans."""
    size_tol = 0.25 if ocr else 0.15
    # 1) merge pieces on the same baseline that a PDF writer emitted separately (justified text)
    lines = sorted(lines, key=lambda l: (l.group, round(l.rect.y0), l.rect.x0))
    merged: list[Line] = []
    for ln in lines:
        p = merged[-1] if merged else None
        if (p and p.group == ln.group and v_overlap(p.rect, ln.rect) > 0.6 * min(p.rect.height, ln.rect.height)
                and 0 <= ln.rect.x0 - p.rect.x1 < 1.5 * p.size and abs(p.size - ln.size) < 1):
            p.rect |= ln.rect
            p.html += " " + ln.html
            p.plain += " " + ln.plain
            p.erase_rects += ln.erase_rects
        else:
            merged.append(ln)

    # 2) stack lines into paragraphs
    paras: list[list[Line]] = []
    for ln in sorted(merged, key=lambda l: (l.baseline, l.rect.x0)):
        for p in reversed(paras):          # columns interleave by height: try every open paragraph
            last, prect = p[-1], union([l.rect for l in p])
            pitch = ln.baseline - last.baseline
            if len(p) >= 2:     # keep the paragraph's own line spacing; extra space = new paragraph
                ok_v = abs(pitch - (last.baseline - p[-2].baseline)) < (0.4 if ocr else 0.3) * ln.size
            else:
                ok_v = 0.6 * ln.size < pitch < 1.75 * ln.size
            same = (last.group == ln.group and abs(last.size - ln.size) <= max(0.8, size_tol * ln.size)
                    and ok_v and h_overlap(prect, ln.rect) > 0
                    and (ocr or last.bold == ln.bold) and same_color(last.color, ln.color))
            if same:
                p.append(ln)
                break
        else:
            paras.append([ln])

    blocks = []
    for i, p in enumerate(paras):
        rect = union([l.rect for l in p])
        texts, htmls = [], []
        for l in p:          # join lines, undoing end-of-line hyphenation
            if texts and texts[-1].endswith("-") and l.plain[:1].islower():
                texts[-1], htmls[-1] = texts[-1][:-1], htmls[-1][:-1]
                texts[-1] += l.plain
                htmls[-1] += l.html
            else:
                texts.append(l.plain)
                htmls.append(l.html)
        plain, markup = " ".join(texts), " ".join(htmls)
        sizes = sorted(l.size for l in p)
        size = sizes[len(sizes) // 2]
        bold = sum(l.bold for l in p) * 2 > len(p) if ocr else p[0].bold
        cell = None
        if p[0].group[0] == "cell" and p[0].group[1] is not None and p[0].group[1] < len(cells):
            cell = pymupdf.Rect(cells[p[0].group[1]])
        pitch = [b.baseline - a.baseline for a, b in zip(p, p[1:])]
        lh = float(np.median(pitch)) / size if pitch else 1.2
        blocks.append(Block(
            id=f"p{page_no}b{i}", rect=rect, lines=p, html=markup, plain=plain, size=size,
            color=p[0].color, bold=bold, italic=p[0].italic, family=p[0].family,
            align=detect_alignment(p, rect, cell, size), line_height=min(1.6, max(1.0, lh)), cell=cell,
            baseline0=p[0].baseline, baseline1=p[-1].baseline,
            translate=bool(HAS_LETTER.search(plain)) and len(plain.strip()) > 1))
    refine_single_line_alignment(blocks, page_rect)
    return blocks


def refine_single_line_alignment(blocks: list[Block], page_rect) -> None:
    """A lone line has no alignment of its own: infer it from other blocks' edges or the page centre."""
    for b in blocks:
        if len(b.lines) != 1 or b.cell is not None:
            continue
        tol = max(2, 0.4 * b.size)
        others = [o.rect for o in blocks if o is not b and o.cell is None]
        left_edge = any(abs(o.x0 - b.rect.x0) < tol for o in others)
        right_edge = any(abs(o.x1 - b.rect.x1) < tol for o in others)
        if right_edge and not left_edge:
            b.align = "right"
        elif not left_edge and abs((b.rect.x0 + b.rect.x1) / 2 - (page_rect.x0 + page_rect.x1) / 2) < tol:
            b.align = "center"


def detect_alignment(p: list[Line], rect, cell, size) -> str:
    tol = max(1.5, 0.35 * size)
    if len(p) == 1:
        ref = cell if cell is not None else None
        if ref is None:
            return "left"
        lpad, rpad = rect.x0 - ref.x0, ref.x1 - rect.x1
        if abs(lpad - rpad) < tol and lpad > 2 * tol:
            return "center"
        if rpad < lpad - 2 * tol and rpad < 3 * tol:
            return "right"
        return "left"
    lefts = [l.rect.x0 for l in p]
    rights = [l.rect.x1 for l in p]
    mids = [(l.rect.x0 + l.rect.x1) / 2 for l in p]
    left_ok = all(abs(x - rect.x0) < tol for x in lefts[1:])
    right_ok = all(abs(x - rect.x1) < tol for x in rights[:-1])
    if left_ok and right_ok and len(p) >= 3:
        return "justify"
    if left_ok:
        return "left"
    if all(abs(x - rect.x1) < tol for x in rights):
        return "right"
    if max(mids) - min(mids) < tol:
        return "center"
    return "left"


# ----------------------------------------------------------------------------- translators

SYSTEM_PROMPT = """You are a professional document translator. You receive text blocks extracted from one \
page of a document and translate each block into {lang}.

Rules:
- Return every block id exactly once. Translate the meaning, using the context of the whole page.
- Keep the inline tags <b>, </b>, <i>, </i> around the corresponding translated words. Do not add other markup.
- Keep numbers, currency amounts, dates, codes, URLs, e-mail addresses, product names and placeholders as they are \
(only adapt number formatting if the target locale requires it).
- Table cells, headings and labels are short: keep them short. Prefer concise phrasing; the translation must fit \
in roughly the same space as the original.
- If a block is already in {lang}, or is a proper name, return it unchanged.
{glossary}"""

OCR_RULES = """
These blocks come from OCR of the attached page image and can contain recognition errors. For each block, read the \
image at that position and give the corrected original text in "source", then translate the corrected text. Set \
"bold" to true if the block is printed in bold."""


class Cache:
    def __init__(self, path: str | None):
        self.path, self.lock = path, threading.Lock()
        self.data = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.data = json.load(f)

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:32]

    def get(self, k):
        return self.data.get(k)

    def put(self, k, v):
        with self.lock:
            self.data[k] = v

    def save(self):
        if self.path:
            with self.lock, open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False)


class ClaudeTranslator:
    name = "claude"

    def __init__(self, lang: str, model: str, effort: str, glossary: str, cache: Cache):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=5)
        self.lang, self.model, self.effort, self.cache = lang, model, effort, cache
        self.system = SYSTEM_PROMPT.format(lang=lang, glossary=f"\nGlossary (always use):\n{glossary}" if glossary else "")

    def _call(self, content: list, schema: dict, system: str) -> dict:
        kwargs = dict(model=self.model, max_tokens=32000, system=system,
                      messages=[{"role": "user", "content": content}],
                      output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}})
        if self.model in ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"):
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        with self.client.beta.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"model declined: {getattr(msg.stop_details, 'explanation', '')}")
        if msg.stop_reason == "max_tokens":
            raise RuntimeError("response truncated (max_tokens) - lower --chunk-chars")
        return json.loads(next(b.text for b in msg.content if b.type == "text"))

    def translate(self, items: list[dict], context: str, image_jpeg: bytes | None = None) -> dict:
        """items: [{id, text, role}] -> {id: {"translation", "source"?, "bold"?}}"""
        out, todo = {}, []
        for it in items:
            hit = self.cache.get(Cache.key(self.model, self.lang, "ocr" if image_jpeg else "t", it["text"]))
            (out.__setitem__(it["id"], hit) if hit else todo.append(it))
        if not todo:
            return out
        props = {"id": {"type": "string"}, "translation": {"type": "string"}}
        if image_jpeg:
            props |= {"source": {"type": "string"}, "bold": {"type": "boolean"}}
        schema = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {
            "items": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                  "required": list(props), "properties": props}}}}
        content = []
        if image_jpeg:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                         "data": base64.standard_b64encode(image_jpeg).decode()}})
        content.append({"type": "text", "text":
                        f"Page text in reading order, for context:\n<page>\n{context}\n</page>\n\n"
                        f"Translate these blocks into {self.lang}:\n{json.dumps(todo, ensure_ascii=False)}"})
        res = self._call(content, schema, self.system + (OCR_RULES if image_jpeg else ""))
        for r in res.get("items", []):
            src = next((t for t in todo if t["id"] == r["id"]), None)
            if src:
                val = {k: r[k] for k in props if k != "id"}
                out[r["id"]] = val
                self.cache.put(Cache.key(self.model, self.lang, "ocr" if image_jpeg else "t", src["text"]), val)
        return out

    def condense(self, items: list[dict]) -> dict:
        """items: [{id, text, max_chars}] -> {id: shorter text}"""
        schema = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {
            "items": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                  "required": ["id", "text"],
                                                  "properties": {"id": {"type": "string"}, "text": {"type": "string"}}}}}}
        res = self._call([{"type": "text", "text":
                           "Shorten each text so it has at most max_chars characters. Keep the meaning, "
                           "the language, the <b>/<i> tags, numbers and names; use abbreviations common in "
                           f"{self.lang} documents if needed.\n" + json.dumps(items, ensure_ascii=False)}],
                         schema, f"You edit {self.lang} document text to fit a fixed space.")
        return {r["id"]: r["text"] for r in res.get("items", [])}


class PseudoTranslator:
    """Offline stand-in: accents letters and makes text ~30% longer, like EN->DE expansion.
    Exercises the whole pipeline (layout, erasing, fitting) without an API key."""
    name = "pseudo"
    MAP = str.maketrans("aeiouAEIOUcnyC", "àéîöüÀÉÎÖÜçñýÇ")

    def __init__(self, *_, **__):
        pass

    def _word(self, w: str) -> str:
        if not HAS_LETTER.search(w) or any(ch.isdigit() for ch in w) or "@" in w or "/" in w:
            return w
        v = re.search(r"[aeiouAEIOU]", w)
        if v and len(w) > 3:
            w = w[:v.end()] + w[v.start():]                  # lengthen: duplicate first vowel
        return w.translate(self.MAP) + ("ën" if len(w) > 6 else "")

    def _text(self, s: str) -> str:
        return "".join(p if p.startswith("<") else re.sub(r"[^\s<>]+", lambda m: self._word(m.group()), p)
                       for p in re.split(r"(<[^>]+>)", s))

    def translate(self, items, context, image_jpeg=None):
        return {it["id"]: {"translation": self._text(it["text"])} for it in items}

    def condense(self, items):
        out = {}
        for it in items:
            words, s = it["text"].split(), ""
            for w in words:
                if len(s) + len(w) + 1 > it["max_chars"]:
                    break
                s = f"{s} {w}".strip()
            out[it["id"]] = s or it["text"][: it["max_chars"]]
        return out


# ----------------------------------------------------------------------------- erasing original text

def erase_text_pixels(img: np.ndarray, blocks: list[Block], grid_mask: np.ndarray | None) -> np.ndarray:
    """Remove only the text pixels of translated blocks; keep backgrounds and table rules."""
    out = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for b in blocks:
        if not b.translate:
            continue
        for ln in b.lines:
            for r in ln.erase_rects:
                x0, y0 = max(0, int(r.x0) - 3), max(0, int(r.y0) - 3)
                x1, y1 = min(w, int(r.x1) + 3), min(h, int(r.y1) + 3)
                crop = img[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                bg = np.median(crop.reshape(-1, 3), axis=0)       # a text line is mostly background
                diff = np.abs(crop.astype(np.int16) - bg).max(axis=2)
                m = (diff > 40).astype(np.uint8) * 255
                k = max(3, int(0.12 * (y1 - y0)))                 # bigger text -> wider anti-aliased fringe
                m = cv2.dilate(m, np.ones((k, k), np.uint8))
                if grid_mask is not None:
                    m[grid_mask[y0:y1, x0:x1] > 0] = 0
                rest = crop[m == 0]
                if len(rest) and rest.std(axis=0).max() < 14:    # flat background: plain fill is cleanest
                    out[y0:y1, x0:x1][m > 0] = np.median(rest, axis=0).astype(np.uint8)
                else:                                             # textured background: inpaint
                    mask[y0:y1, x0:x1] |= m
    if mask.any():
        out = cv2.inpaint(out, mask, 5, cv2.INPAINT_TELEA)
    return out


# ----------------------------------------------------------------------------- fitting & rendering

class Fitter:
    def __init__(self, min_scale: float, rtl: bool):
        self.min_scale, self.rtl = min_scale, rtl
        self.scratch = pymupdf.open()
        self.calls = 0

    def css(self, b: Block) -> str:
        align = "right" if self.rtl and b.align == "left" else b.align
        return ("body, p {margin:0; padding:0;} "
                f"body {{font-family:{b.family}; font-size:{b.size:.2f}px; line-height:{b.line_height:.2f}; "
                f"color:{hex_color(b.color)}; text-align:{align}; {'direction:rtl;' if self.rtl else ''}"
                f"font-weight:{'bold' if b.bold else 'normal'}; font-style:{'italic' if b.italic else 'normal'};}}")

    def html(self, b: Block) -> str:
        return f"<p>{b.translation}</p>"

    def trial(self, rect, b: Block, scale_low: float):
        if self.calls % 200 == 0:
            self.scratch = pymupdf.open()
            self.scratch.new_page(width=4000, height=4000)
        self.calls += 1
        try:
            return self.scratch[0].insert_htmlbox(rect, self.html(b), css=self.css(b), scale_low=scale_low)
        except Exception:
            return (-1, 0)

    def candidates(self, b: Block, others: list, obstacles: list, page_rect) -> list:
        """Original box, then box grown into free space below, then also sideways."""
        # the original line boxes: same first baseline and same number of lines as the source
        s, lh = b.size, b.line_height
        half = (lh - 1) * s / 2
        r = pymupdf.Rect(b.rect.x0 - 0.5, b.baseline0 - 0.8 * s - half,
                         b.rect.x1 + 0.5, b.baseline1 + 0.2 * s + half + 0.5)
        if b.cell is not None:      # inside a table cell: never leave the cell
            c = pymupdf.Rect(b.cell.x0 + 1.5, b.cell.y0 + 1, b.cell.x1 - 1.5, b.cell.y1 - 1)
            r0 = pymupdf.Rect(c.x0, max(r.y0, c.y0), c.x1, min(r.y1, c.y1))
            return [r0, pymupdf.Rect(c.x0, r0.y0, c.x1, c.y1), pymupdf.Rect(c)]
        margin = 14
        blockers = [o.rect for o in others if o is not b] + [o for o in obstacles if not o.intersects(b.rect)]
        # grow down
        below = [o.y0 for o in blockers if h_overlap(o, r) > 1 and o.y0 >= r.y1 - 1]
        down = pymupdf.Rect(r.x0, r.y0, r.x1, min(below + [page_rect.y1 - margin]) - 0.25 * b.size)
        down.y1 = max(down.y1, r.y1)
        # grow sideways inside the vertical band of `down`
        band = [o for o in blockers if v_overlap(o, down) > 1]
        right = min([o.x0 for o in band if o.x0 >= r.x1 - 1] + [page_rect.x1 - margin]) - 0.6 * b.size
        left = max([o.x1 for o in band if o.x1 <= r.x0 + 1] + [page_rect.x0 + margin]) + 0.6 * b.size
        side = pymupdf.Rect(down)
        if b.align in ("left", "justify"):
            side.x1 = max(side.x1, right)
        elif b.align == "right":
            side.x0 = min(side.x0, left)
        else:   # centre: grow symmetrically
            grow = max(0, min(right - r.x1, r.x0 - left))
            side.x0, side.x1 = r.x0 - grow, r.x1 + grow
        if len(b.lines) == 1:   # a single line (heading, label) should rather get wider than wrap
            wide = pymupdf.Rect(side.x0, r.y0, side.x1, r.y1)
            return [r, wide, side]
        return [r, down, side]

    def fit(self, b: Block, others, obstacles, page_rect):
        """Returns (rect, scale_low, scale) of the first option that fits, or None."""
        cands = self.candidates(b, others, obstacles, page_rect)
        for rect in cands:
            spare, _ = self.trial(rect, b, 1.0)
            if spare >= 0:
                return rect, 1.0, 1.0, spare
        spare, scale = self.trial(cands[-1], b, self.min_scale)
        if spare >= 0:
            return cands[-1], self.min_scale, scale, spare
        return None

    def place(self, page, b: Block, rect, scale_low, spare):
        if b.cell is not None and spare > 0:     # keep text vertically centred in table cells
            used = rect.height - spare
            cy = (b.rect.y0 + b.rect.y1) / 2
            y0 = min(max(rect.y0, cy - used / 2), rect.y1 - used)
            rect = pymupdf.Rect(rect.x0, y0, rect.x1, rect.y1)
        page.insert_htmlbox(rect, self.html(b), css=self.css(b), scale_low=scale_low)


# ----------------------------------------------------------------------------- pipeline

def clean_markup(s: str) -> str:
    """Allow only <b>/<i> from the model; escape everything else."""
    s = html.escape(html.unescape(s), quote=False)
    return re.sub(r"&lt;(/?)(b|i)&gt;", r"<\1\2>", s)


def page_context(blocks: list[Block]) -> str:
    return "\n".join(b.plain for b in sorted(blocks, key=lambda b: (b.rect.y0, b.rect.x0)))


def translate_page(job: PageJob, tr, chunk_chars: int, ocr_refine: bool) -> None:
    items = [{"id": b.id, "text": b.html, "role": "table cell" if b.cell is not None else
              ("heading" if b.bold or b.size > 13 else "text")} for b in job.blocks if b.translate]
    if not items:
        return
    jpeg = None
    if job.scanned and ocr_refine and tr.name == "claude":
        img = job.image
        k = 2000 / max(img.shape[:2])
        if k < 1:
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        jpeg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
    ctx = page_context(job.blocks)
    chunks, cur, n = [], [], 0
    for it in items:
        if cur and n + len(it["text"]) > chunk_chars:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(it)
        n += len(it["text"])
    chunks.append(cur)
    result = {}
    for ch in chunks:
        result |= tr.translate(ch, ctx, jpeg)
        missing = [it for it in ch if it["id"] not in result]
        if missing:                       # one retry for anything the model skipped
            result |= tr.translate(missing, ctx, jpeg)
    by_id = {b.id: b for b in job.blocks}
    for bid, r in result.items():
        b = by_id.get(bid)
        if not b:
            continue
        b.translation = clean_markup(r.get("translation") or b.html)
        if r.get("bold") is not None and job.scanned:
            b.bold = bool(r["bold"])
    for b in job.blocks:
        if b.translate and b.translation is None:
            print(f"  ! page {job.index + 1}: no translation for {b.id}, keeping original", file=sys.stderr)
            b.translate = False


def render_blocks(page, job: PageJob, fitter: Fitter, tr) -> None:
    todo = [b for b in job.blocks if b.translate]
    results = {}
    failed = []
    for b in todo:
        r = fitter.fit(b, job.blocks, job.obstacles, page.rect)
        (results.__setitem__(b.id, r) if r else failed.append(b))
    if failed:   # step 5: ask for shorter translations, all failed blocks of the page in one call
        req = [{"id": b.id, "text": b.translation,
                "max_chars": max(4, int(len(re.sub(r"<[^>]+>", "", b.translation)) * 0.72))} for b in failed]
        try:
            short = tr.condense(req)
        except Exception as e:
            print(f"  ! condense failed: {e}", file=sys.stderr)
            short = {}
        for b in failed:
            if b.id in short:
                b.translation, b.condensed = clean_markup(short[b.id]), True
            r = fitter.fit(b, job.blocks, job.obstacles, page.rect)
            if r is None:    # step 6: shrink without a floor
                rect = fitter.candidates(b, job.blocks, job.obstacles, page.rect)[-1]
                spare, scale = fitter.trial(rect, b, 0)
                r, b.overflow = (rect, 0, scale, max(spare, 0)), True
            results[b.id] = r
    for b in todo:
        rect, scale_low, scale, spare = results[b.id]
        b.scale = scale
        fitter.place(page, b, rect, scale_low, spare)


def remove_digital_text(page, blocks: list[Block]) -> None:
    for b in blocks:
        if not b.translate:
            continue
        for ln in b.lines:
            for r in ln.erase_rects:
                h = r.height
                page.add_redact_annot(pymupdf.Rect(r.x0, r.y0 + 0.2 * h, r.x1, r.y1 - 0.2 * h), fill=False)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                          graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                          text=pymupdf.PDF_REDACT_TEXT_REMOVE)


def prepare_page(doc, i: int, args, llama) -> PageJob:
    page = doc[i]
    scanned = args.force_ocr or is_scanned_page(page)
    job = PageJob(i, scanned)
    if not scanned:
        lines, cells = extract_digital(page)
        job.blocks = lines_to_blocks(lines, cells, i + 1, page.rect)
        job.obstacles = page_obstacles(page)
        return job
    img = render_page(page, args.dpi)
    scale = img.shape[1] / page.rect.width              # px per pt
    cells_px, grid = detect_grid(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    if llama:
        lines = llama.lines(i, img, page.rect, scale, cells_px, args.ocr_lang, grid)
    else:
        lines = ocr_tesseract(img, args.ocr_lang, cells_px, scale, grid)
    cells_pt = [pymupdf.Rect(*[v / scale for v in c]) for c in cells_px]
    job.blocks = lines_to_blocks(lines, cells_pt, i + 1, page.rect, ocr=True)
    job.image, job.px_per_pt, job.grid_mask = img, scale, grid
    n, _, stats, _ = cv2.connectedComponentsWithStats(grid, 8)       # table rules are obstacles
    job.obstacles = [pymupdf.Rect(x / scale, y / scale, (x + w) / scale, (y + h) / scale)
                     for x, y, w, h, _ in stats[1:]]
    return job


def finish_scanned_page(page, job: PageJob) -> None:
    clean = erase_text_pixels(job.image, job.blocks, job.grid_mask)
    page.add_redact_annot(page.rect)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_REMOVE,
                          graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
                          text=pymupdf.PDF_REDACT_TEXT_REMOVE)
    jpg = cv2.imencode(".jpg", cv2.cvtColor(clean, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])[1]
    page.insert_image(page.rect, stream=jpg.tobytes())


def draw_debug(doc_path: str, jobs: list[PageJob], out_path: str) -> None:
    dbg = pymupdf.open(doc_path)
    for job in jobs:
        page = dbg[job.index]
        for b in job.blocks:
            col = (0.6, 0.6, 0.6) if not b.translate else (0, 0.6, 0) if b.cell is not None else (0, 0.3, 1)
            page.draw_rect(b.rect, color=col, width=0.6)
            page.insert_text(b.rect.tl + (0, -1), f"{b.id} {b.align[0]}", fontsize=4, color=(1, 0, 0))
    dbg.save(out_path)


def image_to_pdf(path: str, dpi: int) -> tuple[pymupdf.Document, float]:
    """Wrap an image in a 1-page PDF where 1 px = 72/dpi pt, so pt <-> px is exact."""
    pix = pymupdf.Pixmap(path)
    doc = pymupdf.open()
    k = 72 / dpi
    page = doc.new_page(width=pix.width * k, height=pix.height * k)
    page.insert_image(page.rect, filename=path)
    return doc, dpi


def main():
    ap = argparse.ArgumentParser(description="Translate PDFs/images with an LLM, keeping the layout.")
    ap.add_argument("input", help=".pdf, .png, .jpg, .jpeg, .tif, .tiff, .bmp, .webp")
    ap.add_argument("-t", "--target", required=True, help='target language, e.g. "German"')
    ap.add_argument("-o", "--output", help="output path (default: <input>.<lang>.<ext>)")
    ap.add_argument("--translator", choices=["claude", "pseudo"], default="claude")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--ocr", choices=["tesseract", "llamaparse"], default="tesseract",
                    help="OCR backend for scanned pages and images")
    ap.add_argument("--ocr-lang", default="eng", help="Tesseract language(s) of the SOURCE, e.g. eng, deu+eng")
    ap.add_argument("--no-ocr-refine", action="store_true", help="don't send page images to Claude to fix OCR")
    ap.add_argument("--force-ocr", action="store_true", help="treat every page as scanned")
    ap.add_argument("--dpi", type=int, default=300, help="render DPI for scanned PDF pages")
    ap.add_argument("--min-scale", type=float, default=0.70, help="smallest font scale before condensing")
    ap.add_argument("--glossary", help="file with lines 'source term = target term'")
    ap.add_argument("--pages", help="1-based pages, e.g. 1-3,7")
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM requests")
    ap.add_argument("--chunk-chars", type=int, default=6000, help="max source chars per LLM request")
    ap.add_argument("--debug", action="store_true", help="also write a PDF showing the detected blocks")
    args = ap.parse_args()

    src = args.input
    ext = os.path.splitext(src)[1].lower()
    is_image = ext != ".pdf"
    slug = re.sub(r"\W+", "_", args.target.lower()).strip("_")
    out = args.output or f"{os.path.splitext(src)[0]}.{slug}{ext}"

    if is_image:
        doc, img_dpi = image_to_pdf(src, 150)
        args.dpi, args.force_ocr = img_dpi, True
        work_pdf = os.path.splitext(out)[0] + ".work.pdf"
        doc.save(work_pdf)
        doc = pymupdf.open(work_pdf)
    else:
        doc = pymupdf.open(src)
        work_pdf = src
    for p in doc:
        if p.rotation:
            p.remove_rotation()

    pages = list(range(len(doc)))
    if args.pages:
        pages = sorted({i - 1 for part in args.pages.split(",")
                        for i in (range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))
                        if 0 < i <= len(doc)})

    cache = Cache(os.path.splitext(out)[0] + ".cache.json" if args.translator == "claude" else None)
    glossary = open(args.glossary, encoding="utf-8").read() if args.glossary else ""
    tr = (ClaudeTranslator(args.target, args.model, args.effort, glossary, cache)
          if args.translator == "claude" else PseudoTranslator())
    llama = LlamaParseOCR(src) if args.ocr == "llamaparse" else None
    fitter = Fitter(args.min_scale, args.target.lower().split()[-1] in RTL_LANGS)

    t0 = time.time()
    jobs = []
    batch = max(1, args.workers * 2)        # bounded memory: page images live only within a batch
    for s in range(0, len(pages), batch):
        part = pages[s:s + batch]
        # PyMuPDF is not thread-safe: extraction and rendering stay on this thread, LLM calls run in parallel
        part_jobs = [prepare_page(doc, i, args, llama) for i in part]
        try:
            with ThreadPoolExecutor(args.workers) as pool:
                list(pool.map(lambda j: translate_page(j, tr, args.chunk_chars, not args.no_ocr_refine), part_jobs))
        except Exception as e:
            if type(e).__name__ in ("AuthenticationError", "PermissionDeniedError"):
                sys.exit(f"Claude API rejected the credentials ({e}). Set ANTHROPIC_API_KEY, "
                         "or run with --translator pseudo for an offline dry run.")
            raise
        for job in part_jobs:
            page = doc[job.index]
            if job.scanned:
                finish_scanned_page(page, job)
            else:
                remove_digital_text(page, job.blocks)
            render_blocks(page, job, fitter, tr)
            job.image = job.grid_mask = None
            n = sum(b.translate for b in job.blocks)
            print(f"page {job.index + 1}/{len(doc)}: {'scan' if job.scanned else 'digital'}, "
                  f"{n} blocks translated, {sum(b.scale < 0.999 for b in job.blocks if b.translate)} shrunk, "
                  f"{sum(b.condensed for b in job.blocks)} condensed, {sum(b.overflow for b in job.blocks)} overflow")
        jobs += part_jobs
        cache.save()

    if is_image:
        page = doc[0]
        pix = page.get_pixmap(dpi=args.dpi, alpha=False)
        orig = pymupdf.Pixmap(src)
        if (pix.width, pix.height) != (orig.width, orig.height):
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
            arr = cv2.resize(arr, (orig.width, orig.height), interpolation=cv2.INTER_CUBIC)
            pix = pymupdf.Pixmap(pymupdf.csRGB, orig.width, orig.height, arr.tobytes(), False)
        if ext in (".jpg", ".jpeg"):
            pix.save(out, jpg_quality=92)
        elif ext in (".png", ".pnm", ".pgm", ".ppm", ".pbm", ".pam", ".psd", ".ps"):
            pix.save(out)
        else:   # tif, bmp, webp ...
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
            cv2.imwrite(out, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        if args.debug:
            draw_debug(work_pdf, jobs, os.path.splitext(out)[0] + ".layout.pdf")
        doc.close()
        os.remove(work_pdf)
    else:
        doc.subset_fonts()
        doc.ez_save(out)
        if args.debug:
            draw_debug(src, jobs, os.path.splitext(out)[0] + ".layout.pdf")

    report = {"input": src, "output": out, "target": args.target, "translator": tr.name,
              "seconds": round(time.time() - t0, 1), "pages": [
                  {"page": j.index + 1, "scanned": j.scanned, "blocks": [
                      {"id": b.id, "rect": [round(v, 1) for v in b.rect], "align": b.align,
                       "cell": b.cell is not None, "source": b.plain, "translation": b.translation,
                       "scale": round(b.scale, 2), "condensed": b.condensed, "overflow": b.overflow}
                      for b in j.blocks if b.translate]} for j in jobs]}
    with open(os.path.splitext(out)[0] + ".report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"done in {report['seconds']}s -> {out}")


if __name__ == "__main__":
    main()
