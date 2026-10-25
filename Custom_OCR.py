"""
Custom OCR for lab reports: YOLO finds the fields, Tesseract reads them.

    python Custom_OCR.py --image <yourimage.jpg>
    python Custom_OCR.py --image report.pdf                   # PDFs too: every page is read
    python Custom_OCR.py --image samples/                     # every image / PDF in a folder
    python Custom_OCR.py --image x.jpg --mode brief           # the brief's exact preprocessing
    python Custom_OCR.py --image x.jpg --debug                # also save every preprocessing step

Pipeline (one line per stage)
-----------------------------
0. LOAD     a photo is one page; a PDF is rendered page by page (scans at their native
            resolution; digital PDFs at a DPI that matches the training text size, keeping
            their text layer).
1. DETECT   YOLO returns boxes for 4 field types: test_name, value, unit, reference_range.
2. CROP     each box is cut out (plus a few pixels of margin) and stored in a list.
3. PREPROCESS  resize 3x -> grayscale -> Gaussian blur -> Otsu threshold (white text on
            black) -> bitwise_not (black text on white, which Tesseract reads best).
4. READ     Tesseract turns every cleaned crop into text (a digital PDF's own text layer is
            used instead wherever it covers the box: exact, no OCR errors).
5. ASSEMBLE the page tilt is estimated, then values, units and ranges are attached to the
            test name on the same (un-tilted) row, or, for the vitamin layout, the range
            printed underneath it.
6. SAVE     one CSV row per test, with confidence scores and a needs_review flag.

Two preprocessing modes are provided:
* brief  - exactly the steps in the project brief (3x resize, 5x5 blur, Otsu, invert, psm 6).
* tuned  - (default) the same steps with settings chosen by ablation on the training and
           validation photos (cubic 3x resize, no blur, Otsu + invert, 10-px white border,
           digits-only whitelist for values, 1-px crop margin) plus a generic clinical-unit
           lexicon for units.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract

ROOT = Path(__file__).resolve().parent
CLASSES = ["test_name", "value", "unit", "reference_range"]
COLORS = {"test_name": (214, 120, 42), "value": (36, 52, 235), "unit": (50, 160, 30),
          "reference_range": (180, 40, 170)}          # BGR, colour-blind-safe-ish set
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DOC_EXT = IMG_EXT | {".pdf"}
MAX_PDF_PAGES = 20


def find_tesseract() -> str | None:
    """Point pytesseract at the engine when it is installed but not on PATH (the usual
    Windows case). Order: $TESSERACT_CMD, PATH, then the standard install folders."""
    import os
    import shutil
    for cand in (os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"),
                 r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                 r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                 str(Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"),
                 "/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/usr/bin/tesseract"):
        if cand and Path(cand).is_file():
            pytesseract.pytesseract.tesseract_cmd = cand
            return cand
    return None


find_tesseract()


# --------------------------------------------------------------------------- settings
@dataclass
class PreCfg:
    """Every knob of the preprocessing + Tesseract stage, so experiments are reproducible."""
    scale: float = 3.0              # brief: blow the crop up 3x
    interp: int = cv2.INTER_LINEAR  # cv2.resize default
    blur: int = 5                   # Gaussian kernel size; 0 = no blur
    binarize: bool = True           # Otsu threshold + bitwise_not
    border: int = 0                 # white margin (px) added around the cleaned crop
    psm_single: int = 6             # Tesseract page-segmentation mode for one-line fields
    psm_multi: int = 6              # ... for multi-line fields (vitamin reference range)
    value_whitelist: bool = False   # restrict the 'value' field to digits . < >
    pad: int = 3                    # px of context kept around the YOLO box when cropping
    unit_lexicon: bool = False      # snap OCR'd units to a generic clinical-unit vocabulary
    name: str = "custom"


BRIEF = PreCfg(name="brief")
# Chosen by tools/tune_preprocessing.py (one-knob-at-a-time ablation on the 86 train+val photos;
# the 14 test photos were never used): cubic resize, no blur, 10-px white border, digits-only
# whitelist for values, 1-px crop margin. Exact match on tuning photos (oracle boxes):
# brief 67.9% -> tuned 72.3%.
# The unit lexicon is a separate, optional post-processing step (scored with and without).
TUNED = PreCfg(scale=3.0, interp=cv2.INTER_CUBIC, blur=0, binarize=True, border=10,
               psm_single=6, psm_multi=6, value_whitelist=True, pad=1, unit_lexicon=True, name="tuned")
MODES = {"brief": BRIEF, "tuned": TUNED}


# --------------------------------------------------------------------------- stage 0: load (photo or PDF)
@dataclass
class Page:
    """One image to read. `words` is the PDF text layer in this image's pixel coordinates
    ([x0, y0, x1, y1, text], digital PDFs only); None for photos and scanned PDFs."""
    name: str
    img: np.ndarray
    page: int = 1
    words: list | None = None
    kind: str = "image"             # image | pdf-scan | pdf-digital


def load_pages(data: bytes, name: str) -> list[Page]:
    """Decode an uploaded/read file into pages: a photo is one page, a PDF one page per page."""
    if data[:5] == b"%PDF-":
        return pdf_pages(data, name)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a readable image or PDF")
    return [Page(name, img)]


def pdf_pages(data: bytes, name: str, max_pages: int = MAX_PDF_PAGES) -> list[Page]:
    """Render each PDF page at the resolution the detector expects.

    Resolution: if one image covers most of the page (a scan or photo), render at that image's
    own resolution, so the pixels match the photos YOLO was trained on. Otherwise pick the DPI
    that makes body text ~13 px tall, the text size of the training photos.

    Text layer: used only if the page has VISIBLE text and no invisible text. Invisible text
    (render mode 3) is a scanner's or Tesseract's page-level OCR layer, which is usually worse
    than this pipeline's field-level OCR, so such pages are OCR'd as scans.
    """
    import pymupdf
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:                      # corrupt / encrypted
        raise ValueError(f"cannot open PDF: {e}") from e
    if doc.needs_pass:
        raise ValueError("PDF is password-protected")
    pages = []
    stem = Path(name).stem
    for i, pg in enumerate(doc):
        if i >= max_pages:
            break
        area = pg.rect.width * pg.rect.height
        imgs = [im for im in pg.get_image_info() if im.get("width")]
        big = max(imgs, key=lambda im: abs(pymupdf.Rect(im["bbox"]))) if imgs else None
        image_page = big is not None and abs(pymupdf.Rect(big["bbox"]) & pg.rect) >= 0.6 * area
        trace = pg.get_texttrace()
        invisible = any(t["type"] == 3 or t["opacity"] == 0 for t in trace)
        use_text = not invisible and any(t["type"] != 3 and t["opacity"] > 0 for t in trace)
        if image_page:
            box = pymupdf.Rect(big["bbox"])
            dpi = big["width"] / max(box.width, 1) * 72              # the image's native resolution
        elif use_text:
            sizes = [s["size"] for b in pg.get_text("dict")["blocks"] for l in b.get("lines", [])
                     for s in l["spans"] if s["text"].strip()]
            dpi = 13 * 72 / float(np.median(sizes)) if sizes else 150
        else:
            dpi = 150
        s = float(np.clip(dpi, 72, 300)) / 72                       # PDF points -> pixels
        pix = pg.get_pixmap(matrix=pymupdf.Matrix(s, s), alpha=False)  # (dpi= only takes integers)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_GRAY2BGR)
        words = [[w[0] * s, w[1] * s, w[2] * s, w[3] * s, w[4]] for w in pg.get_text("words")] if use_text else None
        pages.append(Page(f"{stem}_p{i + 1}", img, i + 1, words or None,
                          "pdf-digital" if words else "pdf-scan"))
    if not pages:
        raise ValueError("PDF has no pages")
    return pages


def text_layer_read(words: list, box, pad: float = 2.0) -> str:
    """Words of a digital PDF whose centre falls inside the box, in reading order."""
    x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    inside = [w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
    lines: list[list] = []
    for w in sorted(inside, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        h = w[3] - w[1]
        if lines and abs((lines[-1][0][1] + lines[-1][0][3]) / 2 - (w[1] + w[3]) / 2) < 0.5 * h:
            lines[-1].append(w)
        else:
            lines.append([w])
    return " | ".join(" ".join(t[4] for t in sorted(l, key=lambda t: t[0])) for l in lines)


# --------------------------------------------------------------------------- stage 1: detect
def load_model(weights: str | Path = ROOT / "model" / "best.pt"):
    from ultralytics import YOLO          # imported here so OCR-only helpers work without torch
    return YOLO(str(weights))


def detect(model, img: np.ndarray, conf: float = 0.25, imgsz: int | None = None) -> list[dict]:
    """Run YOLO and return [{'cls','conf','box':[x0,y0,x1,y1]}] in pixel coordinates.
    Default input size is 1024 (training size), raised for large rendered PDF pages so their
    text is not shrunk below the size the detector learned (max 1600)."""
    if imgsz is None:              # photos (<= 1280 px) keep exactly the evaluated setting
        side = max(img.shape[:2])
        imgsz = 1024 if side <= 1280 else int(min(np.ceil(side / 32) * 32, 1600))
    res = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    out = []
    for b in res.boxes:
        x0, y0, x1, y1 = (float(v) for v in b.xyxy[0].tolist())
        out.append({"cls": CLASSES[int(b.cls)], "conf": float(b.conf), "box": [x0, y0, x1, y1]})
    return dedupe(out)


def dedupe(dets: list[dict], iou_thr: float = 0.5, contain_thr: float = 0.7) -> list[dict]:
    """YOLO26 is NMS-free, so it can (rarely) return two boxes for one field, e.g. a full and
    a partial test name. Keep the more confident box when two same-class boxes overlap a lot."""
    kept = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        dup = False
        for k in kept:
            if k["cls"] != d["cls"]:
                continue
            inter = _inter(k["box"], d["box"])
            a1, a2 = _area(k["box"]), _area(d["box"])
            if inter / (a1 + a2 - inter + 1e-9) > iou_thr or inter / (min(a1, a2) + 1e-9) > contain_thr:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


# --------------------------------------------------------------------------- stage 2: crop
def crop(img: np.ndarray, box, pad: int = 3) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0 = max(int(np.floor(x0)) - pad, 0), max(int(np.floor(y0)) - pad, 0)
    x1, y1 = min(int(np.ceil(x1)) + pad, w), min(int(np.ceil(y1)) + pad, h)
    return img[y0:y1, x0:x1]


# --------------------------------------------------------------------------- stage 3: preprocess
def preprocess(roi: np.ndarray, cfg: PreCfg = TUNED, steps: dict | None = None) -> np.ndarray:
    """Resize -> gray -> blur -> Otsu (white-on-black) -> bitwise_not (black-on-white)."""
    big = cv2.resize(roi, None, fx=cfg.scale, fy=cfg.scale, interpolation=cfg.interp)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY) if big.ndim == 3 else big
    blur = cv2.GaussianBlur(gray, (cfg.blur, cfg.blur), 0) if cfg.blur else gray
    if cfg.binarize:
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        out = cv2.bitwise_not(thresh)
    else:
        thresh, out = None, blur
    if cfg.border:
        out = cv2.copyMakeBorder(out, cfg.border, cfg.border, cfg.border, cfg.border,
                                 cv2.BORDER_CONSTANT, value=255)
    if steps is not None:                       # keep every stage for the debug figure
        steps.update({"1_crop": roi, "2_resized": big, "3_gray": gray, "4_blur": blur,
                      "5_otsu_inv": thresh if thresh is not None else blur, "6_final": out})
    return out


# --------------------------------------------------------------------------- stage 4: read
def is_multiline(roi: np.ndarray, line_px: float) -> bool:
    """A box more than ~2.2 text-lines tall is read as a block (vitamin D reference range)."""
    return roi.shape[0] > 2.2 * line_px


def tesseract_read(clean: np.ndarray, cls: str, multiline: bool, cfg: PreCfg = TUNED):
    """Return (text, mean word confidence 0-100). Lines are joined with ' | '."""
    psm = cfg.psm_multi if multiline else cfg.psm_single
    config = f"--oem 3 --psm {psm}"
    if cfg.value_whitelist and cls == "value":
        config += " -c tessedit_char_whitelist=0123456789.<>"
    d = pytesseract.image_to_data(clean, config=config, output_type=pytesseract.Output.DICT)
    lines, confs = {}, []
    for i, word in enumerate(d["text"]):
        word = word.strip()
        if not word:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        lines.setdefault(key, []).append(word)
        c = float(d["conf"][i])
        if c >= 0:
            confs.append(c)
    text = clean_text(" | ".join(" ".join(ws) for _, ws in sorted(lines.items())), cls)
    if cfg.unit_lexicon and cls == "unit":
        text = snap_unit(text)
    return text, (float(np.mean(confs)) if confs else 0.0)


def clean_text(s: str, cls: str) -> str:
    """Light, generic clean-up: strip stray edge punctuation that box borders produce."""
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" |_'\"`‘’,;")
    if cls == "value":
        s = s.replace(" ", "").rstrip(".")          # a number never ends with a full stop
    return s


# --------------------------------------------------------------------------- optional: unit lexicon
# Units on lab reports come from a small, closed vocabulary. This list is GENERIC clinical
# units (not copied from this dataset) so snapping to it does not leak the answers.
UNIT_LEXICON = [
    "mg/dl", "g/dl", "gm/dl", "ng/dl", "µg/dl", "ng/ml", "pg/ml", "µg/ml", "mg/l", "g/l",
    "µg/l", "ng/l", "mmol/l", "µmol/l", "nmol/l", "pmol/l", "mEq/l", "IU/l", "U/l", "U/ml",
    "IU/ml", "mIU/ml", "µIU/ml", "mU/l", "%", "fl", "pg", "cells/µl", "10^3/µl", "10^6/µl",
    "mm/hr", "sec", "Ratio", "copies/ml", "mg/24hr", "mOsm/kg", "µg/g", "mg/g", "U/g",
]
# Characters Tesseract commonly confuses with each other on small print (generic OCR
# confusions, not dataset-specific). Swapping within a group costs 0.3 instead of 1.
_CONFUSABLE = ["li1|!\\tI", "gqa9", "µupyvw", "o0", "s5$", "b8", "m"]
_GROUP = {c: k for k, grp in enumerate(_CONFUSABLE) for c in grp}


def _wdist(a: str, b: str) -> float:
    """Edit distance where visually-confusable substitutions are cheap."""
    prev = [float(j) for j in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [float(i)]
        for j, cb in enumerate(b, 1):
            if ca == cb:
                sub = 0.0
            elif ca in _GROUP and _GROUP.get(ca) == _GROUP.get(cb):
                sub = 0.3
            else:
                sub = 1.0
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub))
        prev = cur
    return prev[-1]


def snap_unit(raw: str, max_rel: float = 0.34) -> str:
    """Return the closest lexicon unit if it is clearly the nearest; otherwise the raw text."""
    s = raw.strip().replace(" ", "")
    if not s:
        return raw
    low = s.lower()
    scored = sorted((_wdist(low, u.lower()), u) for u in UNIT_LEXICON)
    (d1, u1), (d2, _) = scored[0], scored[1]
    if d1 == 0:
        return u1
    if d1 <= max_rel * len(u1) and d2 - d1 >= 0.3:      # close AND unambiguous
        return u1
    return raw


# --------------------------------------------------------------------------- stage 5: assemble
def estimate_skew(img: np.ndarray, max_deg: float = 4.0, step: float = 0.1) -> float:
    """Angle (radians, y pointing down) of the text lines, by projection-profile search:
    rotate the binarised page and keep the angle whose row profile is sharpest.
    On this dataset: median error 0.1 deg, worst 0.65 deg, ~25 ms per photo."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    s = 600 / max(g.shape)
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    h, w = b.shape
    best, best_a = -1.0, 0.0
    for a in np.arange(-max_deg, max_deg + 1e-9, step):
        rot = cv2.warpAffine(b, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0), (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.sum(np.diff(rot.sum(1).astype(np.float64)) ** 2))
        if score > best:
            best, best_a = score, a
    return float(np.deg2rad(best_a))


def assemble_rows(fields: list[dict], angle: float = 0.0) -> list[dict]:
    """Attach value/unit/range to the right test name.

    Photos are tilted, so a unit at the far right of a row can sit higher on the page than its
    own test name. We first "un-tilt" every box (y' = y - x*tan(angle)), then:
    * if a class has exactly one box per test name -> match by order (robust to any tilt);
    * otherwise value/unit go to the test name on the same un-tilted line, and a
      reference_range goes to the nearest test name on or above it (the vitamin page prints
      the range BELOW the name).
    """
    t = np.tan(angle)
    yk = lambda f: _cy(f) - _cx(f) * t                       # tilt-corrected vertical position
    tests = sorted([f for f in fields if f["cls"] == "test_name"], key=yk)
    rows = [{"test": f, "value": None, "unit": None, "reference_range": None} for f in tests]
    if not rows:
        return []
    for c in ("value", "unit", "reference_range"):
        boxes = [f for f in fields if f["cls"] == c]
        if len(boxes) == len(rows):                          # one per row: order is enough
            for r, f in zip(rows, sorted(boxes, key=yk)):
                r[c] = f
            continue
        for f in sorted(boxes, key=lambda f: -f["conf"]):     # most confident box wins a slot
            h = f["box"][3] - f["box"][1]
            if c in ("value", "unit"):
                i = int(np.argmin([abs(yk(r["test"]) - yk(f)) for r in rows]))
            else:
                above = [i for i, r in enumerate(rows) if yk(r["test"]) - h / 2 <= yk(f) + 0.25 * h]
                i = max(above, key=lambda i: yk(rows[i]["test"])) if above else 0
            if rows[i][c] is None:
                rows[i][c] = f
    return rows


def _cy(f):
    return (f["box"][1] + f["box"][3]) / 2


def _cx(f):
    return (f["box"][0] + f["box"][2]) / 2


# --------------------------------------------------------------------------- full pipeline
NUMERIC = re.compile(r"^[<>]?\d+(\.\d+)?$")


def run_image(model, path: Path, cfg: PreCfg = TUNED, conf: float = 0.25, debug_dir: Path | None = None):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    dets = detect(model, img, conf=conf)
    return read_fields(img, dets, cfg, path.name, debug_dir)


def run_document(model, data: bytes, name: str = "upload", cfg: PreCfg = TUNED, conf: float = 0.25,
                 debug_dir: Path | None = None) -> list[dict]:
    """A photo or a PDF (any number of pages) held in memory -> one result per page:
    {'page', 'name', 'kind', 'rows' (CSV-ready, with a 'page' column), 'fields', 'annotated'}."""
    out = []
    for pg in load_pages(data, name):
        records, fields = read_fields(pg.img, detect(model, pg.img, conf=conf), cfg, pg.name, debug_dir, pg.words)
        rows = [{"image": name, "page": pg.page,
                 **{k: v for k, v in r.items() if not k.startswith("_") and k != "image"}} for r in records]
        out.append({"page": pg.page, "name": pg.name, "kind": pg.kind, "rows": rows, "fields": fields,
                    "annotated": draw(pg.img, fields)})
    return out


def read_fields(img, dets, cfg: PreCfg, name: str = "", debug_dir: Path | None = None, words: list | None = None):
    """Stages 2-5 for a list of boxes (from YOLO, or ground-truth boxes for oracle tests).
    `words` (a digital PDF's text layer) is read directly where it covers a box; OCR otherwise."""
    crops = [crop(img, d["box"], cfg.pad) for d in dets]                 # stage 2: list of crops
    line_px = np.median([c.shape[0] for c, d in zip(crops, dets) if d["cls"] == "value"] or [12])
    fields = []
    for k, (d, roi) in enumerate(zip(dets, crops)):
        if roi.size == 0:
            continue
        multi = is_multiline(roi, line_px)
        exact = clean_text(text_layer_read(words, d["box"]), d["cls"]) if words else ""
        if exact:                                                        # digital PDF: exact text
            fields.append({**d, "text": exact, "ocr_conf": 100.0, "multiline": multi, "source": "pdf_text"})
            continue
        steps = {} if debug_dir is not None and k == 0 else None
        clean = preprocess(roi, cfg, steps)                              # stage 3
        text, tconf = tesseract_read(clean, d["cls"], multi, cfg)         # stage 4
        fields.append({**d, "text": text, "ocr_conf": tconf, "multiline": multi, "source": "ocr"})
        if steps:
            save_steps(steps, debug_dir, Path(name).stem)
    rows = assemble_rows(fields, estimate_skew(img))                      # stage 5
    records = []
    for i, r in enumerate(rows):
        rec = {"image": name, "row": i + 1}
        confs_det, confs_ocr = [], []
        for c in CLASSES:
            f = r["test"] if c == "test_name" else r[c]
            rec[c] = f["text"] if f else ""
            if f:
                confs_det.append(f["conf"])
                confs_ocr.append(f["ocr_conf"])
        rec["det_conf_min"] = round(min(confs_det), 3)
        rec["ocr_conf_min"] = round(min(confs_ocr), 1)
        missing = [c for c in CLASSES if not rec[c]]
        reasons = []
        if missing:
            reasons.append("missing:" + "+".join(missing))
        if rec["value"] and not NUMERIC.match(rec["value"]):
            reasons.append("value_not_numeric")
        if rec["det_conf_min"] < 0.5:
            reasons.append("low_detection_conf")
        if rec["ocr_conf_min"] < 60:
            reasons.append("low_ocr_conf")
        rec["needs_review"] = bool(reasons)
        rec["review_reason"] = ";".join(reasons)
        srcs = {f["source"] for f in (r["test"], r["value"], r["unit"], r["reference_range"]) if f}
        rec["source"] = srcs.pop() if len(srcs) == 1 else "mixed"     # ocr | pdf_text | mixed
        rec["_test_box"] = r["test"]["box"]          # kept for scoring, not written to the CSV
        records.append(rec)
    return records, fields


# --------------------------------------------------------------------------- outputs
def draw(img, fields):
    vis = img.copy()
    for f in fields:
        x0, y0, x1, y1 = map(int, f["box"])
        cv2.rectangle(vis, (x0, y0), (x1, y1), COLORS[f["cls"]], 2)
        cv2.putText(vis, f"{f['cls']} {f['conf']:.2f}", (x0, max(y0 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS[f["cls"]], 1, cv2.LINE_AA)
    return vis


def save_steps(steps: dict, out: Path, stem: str):
    out.mkdir(parents=True, exist_ok=True)
    for k, im in steps.items():
        cv2.imwrite(str(out / f"{stem}_{k}.png"), im)


def main(argv=None):
    ap = argparse.ArgumentParser(description="YOLO + Tesseract custom OCR for lab reports")
    ap.add_argument("--image", required=True, help="an image file or a folder of images")
    ap.add_argument("--weights", default=str(ROOT / "model" / "best.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "04_inference"))
    ap.add_argument("--mode", choices=list(MODES), default="tuned")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--debug", action="store_true", help="save every preprocessing step")
    ap.add_argument("--tesseract", default=None, help="path to tesseract.exe if not on PATH")
    ap.add_argument("--excel", action="store_true", help="also write an .xlsx next to every CSV")
    a = ap.parse_args(argv)
    if a.tesseract:
        pytesseract.pytesseract.tesseract_cmd = a.tesseract

    src = Path(a.image)
    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in DOC_EXT) if src.is_dir() else [src]
    if not paths:
        sys.exit(f"No images or PDFs found at {src}")
    out = Path(a.out)
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    model = load_model(a.weights)
    all_rows = []
    for p in paths:
        try:
            pages = run_document(model, p.read_bytes(), p.name, MODES[a.mode], a.conf,
                                 out / "steps" if a.debug else None)
        except ValueError as e:
            print(f"{p.name}: skipped ({e})")
            continue
        rows = [r for pg in pages for r in pg["rows"]]
        for pg in pages:                     # one annotated image per photo / per PDF page
            stem = p.stem if len(pages) == 1 and pg["kind"] == "image" else pg["name"]
            cv2.imwrite(str(out / "annotated" / f"{stem}_boxes.jpg"), pg["annotated"])
        cols = ["image", "page", "row"] + CLASSES + ["det_conf_min", "ocr_conf_min", "needs_review",
                                                     "review_reason", "source"]
        df_p = pd.DataFrame(rows, columns=cols)
        df_p.to_csv(out / f"{p.stem}.csv", index=False, encoding="utf-8-sig")
        if a.excel:
            df_p.to_excel(out / f"{p.stem}.xlsx", index=False)
        all_rows += rows
        what = f"{len(pages)} page{'s' if len(pages) > 1 else ''}, " if p.suffix.lower() == ".pdf" else ""
        print(f"{p.name}: {what}{len(rows)} tests extracted, {sum(r['needs_review'] for r in rows)} flagged for review")
    df = pd.DataFrame(all_rows)
    if len(paths) > 1:
        df.to_csv(out / "all_reports.csv", index=False, encoding="utf-8-sig")
        if a.excel:
            df.to_excel(out / "all_reports.xlsx", index=False)
    print(f"\nCSV saved to {out.resolve()}")
    if len(paths) == 1 and len(df):
        with pd.option_context("display.max_colwidth", 40, "display.width", 200):
            print(df[["page", "row"] + CLASSES + ["needs_review"]].to_string(index=False))
    return df


if __name__ == "__main__":
    main()
