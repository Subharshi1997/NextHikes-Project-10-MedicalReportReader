"""
Scoring helpers for the Custom OCR pipeline.

Metrics (plain English)
-----------------------
* Exact match  - after light normalisation (upper-case, no spaces, µ->U), is the extracted
                 text identical to the truth? This is the number a lab cares about: a value
                 that is "almost right" (2.14 vs 2.4) is still wrong.
* CER          - character error rate = edits needed to turn the prediction into the truth,
                 divided by the truth length. 0 = perfect, 1 = everything wrong/missing.
* Oracle boxes - feed the hand-checked ground-truth boxes instead of YOLO's boxes. The gap
                 between oracle and YOLO scores is the error caused by detection; whatever
                 remains with oracle boxes is caused by preprocessing + Tesseract.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Custom_OCR as co  # noqa: E402

CLASSES = co.CLASSES


def normalize(s: str) -> str:
    s = (s or "").upper().replace("µ", "U").replace("μ", "U")
    for ch in " |\t\n":
        s = s.replace(ch, "")
    return s.replace("—", "-").replace("–", "-")


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, truth: str) -> float:
    p, t = normalize(pred), normalize(truth)
    return min(1.0, levenshtein(p, t) / max(len(t), 1))


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def load_annotations(path=ROOT / "datasets" / "annotations.json") -> dict:
    return json.load(open(path, encoding="utf-8"))


def oracle_dets(ann_img: dict) -> list[dict]:
    """Ground-truth boxes in the same format YOLO returns (conf = 1)."""
    return [{"cls": b["cls"], "conf": 1.0, "box": b["box"], "gt_text": b["text"],
             "row_gt": b["row"], "full": b["full"]} for b in ann_img["boxes"]]


def _oracle_one(images_dir, ann_img: dict, n: str, cfg: co.PreCfg) -> list[dict]:
    img = cv2.imread(str(Path(images_dir) / n))
    dets = [d for d in oracle_dets(ann_img) if d["full"]]
    _, fields = co.read_fields(img, dets, cfg, n)
    return [{"image": n, "layout": ann_img["layout"], "field": f["cls"], "row": f["row_gt"],
             "truth": f["gt_text"], "pred": f["text"],
             "exact": normalize(f["text"]) == normalize(f["gt_text"]),
             "cer": cer(f["text"], f["gt_text"])} for f in fields]


def score_oracle(images_dir: Path, ann: dict, names: list[str], cfg: co.PreCfg, workers: int = 1) -> pd.DataFrame:
    """OCR every fully-visible ground-truth box with config `cfg`; one row per field.
    workers > 1 reads the photos in parallel processes (Tesseract is single-image, CPU-bound)."""
    if workers <= 1:
        rows = [r for n in names for r in _oracle_one(images_dir, ann[n], n, cfg)]
    else:
        import os
        from concurrent.futures import ProcessPoolExecutor
        os.environ["OMP_THREAD_LIMIT"] = "1"           # one Tesseract thread per process
        with ProcessPoolExecutor(workers) as ex:
            parts = ex.map(_oracle_one, [images_dir] * len(names), [ann[n] for n in names], names,
                           [cfg] * len(names))
            rows = [r for part in parts for r in part]
    return pd.DataFrame(rows)


def score_end_to_end(images_dir: Path, ann: dict, names: list[str], cfg: co.PreCfg, model,
                     conf: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline (YOLO -> OCR -> row assembly -> CSV) scored against the truth.

    Predicted CSV rows are matched to true rows through the test_name box (IoU >= 0.3),
    so a value attached to the wrong test counts as wrong.
    Returns (field-level scores, the raw predicted CSV rows).
    """
    scores, preds = [], []
    for n in names:
        img = cv2.imread(str(Path(images_dir) / n))
        dets = co.detect(model, img, conf=conf)
        records, _ = co.read_fields(img, dets, cfg, n)
        preds += records
        gt_rows = {}
        for b in ann[n]["boxes"]:
            gt_rows.setdefault(b["row"], {})[b["cls"]] = b
        used = set()
        for r, g in sorted(gt_rows.items()):
            # Match the predicted row through the test-name box (even if the photo cuts it off,
            # the visible part still anchors the row). Score exactly the fully-visible fields,
            # i.e. the same field set as score_oracle(), so A-E are directly comparable.
            best, best_iou = None, 0.3
            if "test_name" in g:
                for k, rec in enumerate(records):
                    if k in used:
                        continue
                    v = iou(rec["_test_box"], g["test_name"]["box"])
                    if v >= best_iou:
                        best, best_iou = k, v
            if best is not None:
                used.add(best)
            for c in CLASSES:
                if c not in g or not g[c]["full"]:
                    continue        # a field cut off by the photo edge cannot be read by anyone
                pred = records[best][c] if best is not None else ""
                scores.append({"image": n, "layout": ann[n]["layout"], "row": r, "field": c,
                               "truth": g[c]["text"], "pred": pred, "row_found": best is not None,
                               "needs_review": records[best]["needs_review"] if best is not None else True,
                               "exact": normalize(pred) == normalize(g[c]["text"]),
                               "cer": cer(pred, g[c]["text"])})
    return pd.DataFrame(scores), pd.DataFrame(preds)


def spurious_rows(preds: pd.DataFrame, ann: dict) -> pd.DataFrame:
    """Predicted CSV rows whose test-name box matches no real test (false-positive detections)."""
    out = []
    for _, p in preds.iterrows():
        truth = [b["box"] for b in ann[p["image"]]["boxes"] if b["cls"] == "test_name"]
        best = max([iou(p["_test_box"], t) for t in truth] or [0])
        if best < 0.3:
            out.append({"image": p["image"], "test_name": p["test_name"], "value": p["value"],
                        "needs_review": p["needs_review"], "review_reason": p["review_reason"]})
    return pd.DataFrame(out, columns=["image", "test_name", "value", "needs_review", "review_reason"])


def summary(df: pd.DataFrame, by="field") -> pd.DataFrame:
    s = df.groupby(by).agg(fields=("exact", "size"), exact_match=("exact", "mean"), mean_cer=("cer", "mean"))
    s.loc["ALL"] = [len(df), df["exact"].mean(), df["cer"].mean()]
    s["exact_match"] = (s["exact_match"] * 100).round(1)
    s["mean_cer"] = s["mean_cer"].round(3)
    return s


def detection_recall(model, images_dir: Path, ann: dict, names: list[str], conf=0.25, thr=0.5) -> pd.DataFrame:
    """For every fully-visible true box: did YOLO find a same-class box with IoU >= thr?"""
    out = []
    for n in names:
        img = cv2.imread(str(Path(images_dir) / n))
        dets = co.detect(model, img, conf=conf)
        for b in ann[n]["boxes"]:
            if not b["full"]:
                continue
            best = max([iou(d["box"], b["box"]) for d in dets if d["cls"] == b["cls"]] or [0])
            out.append({"image": n, "layout": ann[n]["layout"], "field": b["cls"], "iou": best, "found": best >= thr})
    return pd.DataFrame(out)


def full_page_baseline(images_dir: Path, ann: dict, names: list[str]) -> pd.DataFrame:
    """Tesseract on the whole page, no YOLO: is each true value even present in the text?"""
    import pytesseract
    out = []
    for n in names:
        img = cv2.imread(str(Path(images_dir) / n))
        text = pytesseract.image_to_string(img, config="--oem 3 --psm 3")
        tokens = {normalize(t) for t in text.split()}
        for b in ann[n]["boxes"]:
            if b["cls"] == "value" and b["full"]:
                out.append({"image": n, "truth": b["text"], "found_somewhere": normalize(b["text"]) in tokens})
    return pd.DataFrame(out)


def timed(fn, *a, **k):
    t = time.time()
    r = fn(*a, **k)
    return r, time.time() - t
