"""
Build the sample PDFs in samples/ (all from redacted images, so they are safe to publish).

* lab_reports_scanned.pdf - the three sample photos as a 3-page "scanned" PDF (an image per page,
                            no text layer), like a phone scanner app produces.
* liver_report_digital.pdf - a TEST-split liver photo (never trained on), redacted, with the true text
                             of every field placed under the page image as a real text layer, like a
                             lab-system PDF. The page renders exactly like the report, and the pipeline
                             reads each detected field from the text layer instead of running OCR.

    python tools/make_sample_pdfs.py            (needs datasets/raw for the digital sample)
"""
import json
import sys
from pathlib import Path

import cv2
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from privacy import redact  # noqa: E402

S = ROOT / "samples"


def scanned(photos=("thyroid_report.jpg", "liver_report.jpg", "vitamin_report.jpg"), margin=24):
    doc = pymupdf.open()
    for p in photos:
        pix = pymupdf.Pixmap(str(S / p))
        landscape = pix.width > pix.height
        page = doc.new_page(width=842 if landscape else 595, height=595 if landscape else 842)   # A4
        r = page.rect + (margin, margin, -margin, -margin)
        k = min(r.width / pix.width, r.height / pix.height)
        w, h = pix.width * k, pix.height * k
        x0, y0 = r.x0 + (r.width - w) / 2, r.y0 + (r.height - h) / 2
        page.insert_image(pymupdf.Rect(x0, y0, x0 + w, y0 + h), filename=str(S / p))
    doc.save(S / "lab_reports_scanned.pdf", garbage=4, deflate=True)
    return S / "lab_reports_scanned.pdf"


def digital(name="thyrocare_0_3558.jpg"):
    ann = json.load(open(ROOT / "datasets" / "annotations.json", encoding="utf-8"))[name]
    assert ann["split"] == "test", "use a photo the detector never trained on"
    ok, jpg = cv2.imencode(".jpg", redact(cv2.imread(str(ROOT / "datasets" / "raw" / name)), ann["redact"]),
                           [cv2.IMWRITE_JPEG_QUALITY, 90])
    k = 0.75                                                  # 96-dpi pixels -> PDF points
    doc = pymupdf.open()
    page = doc.new_page(width=ann["w"] * k, height=ann["h"] * k)
    for b in ann["boxes"]:                                    # text first ...
        if not b["full"]:
            continue
        rect = pymupdf.Rect(*[v * k for v in b["box"]])
        lines = b["text"].split(" | ")
        fs = max(4.0, rect.height / len(lines) * 0.62)
        for i, t in enumerate(lines):
            page.insert_text((rect.x0 + 1, rect.y0 + (i + 0.82) * rect.height / len(lines)), t,
                             fontsize=fs, fontname="helv", color=(0, 0, 0))
    page.insert_image(page.rect, stream=jpg.tobytes())        # ... then the page image on top
    doc.save(S / "liver_report_digital.pdf", garbage=4, deflate=True)
    return S / "liver_report_digital.pdf"


if __name__ == "__main__":
    print("written", scanned())
    if (ROOT / "datasets" / "raw").exists():
        print("written", digital())
    else:
        print("datasets/raw not found: digital sample skipped")
