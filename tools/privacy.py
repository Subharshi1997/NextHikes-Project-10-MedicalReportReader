"""Blur patient-identifying regions before any image leaves the project.

The raw photos show a patient's name/age, a phone number, a collection address,
lab/barcode IDs and the signing doctors' names and signatures. None of these are needed for the OCR task, so every figure, sample
image and result picture published with this repo goes through `redact()` first.
"""
import cv2
import numpy as np


def redact(img: np.ndarray, boxes, pad: int = 4) -> np.ndarray:
    """Return a copy with each [x0,y0,x1,y1] box replaced by a flat grey block."""
    out = img.copy()
    h, w = out.shape[:2]
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(int(x0) - pad, 0), max(int(y0) - pad, 0)
        x1, y1 = min(int(x1) + pad, w), min(int(y1) + pad, h)
        out[y0:y1, x0:x1] = (200, 200, 200)
        if x1 - x0 > 60 and y1 - y0 > 12:
            cv2.putText(out, "REDACTED", (x0 + 4, min(y0 + 14, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (90, 90, 90), 1, cv2.LINE_AA)
    return out
