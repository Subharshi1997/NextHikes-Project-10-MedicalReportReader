"""
Web app + API for the Custom OCR pipeline (brief section 6: automation and deployment).

One server, two audiences:
* people   - GET /            upload a lab-report photo or PDF, check the extracted table, download CSV / Excel
* machines - POST /api/ocr     multipart upload (image or PDF) -> JSON rows + one annotated image per page
             GET  /ping        \  the Amazon SageMaker hosting contract, so the SAME Docker image
             POST /invocations /  runs as a SageMaker real-time endpoint (see aws/)

    uvicorn app.server:app --port 8000                  # local
    docker run -p 8080:8080 custom-ocr serve             # container (SageMaker listens on 8080)

Environment variables
---------------------
OCR_WEIGHTS  path to the YOLO weights (default model/best.pt; SageMaker: /opt/ml/model/best.pt if present)
OCR_MODE     tuned (default) or brief
OCR_CONF     detection confidence threshold (default 0.25)
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Custom_OCR as co  # noqa: E402

MAX_BYTES = 25 * 1024 * 1024
COLUMNS = ["page", "row"] + co.CLASSES + ["det_conf_min", "ocr_conf_min", "needs_review", "review_reason", "source"]


def _weights() -> Path:
    for cand in (os.environ.get("OCR_WEIGHTS"), "/opt/ml/model/best.pt", ROOT / "model" / "best.pt"):
        if cand and Path(cand).is_file():
            return Path(cand)
    raise FileNotFoundError("no YOLO weights found: set OCR_WEIGHTS or put best.pt in model/")


app = FastAPI(title="Custom OCR for lab reports", version="1.0")
STATE: dict = {}


@app.on_event("startup")
def _load():
    STATE["weights"] = _weights()
    STATE["model"] = co.load_model(STATE["weights"])
    STATE["cfg"] = co.MODES[os.environ.get("OCR_MODE", "tuned")]
    STATE["conf"] = float(os.environ.get("OCR_CONF", "0.25"))


def _ocr(data: bytes, name: str) -> dict:
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "file larger than 25 MB")
    t = time.time()
    try:
        pages = co.run_document(STATE["model"], data, name, STATE["cfg"], STATE["conf"])
    except ValueError as e:
        raise HTTPException(415, str(e))

    def b64(img):
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(jpg.tobytes()).decode() if ok else None

    page_out = [{"page": p["page"], "kind": p["kind"], "n_fields": len(p["fields"]), "n_rows": len(p["rows"]),
                 "annotated_jpeg_base64": b64(p["annotated"])} for p in pages]
    rows = [{k: r.get(k) for k in COLUMNS} for p in pages for r in p["rows"]]
    return {"image": name, "kind": pages[0]["kind"], "pages": page_out, "rows": rows,
            "n_fields": sum(p["n_fields"] for p in page_out), "seconds": round(time.time() - t, 2),
            "annotated_jpeg_base64": page_out[0]["annotated_jpeg_base64"]}    # first page (backward compatible)


# --------------------------------------------------------------------------- people
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/health")
def health():
    ckpt = getattr(STATE["model"], "ckpt", None) or {}
    arch = Path(str((ckpt.get("train_args") or {}).get("model", STATE["weights"].name))).stem
    return {"status": "ok", "weights": STATE["weights"].name, "detector": arch, "mode": STATE["cfg"].name,
            "tesseract": str(co.pytesseract.pytesseract.tesseract_cmd)}


SAMPLES = ROOT / "samples"           # redacted demo photos shipped with the repo


@app.get("/api/samples")
def samples():
    return sorted(p.name for p in SAMPLES.glob("*") if p.suffix.lower() in co.DOC_EXT)


def _sample(name: str) -> Path:
    path = SAMPLES / Path(name).name                       # no path traversal
    if not path.is_file() or path.suffix.lower() not in co.DOC_EXT:
        raise HTTPException(404, "unknown sample")
    return path


@app.get("/api/samples/{name}/image", include_in_schema=False)
def sample_image(name: str):
    path = _sample(name)
    if path.suffix.lower() != ".pdf":
        return FileResponse(path)
    import pymupdf                                      # thumbnail of a PDF's first page
    with pymupdf.open(path) as doc:
        png = doc[0].get_pixmap(matrix=pymupdf.Matrix(0.6, 0.6), alpha=False).tobytes("png")
    return Response(png, media_type="image/png")


@app.post("/api/samples/{name}")
def run_sample(name: str):
    path = _sample(name)
    return _ocr(path.read_bytes(), path.name)


@app.post("/api/ocr")
async def api_ocr(file: UploadFile = File(...)):
    return _ocr(await file.read(), file.filename or "upload")


@app.post("/api/export/{fmt}")
async def export(fmt: str, request: Request):
    """Turn the (possibly hand-corrected) table sent by the page into a CSV or Excel file."""
    body = await request.json()
    df = pd.DataFrame(body.get("rows", []))
    stem = Path(body.get("image") or "report").stem
    if fmt == "csv":
        data = df.to_csv(index=False).encode("utf-8-sig")
        media = "text/csv"
    elif fmt == "xlsx":
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        data, media = buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        raise HTTPException(404, "format must be csv or xlsx")
    return StreamingResponse(io.BytesIO(data), media_type=media,
                             headers={"Content-Disposition": f'attachment; filename="{stem}.{fmt}"'})


# --------------------------------------------------------------------------- SageMaker contract
@app.get("/ping")
def ping():
    return Response(status_code=200 if "model" in STATE else 503)


@app.post("/invocations")
async def invocations(request: Request):
    """Body = raw image or PDF bytes (image/jpeg, image/png, application/pdf). Accept: application/json
    (default, rows + annotated image) or text/csv (just the table)."""
    out = _ocr(await request.body(), "invocation")
    if "text/csv" in request.headers.get("accept", ""):
        return Response(pd.DataFrame(out["rows"]).to_csv(index=False), media_type="text/csv")
    return JSONResponse(out)
