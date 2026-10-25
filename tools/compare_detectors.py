"""
Head-to-head: the brief's YOLOv3 against YOLO26n, on the same split and the same OCR stage.

For each weights file, on the unseen TEST photos only:
* detection  - mAP@0.5, mAP@0.5:0.95 (Ultralytics val) and field recall at IoU >= 0.5
* end to end - exact match and CER of the final CSV, with the brief's preprocessing (C)
               and with the shipped tuned preprocessing + unit lexicon (E)
* cost       - parameters, weights size, mean seconds per photo (detect + OCR)

    python tools/compare_detectors.py --weights model/yolov3u_best.pt model/yolo26n_best.pt
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import Custom_OCR as co  # noqa: E402
import evaluate as ev    # noqa: E402


def compare(weights, images, out, device=None):
    ann = ev.load_annotations()
    test = sorted(n for n, a in ann.items() if a["split"] == "test")
    rows, per_field = [], []
    for w in weights:
        w = Path(w)
        model = co.load_model(w)
        m = model.val(data=str(ROOT / "datasets" / "yolo" / "data.yaml"), split="test", imgsz=1024,
                      batch=4, device=device, plots=False, verbose=False)
        rec = ev.detection_recall(model, images, ann, test)
        co.detect(model, co.cv2.imread(str(Path(images) / test[0])))          # warm-up before timing
        t = time.time()
        e_scores, _ = ev.score_end_to_end(images, ann, test, co.TUNED, model)
        sec = (time.time() - t) / len(test)
        c_scores, _ = ev.score_end_to_end(images, ann, test, co.BRIEF, model)
        name = w.stem.replace("_best", "")
        rows.append({
            "detector": name, "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 1),
            "weights_MB": round(w.stat().st_size / 1e6, 1),
            "test_mAP50": round(float(m.box.map50), 3), "test_mAP50_95": round(float(m.box.map), 3),
            "field_recall_iou50": round(rec["found"].mean(), 3),
            "exact_brief_C": round(c_scores["exact"].mean() * 100, 1),
            "exact_shipped_E": round(e_scores["exact"].mean() * 100, 1),
            "cer_shipped_E": round(e_scores["cer"].mean(), 3),
            "value_exact_E": round(e_scores.loc[e_scores.field == "value", "exact"].mean() * 100, 1),
            "sec_per_photo": round(sec, 2), "test_photos": len(test), "fields": len(e_scores)})
        s = ev.summary(e_scores)["exact_match"].rename(name)
        per_field.append(s)
        print(rows[-1], flush=True)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "detector_comparison.csv", index=False)
    pd.concat(per_field, axis=1).to_csv(out / "detector_comparison_by_field.csv")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--images", default=str(ROOT / "datasets" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "05_evaluation"))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    print(compare(a.weights, a.images, Path(a.out), a.device).to_string(index=False))
