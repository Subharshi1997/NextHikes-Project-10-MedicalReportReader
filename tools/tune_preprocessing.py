"""
Sequential (coordinate-descent) ablation of the preprocessing + Tesseract settings.

Starts from the brief's exact recipe and changes ONE knob at a time, keeping a change
only if it raises field exact-match on the TUNING images (train + val). The test
images are never used here, so the final test score is an honest, unseen measurement.

Boxes are the ground-truth ("oracle") boxes, so this isolates the OCR stage from YOLO.

    python tools/tune_preprocessing.py --images path/to/raw_jpgs
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import Custom_OCR as co  # noqa: E402
import evaluate as ev    # noqa: E402

SEARCH = [                                   # knob, candidate values (brief value first)
    ("psm_single", [6, 7, 13]),
    ("border", [0, 10]),
    ("interp", [cv2.INTER_LINEAR, cv2.INTER_CUBIC]),
    ("blur", [5, 3, 0]),
    ("binarize", [True, False]),
    ("scale", [3.0, 2.0, 4.0]),
    ("value_whitelist", [False, True]),
    ("pad", [3, 1, 6]),
]
INTERP_NAME = {cv2.INTER_LINEAR: "linear", cv2.INTER_CUBIC: "cubic"}


def describe(cfg):
    d = dataclasses.asdict(cfg)
    d["interp"] = INTERP_NAME.get(d["interp"], d["interp"])
    return {k: v for k, v in d.items() if k != "name"}


def main(images, out_dir, workers=1):
    ann = ev.load_annotations()
    tune = [n for n, a in ann.items() if a["split"] != "test"]
    cache, log = {}, []

    def score(cfg):
        key = json.dumps(describe(cfg), sort_keys=True)
        if key not in cache:
            df = ev.score_oracle(images, ann, tune, cfg, workers)
            cache[key] = (df["exact"].mean(), df["cer"].mean(), df)
        return cache[key]

    best = dataclasses.replace(co.BRIEF, name="step0_brief")
    b_exact, b_cer, _ = score(best)
    log.append({"step": 0, "knob": "(brief baseline)", "value": "-", "exact_match": b_exact,
                "mean_cer": b_cer, "kept": True, **describe(best)})
    print(f"brief baseline: exact={b_exact:.3f} cer={b_cer:.3f}", flush=True)
    step = 0
    for knob, values in SEARCH:
        for v in values:
            if getattr(best, knob) == v:
                continue
            step += 1
            cand = dataclasses.replace(best, **{knob: v})
            e, c, _ = score(cand)
            kept = (e > b_exact + 1e-9) or (abs(e - b_exact) < 1e-9 and c < b_cer)
            shown = INTERP_NAME.get(v, v) if knob == "interp" else v
            log.append({"step": step, "knob": knob, "value": shown, "exact_match": e,
                        "mean_cer": c, "kept": kept, **describe(cand)})
            print(f"{knob}={shown}: exact={e:.3f} cer={c:.3f} {'KEEP' if kept else ''}", flush=True)
            if kept:
                best, b_exact, b_cer = cand, e, c
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(out_dir / "ablation_log.csv", index=False)
    json.dump(describe(best), open(out_dir / "best_preprocessing.json", "w"), indent=1)
    cache[json.dumps(describe(best), sort_keys=True)][2].to_csv(out_dir / "oracle_tuning_best_fields.csv", index=False)
    print("BEST", describe(best), f"exact={b_exact:.3f} cer={b_cer:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "05_evaluation"))
    ap.add_argument("--workers", type=int, default=1, help="parallel processes for Tesseract")
    a = ap.parse_args()
    main(a.images, a.out, a.workers)
