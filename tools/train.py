"""
Train the field detector (YOLO26n, fine-tuned from COCO-pretrained weights).

Why these settings (plain English)
----------------------------------
* imgsz=1024  - lab-report text is tiny (6-10 px tall). Training near the photos'
                native resolution (max side 1068 px) keeps those characters visible.
* fliplr=0    - a mirrored report is not a real report: the test name is always on
                the left and the range on the right, so horizontal flips would
                teach the model wrong geometry.
* degrees/perspective small - phone photos are slightly tilted, never upside down.
* batch=4     - fits in 8 GB RAM on CPU; on a Colab T4 GPU use batch=16.

Usage
-----
    python tools/train.py --data datasets/yolo/data.yaml --epochs 100 --device cpu
    python tools/train.py --data datasets/yolo/data.yaml --epochs 100 --device 0 --batch 16   # Colab GPU
    python tools/train.py --weights yolov3u.pt --device 0 --batch 4    # the brief's YOLOv3 (Darknet-53)
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

HYP = dict(
    imgsz=1024, fliplr=0.0, flipud=0.0, degrees=3.0, translate=0.10, scale=0.30,
    perspective=0.0003, shear=1.0, mosaic=1.0, close_mosaic=15, mixup=0.0,
    hsv_h=0.02, hsv_s=0.5, hsv_v=0.4, patience=40, seed=42, deterministic=True,
    plots=True, workers=0,
)


def train(data, epochs=100, batch=4, device="cpu", weights="yolo26n.pt",
          project=ROOT / "results" / "03_training", name=None, save_as=None):
    name = name or f"{Path(weights).stem}_1024"
    model = YOLO(weights)                      # COCO-pretrained starting point
    model.train(data=str(data), epochs=epochs, batch=batch, device=device,
                project=str(project), name=name, exist_ok=True, **HYP)
    run = Path(project) / name
    (ROOT / "model").mkdir(exist_ok=True)
    # brief: weights in model/  (yolov3u.pt -> model/yolov3u_best.pt, unless --save-as says otherwise)
    shutil.copy(run / "weights" / "best.pt", ROOT / "model" / (save_as or f"{Path(weights).stem}_best.pt"))
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets" / "yolo" / "data.yaml"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights", default="yolo26n.pt", help="yolo26n.pt, or yolov3u.pt for the brief's YOLOv3")
    ap.add_argument("--name", default=None, help="run folder name (default: <weights>_1024)")
    ap.add_argument("--save-as", default=None, help="file name inside model/ (default: <weights>_best.pt)")
    a = ap.parse_args()
    print("run folder:", train(a.data, a.epochs, a.batch, a.device, a.weights, name=a.name, save_as=a.save_as))
