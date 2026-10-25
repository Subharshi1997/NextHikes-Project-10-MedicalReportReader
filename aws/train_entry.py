"""
Runs INSIDE a SageMaker training job (the container is started as `docker run <image> train`).

SageMaker's file contract
-------------------------
/opt/ml/input/data/training/          the S3 channel, i.e. s3://<bucket>/<prefix>/datasets/yolo/
    images/{train,val,test}/  labels/{train,val,test}/
/opt/ml/input/config/hyperparameters.json   {"weights": "yolov3u.pt", "epochs": "100", ...} (all strings)
/opt/ml/model/                        everything here is tarred to model.tar.gz in S3 -> best.pt goes here
/opt/ml/output/failure                a failure message, shown in the SageMaker console

Set SM_ROOT to try it locally:  SM_ROOT=/tmp/fake_opt_ml python aws/train_entry.py
"""
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

SM = Path(os.environ.get("SM_ROOT", "/opt/ml"))
DATA = SM / "input" / "data" / "training"
HP_FILE = SM / "input" / "config" / "hyperparameters.json"
MODEL_DIR = SM / "model"
FAIL = SM / "output" / "failure"

DEFAULTS = {"weights": "yolov3u.pt", "epochs": "100", "batch": "8", "imgsz": "1024", "device": "0"}


def main():
    hp = {**DEFAULTS, **(json.loads(HP_FILE.read_text()) if HP_FILE.exists() else {})}
    # the S3 copy of data.yaml may carry another machine's paths, and the input channel can be
    # read-only, so write a fresh yaml (absolute path to the channel) in a temp folder
    import tempfile
    yaml = Path(tempfile.mkdtemp()) / "data_sagemaker.yaml"
    yaml.write_text(f"path: {DATA.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                    "nc: 4\nnames: ['test_name', 'value', 'unit', 'reference_range']\n")

    from ultralytics import YOLO
    import train as tr                                    # same augmentation settings as tools/train.py

    device = hp["device"]
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"
    work = SM / "output" / "data"                         # uploaded as output.tar.gz
    model = YOLO(hp["weights"])
    model.train(data=str(yaml), epochs=int(hp["epochs"]), batch=int(hp["batch"]), device=device,
                project=str(work), name="run", exist_ok=True, **{**tr.HYP, "imgsz": int(hp["imgsz"])})

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    run = work / "run"
    shutil.copy(run / "weights" / "best.pt", MODEL_DIR / "best.pt")     # the endpoint loads this
    shutil.copy(run / "results.csv", MODEL_DIR / "results.csv")
    print("saved", MODEL_DIR / "best.pt")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        FAIL.parent.mkdir(parents=True, exist_ok=True)
        FAIL.write_text(traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)
