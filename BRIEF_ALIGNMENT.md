# Alignment with the project brief ("Project 10: Custom-Object Character Recognition (OCR) on AWS")

Every requirement in the brief, in the brief's order, with where it is met and how it was checked.

**Status key:** ✅ met and verified by running it · 🟡 met with a stated deviation · ⚪ implemented, but not runnable here (needs an AWS account)

## Overview, Model, Data, Dependencies

| # | Brief says | Status | Where / evidence |
|---|---|---|---|
| 1 | Custom OCR combining YOLO and Tesseract to read a lab report | ✅ | `Custom_OCR.py`: YOLO finds 4 field types, Tesseract reads each crop |
| 2 | Convert it into an editable file | ✅ | CSV per photo + `all_reports.csv`; Excel with `--excel`; editable table + CSV/XLSX download in the web app |
| 3 | Use YOLO_V3 trained on the personal dataset | 🟡 | YOLOv3 (Ultralytics `yolov3u`, Darknet-53 backbone) trained for 100 epochs on the dataset: `model/yolov3u_best.pt`, val mAP@.5:.95 0.818, test 0.806. **Deviation:** YOLO26n was trained on the same split and is the shipped `model/best.pt`, chosen by a rule fixed before test scoring (YOLO26n +0.060 on validation; 89.2% vs 82.9% exact match on test). YOLOv3 is one flag away: `--weights model/yolov3u_best.pt`. The `u` variant replaces the 2018 anchor-based head with Ultralytics' anchor-free head, so it installs with pip instead of a Darknet compile. |
| 4 | Coordinates of detections → crop → store in another list | ✅ | `read_fields()`: `crops = [crop(img, d["box"], cfg.pad) for d in dets]` |
| 5 | That list is passed through Tesseract | ✅ | `tesseract_read()` on every crop |
| 6 | Make a folder named `model` and put the weights inside | ✅ | `model/best.pt`, `model/yolov3u_best.pt`, `model/yolo26n_best.pt` |
| 7 | Dataset from the Google Drive link | ✅ | All **100** photos used (the original repo had used only 50); 1,760 labels, every box checked visually (`results/02_annotation/qa_new_photos/`) |
| 8 | Install Tesseract, pytesseract, OpenCV | ✅ | `requirements.txt`; Tesseract auto-detected on Windows even when not on PATH (`find_tesseract()`). The brief's `pytessercat` / `pip opencv` are typos for `pytesseract` / `opencv-python`. |
| 9 | Output like the brief's picture: Test Name, Unit, Reference, Value | ✅ | CSV columns `test_name, value, unit, reference_range` + confidence + `needs_review` |
| + | *(beyond the brief)* PDF input | ✅ | Scanned, searchable and digital PDFs, multi-page (`load_pages()` in `Custom_OCR.py`). Digital PDFs are read from their text layer: 44/44 fields exact on `samples/liver_report_digital.pdf`. Photo results unchanged (89.2% re-verified). |

## Getting started and the step-by-step workflow

| # | Brief says | Status | Where / evidence |
|---|---|---|---|
| 10 | `python Custom_OCR.py --image <yourimage.jpg>` | ✅ | Runs as written (verified on `samples/liver_report.jpg`); also accepts a folder |
| 11 | Take the region inside each YOLOv3 box | ✅ | `crop()` |
| 12 | `cv2.resize()` 3x | ✅ | `preprocess()`, `scale=3.0` in both modes |
| 13 | Grayscale + small Gaussian blur | ✅ / 🟡 | `--mode brief` uses a 5x5 blur exactly as described. The default `tuned` mode drops the blur, because the ablation on training photos showed it lowers accuracy on 3x-upscaled small print (notebook 05). |
| 14 | Threshold to white-on-black with Otsu | ✅ | `cv2.THRESH_BINARY_INV \| cv2.THRESH_OTSU` (removing it drops accuracy from 71% to 31%) |
| 15 | `bitwise_not` to black-on-white for Tesseract | ✅ | `cv2.bitwise_not(thresh)` |
| 16 | Output saved as a CSV file | ✅ | `results/04_inference/*.csv` |

## Task list

| # | Brief task | Status | Where / evidence |
|---|---|---|---|
| 1.1 | Colab: mount Drive; folders `datasets/`, `models/`, `results/` | ✅ | First cell of every notebook mounts Drive and creates `datasets/`, `model/`, `results/` (the brief's Model section says `model`, Task 1.1 says `models`; `model` is used). Ran locally here (no Colab), where "Drive" = the repo folder. |
| 1.2 | Launch a SageMaker notebook instance, install Tesseract / PyTesseract / OpenCV / YOLOv3, give it S3 access | ⚪ | `python aws/sm_jobs.py notebook` + `aws/notebook_lifecycle_on_start.sh` (installs Tesseract + a kernel with Ultralytics/OpenCV/pytesseract/boto3, auto-stops when idle); the instance's IAM role gives S3 access; `aws/s3_sync.py pull` fetches the data. Request payload verified with `--dry-run`; not run on a live account. |
| 2.1 / 2.2 | Upload the dataset, resize, create YOLO annotations, store preprocessed data | ✅ | Notebooks 01-02, `tools/build_dataset.py`. Images are resized by YOLO to 1024 px at train time rather than stored resized, which keeps the original resolution for OCR. |
| 3.1 | Train YOLOv3 on a GPU, save weights to `model`, validate on a subset, save validation images with boxes | ✅ | Notebook 03: YOLOv3 + YOLO26n trained on an RTX 3050 GPU, weights in `model/`, per-class validation metrics, redacted validation images in `results/03_training/val_predictions/`. The same training also runs as a SageMaker training job (`aws/train_entry.py`, tested in the Docker image against SageMaker's `/opt/ml` contract). |
| 3.2 / 4 | Inference on test images → crop → preprocess → Tesseract → CSV | ✅ | Notebook 04, `results/04_inference/test_set/` |
| 5 | Compare OCR output with ground truth; fine-tune model, preprocessing, Tesseract | ✅ | Notebook 05: exact match + CER per field on 14 unseen photos; one-knob ablation of preprocessing on training photos only; YOLOv3 vs YOLO26n head-to-head; unit lexicon; error analysis |
| 6 | Automation and deployment | ✅ / ⚪ | ✅ batch CLI, web app + JSON API, Docker image (all run and tested here). ⚪ SageMaker serverless endpoint + Processing job (`aws/sm_jobs.py deploy / batch`): the container passes SageMaker's `/ping` and `/invocations` contract locally; not deployed to a live account. (The brief's section 6 text duplicates section 5, so it is read by its title.) |
| 7 | Document the pipeline; save final reports in Drive | ✅ | Six notebooks, `README.md`, `reports/Custom_OCR_Lab_Reports_Report.pdf` (regenerated from `results/`), this file |
| Cost | Estimate: $0 on Colab + Drive | ✅ | $0 on Colab/Drive; AWS path documented at well under $1 with idle auto-stop, spot training, a serverless endpoint and a `cleanup` command |

## What would be needed to close the ⚪ items

An AWS account with an S3 bucket and a SageMaker execution role, then: `aws/s3_sync.py create/push` → `aws/sm_jobs.py image` → `notebook`, `train`, `deploy`, `invoke`, `batch` → `cleanup`. Every one of these prints its exact request with `--dry-run` today.
