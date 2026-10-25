#!/bin/sh
# serve -> web app / API on :8080 (SageMaker endpoints call `docker run <image> serve`)
# train -> SageMaker training job (`docker run <image> train`), see aws/train_entry.py
# ocr   -> batch CLI, remaining arguments go to Custom_OCR.py
set -e
cd /opt/program
case "$1" in
  serve) shift; exec uvicorn app.server:app --host 0.0.0.0 --port "${PORT:-8080}" "$@" ;;
  train) shift; exec python aws/train_entry.py "$@" ;;
  ocr)   shift; exec python Custom_OCR.py "$@" ;;
  *)     exec "$@" ;;
esac
