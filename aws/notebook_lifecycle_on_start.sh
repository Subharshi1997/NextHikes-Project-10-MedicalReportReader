#!/bin/bash
# SageMaker notebook-instance lifecycle configuration ("Start notebook" script) - brief Task 1.2.
# Runs as root every time the instance starts, so the environment survives stop/start:
#   * installs the Tesseract OCR engine
#   * creates a "custom-ocr" Jupyter kernel with PyTorch, Ultralytics, OpenCV, pytesseract, boto3
#   * auto-stops the instance after 60 idle minutes (cost control)
# Paste it in: SageMaker console -> Notebook -> Lifecycle configurations -> Create -> Start notebook.
# (Create it with: aws sagemaker create-notebook-instance-lifecycle-config ... via aws/sm_jobs.py notebook)
set -e

# --- 1. Tesseract (Amazon Linux 2 / AL2023 notebook images) --------------------------------
if ! command -v tesseract >/dev/null 2>&1; then
  (yum install -y tesseract tesseract-langpack-eng 2>/dev/null || dnf install -y tesseract tesseract-langpack-eng 2>/dev/null) \
  || sudo -u ec2-user -i bash -c "conda install -y -n base -c conda-forge tesseract"
fi

# --- 2. Python kernel for the project (as ec2-user, in the background: start must finish < 5 min)
sudo -u ec2-user -i nohup bash -c '
  set -e
  ENV=/home/ec2-user/SageMaker/envs/custom-ocr
  if [ ! -d "$ENV" ]; then
    python3 -m venv "$ENV"
    "$ENV/bin/pip" -q install --upgrade pip
    "$ENV/bin/pip" -q install ultralytics opencv-python-headless pytesseract pandas matplotlib pillow \
        imagehash boto3 openpyxl ipykernel
  fi
  "$ENV/bin/python" -m ipykernel install --user --name custom-ocr --display-name "Python (custom-ocr)"
' > /home/ec2-user/SageMaker/lifecycle_setup.log 2>&1 &

# --- 3. Stop the instance after 60 idle minutes so a forgotten notebook does not keep billing
IDLE_MIN=60
cat > /usr/local/bin/autostop.py <<'PY'
"""Stop this notebook instance when every Jupyter kernel has been idle for N minutes."""
import datetime as dt, json, ssl, sys, urllib.request
import boto3
idle_s = int(sys.argv[1]) * 60
try:      # Jupyter on SageMaker notebook instances listens on https://localhost:8443
    ks = json.load(urllib.request.urlopen("https://localhost:8443/api/kernels", timeout=5,
                                          context=ssl._create_unverified_context()))
except Exception:
    sys.exit(0)  # cannot tell -> assume busy
now = dt.datetime.utcnow()
last = max([dt.datetime.strptime(k["last_activity"][:19], "%Y-%m-%dT%H:%M:%S") for k in ks] or [now])
if all(k["execution_state"] == "idle" for k in ks) and (now - last).total_seconds() > idle_s:
    name = json.load(open("/opt/ml/metadata/resource-metadata.json"))["ResourceName"]
    boto3.client("sagemaker").stop_notebook_instance(NotebookInstanceName=name)
PY
PY_BIN=/home/ec2-user/anaconda3/bin/python   # ships with boto3 on notebook instances
(crontab -l 2>/dev/null; echo "*/5 * * * * $PY_BIN /usr/local/bin/autostop.py $IDLE_MIN") | sort -u | crontab -
