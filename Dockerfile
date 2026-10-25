# One image for every way the pipeline runs:
#   docker run -p 8080:8080 custom-ocr serve            web app + API (+ SageMaker endpoint contract)
#   docker run -v $PWD/in:/in custom-ocr ocr --image /in --out /in/out     batch CLI
#   SageMaker training job: SageMaker runs `docker run <image> train`
#
# CPU torch by default (small image, fine for inference ~2 s/photo). For a GPU training image:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t custom-ocr:gpu .
FROM python:3.11-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# Tesseract OCR engine (+ English data) and the shared libraries OpenCV needs
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program
COPY requirements.txt requirements-deploy.txt ./
RUN pip install torch torchvision --index-url ${TORCH_INDEX} \
    && pip install -r requirements.txt -r requirements-deploy.txt

COPY Custom_OCR.py ./
COPY app/ app/
COPY tools/ tools/
COPY aws/ aws/
COPY model/ model/
COPY samples/ samples/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["entrypoint.sh"]
CMD ["serve"]
