"""
Run the project on Amazon SageMaker + S3 (the "on AWS" part of the brief), using plain boto3.

Every step is one sub-command. Add --dry-run to any of them to print the exact API request
without calling AWS (no credentials needed) - useful for review before spending money.

  0. one-off: an IAM role SageMaker can assume, with AmazonSageMakerFullAccess + S3 access to
     the bucket (console: IAM -> Roles -> Create role -> SageMaker). Pass it as --role.
  1. python aws/s3_sync.py --bucket B create && python aws/s3_sync.py --bucket B push   (data -> S3)
  2. python aws/sm_jobs.py image    --gpu        # build the Docker image, push it to ECR
     python aws/sm_jobs.py image                 # CPU image for inference
  3. python aws/sm_jobs.py notebook --role R --bucket B       # Task 1.2: notebook instance
  4. python aws/sm_jobs.py train    --role R --bucket B       # Task 3.1: YOLOv3 training job (spot GPU)
     python aws/sm_jobs.py fetch    --job <job-name>          #   -> model/sagemaker_best.pt
  5. python aws/sm_jobs.py deploy   --role R --model-data s3://.../model.tar.gz   # serverless endpoint
     python aws/sm_jobs.py invoke   --image samples/thyroid_report.jpg           # -> CSV
  6. python aws/sm_jobs.py batch    --role R --bucket B       # Processing job: S3 photos -> S3 CSVs
  7. python aws/sm_jobs.py cleanup                            # delete endpoint, stop notebook ($0 idle)

Cost (approximate us-east-1 on-demand rates; confirm on the SageMaker pricing page for your region):
  notebook ml.t3.medium ~$0.05/h (auto-stops after 60 idle min) | training ml.g4dn.xlarge ~$0.74/h,
  ~70% less on spot, roughly 1 h for YOLOv3 x 100 epochs | serverless endpoint: pay per request, $0 idle |
  S3: ~0.1 GB -> < $0.01/month. The whole project end to end: well under $1.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
REPO_NAME = "custom-ocr"
NAME = "custom-ocr"
DRY = False


def call(client, op: str, **req):
    """Run one AWS API call, or print it when --dry-run is on."""
    if DRY:
        print(f"[dry-run] {client.meta.service_model.service_name}.{op}(")
        print(json.dumps(req, indent=2, default=str))
        print(")")
        return {}
    return getattr(client, op)(**req)


def account_region(session) -> tuple[str, str]:
    region = session.region_name or "us-east-1"
    if DRY:
        return "123456789012", region
    return session.client("sts").get_caller_identity()["Account"], region


def image_uri(session, tag: str) -> str:
    acct, region = account_region(session)
    return f"{acct}.dkr.ecr.{region}.amazonaws.com/{REPO_NAME}:{tag}"


def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


# --------------------------------------------------------------------------- 2. image -> ECR
def cmd_image(a, session):
    tag = "gpu" if a.gpu else "cpu"
    uri = image_uri(session, tag)
    build = ["docker", "build", "-t", uri, str(ROOT)]
    if a.gpu:
        build[2:2] = ["--build-arg", "TORCH_INDEX=https://download.pytorch.org/whl/cu128"]
    ecr = session.client("ecr")
    if DRY:
        print("[dry-run]", " ".join(build), "\n[dry-run] docker push", uri)
        return
    try:
        ecr.create_repository(repositoryName=REPO_NAME, imageScanningConfiguration={"scanOnPush": True})
    except ecr.exceptions.RepositoryAlreadyExistsException:
        pass
    tok = ecr.get_authorization_token()["authorizationData"][0]
    user, pwd = base64.b64decode(tok["authorizationToken"]).decode().split(":", 1)
    subprocess.run(["docker", "login", "-u", user, "--password-stdin", tok["proxyEndpoint"]],
                   input=pwd.encode(), check=True)
    subprocess.run(build, check=True)
    subprocess.run(["docker", "push", uri], check=True)
    print("pushed", uri)


# --------------------------------------------------------------------------- 3. notebook instance
def cmd_notebook(a, session):
    sm = session.client("sagemaker")
    script = (ROOT / "aws" / "notebook_lifecycle_on_start.sh").read_text()
    lc = f"{NAME}-on-start"
    call(sm, "create_notebook_instance_lifecycle_config", NotebookInstanceLifecycleConfigName=lc,
         OnStart=[{"Content": base64.b64encode(script.encode()).decode()}])
    call(sm, "create_notebook_instance", NotebookInstanceName=a.name, InstanceType=a.instance,
         RoleArn=a.role, LifecycleConfigName=lc, VolumeSizeInGB=10,
         DefaultCodeRepository=a.git_repo, Tags=[{"Key": "project", "Value": NAME}])
    print(f"notebook '{a.name}' requested; open it from the SageMaker console when InService.\n"
          f"In a terminal there:  python aws/s3_sync.py --bucket {a.bucket} pull")


# --------------------------------------------------------------------------- 4. training job
def cmd_train(a, session):
    sm = session.client("sagemaker")
    job = f"{NAME}-{Path(a.weights).stem}-{stamp()}"
    req = dict(
        TrainingJobName=job,
        AlgorithmSpecification={"TrainingImage": a.image_uri or image_uri(session, "gpu"),
                                "TrainingInputMode": "File",
                                "MetricDefinitions": [
                                    {"Name": "mAP50", "Regex": r"all\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)"}]},
        RoleArn=a.role,
        HyperParameters={"weights": a.weights, "epochs": str(a.epochs), "batch": str(a.batch), "imgsz": "1024"},
        InputDataConfig=[{"ChannelName": "training", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": f"s3://{a.bucket}/{a.prefix}/datasets/yolo/",
            "S3DataDistributionType": "FullyReplicated"}}}],
        OutputDataConfig={"S3OutputPath": f"s3://{a.bucket}/{a.prefix}/sagemaker/"},
        ResourceConfig={"InstanceType": a.instance, "InstanceCount": 1, "VolumeSizeInGB": 30},
        StoppingCondition={"MaxRuntimeInSeconds": 3 * 3600},
        Tags=[{"Key": "project", "Value": NAME}],
    )
    if a.spot:      # managed spot: up to ~70% cheaper; SageMaker restarts the job if interrupted
        req["EnableManagedSpotTraining"] = True
        req["StoppingCondition"]["MaxWaitTimeInSeconds"] = 6 * 3600
    call(sm, "create_training_job", **req)
    print("training job:", job)
    if a.wait and not DRY:
        cmd_fetch(argparse.Namespace(job=job), session)


def cmd_fetch(a, session):
    """Wait for a training job, then copy its best.pt into model/sagemaker_best.pt."""
    sm = session.client("sagemaker")
    if DRY:
        print("[dry-run] wait for", a.job, "then download model.tar.gz")
        return
    sm.get_waiter("training_job_completed_or_stopped").wait(TrainingJobName=a.job)
    d = sm.describe_training_job(TrainingJobName=a.job)
    print(a.job, d["TrainingJobStatus"], f"billable {d.get('BillableTimeInSeconds', 0)} s")
    if d["TrainingJobStatus"] != "Completed":
        sys.exit(d.get("FailureReason", "failed"))
    uri = d["ModelArtifacts"]["S3ModelArtifacts"]
    bucket, key = uri[5:].split("/", 1)
    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "model.tar.gz"
        session.client("s3").download_file(bucket, key, str(tgz))
        with tarfile.open(tgz) as t:
            t.extract("best.pt", tmp, filter="data")
        (ROOT / "model").mkdir(exist_ok=True)
        Path(tmp, "best.pt").replace(ROOT / "model" / "sagemaker_best.pt")
    print("model artifacts:", uri, "\nsaved model/sagemaker_best.pt")


# --------------------------------------------------------------------------- 5. endpoint
def cmd_deploy(a, session):
    sm = session.client("sagemaker")
    model_name, cfg = f"{NAME}-{stamp()}", f"{NAME}-cfg-{stamp()}"
    container = {"Image": a.image_uri or image_uri(session, "cpu"),
                 "Environment": {"OCR_MODE": "tuned", "OCR_CONF": "0.25"}}
    if a.model_data:            # weights from a training job; otherwise the ones baked into the image
        container["ModelDataUrl"] = a.model_data
    call(sm, "create_model", ModelName=model_name, ExecutionRoleArn=a.role, PrimaryContainer=container)
    variant = {"VariantName": "AllTraffic", "ModelName": model_name}
    if a.instance == "serverless":   # pay per request, scales to zero: right for a low-volume OCR tool
        variant["ServerlessConfig"] = {"MemorySizeInMB": 6144, "MaxConcurrency": 2}
    else:
        variant.update(InstanceType=a.instance, InitialInstanceCount=1)
    call(sm, "create_endpoint_config", EndpointConfigName=cfg, ProductionVariants=[variant])
    exists = False
    if not DRY:
        try:
            sm.describe_endpoint(EndpointName=a.endpoint)
            exists = True
        except sm.exceptions.ClientError:
            pass
    call(sm, "update_endpoint" if exists else "create_endpoint", EndpointName=a.endpoint, EndpointConfigName=cfg)
    if not DRY:
        print("waiting for endpoint", a.endpoint, "...")
        sm.get_waiter("endpoint_in_service").wait(EndpointName=a.endpoint)
    print("endpoint ready:", a.endpoint)


def cmd_invoke(a, session):
    rt = session.client("sagemaker-runtime")
    body = Path(a.image).read_bytes()
    ext = Path(a.image).suffix.lower()
    ctype = {".png": "image/png", ".pdf": "application/pdf"}.get(ext, "image/jpeg")
    if DRY:
        print(f"[dry-run] sagemaker-runtime.invoke_endpoint({a.endpoint}, {len(body)} bytes, {ctype})")
        return
    r = rt.invoke_endpoint(EndpointName=a.endpoint, ContentType=ctype, Accept="text/csv", Body=body)
    print(r["Body"].read().decode())


# --------------------------------------------------------------------------- 6. batch processing job
def cmd_batch(a, session):
    """Brief section 6 'automation': OCR every photo under an S3 prefix, write CSV + Excel back to S3."""
    sm = session.client("sagemaker")
    job = f"{NAME}-batch-{stamp()}"
    src = a.input or f"s3://{a.bucket}/{a.prefix}/inbox/"
    dst = a.output or f"s3://{a.bucket}/{a.prefix}/results/batch/{job}/"
    call(sm, "create_processing_job",
         ProcessingJobName=job, RoleArn=a.role,
         AppSpecification={"ImageUri": a.image_uri or image_uri(session, "cpu"),
                           "ContainerEntrypoint": ["entrypoint.sh", "ocr", "--image", "/opt/ml/processing/input",
                                                   "--out", "/opt/ml/processing/output", "--excel"]},
         ProcessingInputs=[{"InputName": "photos", "S3Input": {
             "S3Uri": src, "LocalPath": "/opt/ml/processing/input", "S3DataType": "S3Prefix",
             "S3InputMode": "File"}}],
         ProcessingOutputConfig={"Outputs": [{"OutputName": "csv", "S3Output": {
             "S3Uri": dst, "LocalPath": "/opt/ml/processing/output", "S3UploadMode": "EndOfJob"}}]},
         ProcessingResources={"ClusterConfig": {"InstanceCount": 1, "InstanceType": a.instance,
                                                "VolumeSizeInGB": 10}},
         StoppingCondition={"MaxRuntimeInSeconds": 3600},
         Tags=[{"Key": "project", "Value": NAME}])
    print(f"processing job {job}: {src} -> {dst}")


# --------------------------------------------------------------------------- 7. cleanup
def cmd_cleanup(a, session):
    """Delete the endpoint (+ its configs and models) and stop the notebook: nothing left billing."""
    sm = session.client("sagemaker")
    if DRY:
        print(f"[dry-run] delete endpoint {a.endpoint}, its endpoint configs and models; stop notebook {a.notebook}")
        return
    try:
        cfg = sm.describe_endpoint(EndpointName=a.endpoint)["EndpointConfigName"]
        models = [v["ModelName"] for v in sm.describe_endpoint_config(EndpointConfigName=cfg)["ProductionVariants"]]
        sm.delete_endpoint(EndpointName=a.endpoint)
        sm.delete_endpoint_config(EndpointConfigName=cfg)
        for m in models:
            sm.delete_model(ModelName=m)
        print("deleted endpoint", a.endpoint, "config", cfg, "models", models)
    except sm.exceptions.ClientError as e:
        print("endpoint:", e.response["Error"]["Message"])
    try:
        if sm.describe_notebook_instance(NotebookInstanceName=a.notebook)["NotebookInstanceStatus"] == "InService":
            sm.stop_notebook_instance(NotebookInstanceName=a.notebook)
            print("stopping notebook", a.notebook)
    except sm.exceptions.ClientError as e:
        print("notebook:", e.response["Error"]["Message"])


def main(argv=None):
    global DRY
    ap = argparse.ArgumentParser(description="Custom OCR on SageMaker + S3",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--region", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the API requests, call nothing")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, fn, *, role=False, bucket=False):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        if role:
            s.add_argument("--role", required=True, help="SageMaker execution role ARN")
        if bucket:
            s.add_argument("--bucket", required=True)
            s.add_argument("--prefix", default="custom-ocr")
        return s

    s = p("image", cmd_image)
    s.add_argument("--gpu", action="store_true", help="CUDA torch (for training jobs)")
    s = p("notebook", cmd_notebook, role=True, bucket=True)
    s.add_argument("--name", default=f"{NAME}-notebook")
    s.add_argument("--instance", default="ml.t3.medium")
    s.add_argument("--git-repo", default="https://github.com/Subharshi1997/NextHikes-Project-9-MedicalReportReader.git")
    s = p("train", cmd_train, role=True, bucket=True)
    s.add_argument("--weights", default="yolov3u.pt", help="yolov3u.pt (brief) or yolo26n.pt")
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--instance", default="ml.g4dn.xlarge")
    s.add_argument("--no-spot", dest="spot", action="store_false")
    s.add_argument("--wait", action="store_true")
    s.add_argument("--image-uri", default=None)
    s = p("fetch", cmd_fetch)
    s.add_argument("--job", required=True)
    s = p("deploy", cmd_deploy, role=True)
    s.add_argument("--model-data", default=None, help="s3://.../model.tar.gz from a training job")
    s.add_argument("--endpoint", default=f"{NAME}-endpoint")
    s.add_argument("--instance", default="serverless", help="'serverless' or e.g. ml.m5.large")
    s.add_argument("--image-uri", default=None)
    s = p("invoke", cmd_invoke)
    s.add_argument("--image", required=True, help="photo or PDF")
    s.add_argument("--endpoint", default=f"{NAME}-endpoint")
    s = p("batch", cmd_batch, role=True, bucket=True)
    s.add_argument("--input", default=None, help="S3 prefix with photos (default <prefix>/inbox/)")
    s.add_argument("--output", default=None)
    s.add_argument("--instance", default="ml.t3.medium")
    s.add_argument("--image-uri", default=None)
    s = p("cleanup", cmd_cleanup)
    s.add_argument("--endpoint", default=f"{NAME}-endpoint")
    s.add_argument("--notebook", default=f"{NAME}-notebook")

    a = ap.parse_args(argv)
    DRY = a.dry_run
    a.fn(a, boto3.session.Session(region_name=a.region))


if __name__ == "__main__":
    main()
