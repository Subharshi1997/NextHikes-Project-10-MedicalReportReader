"""
Mirror the project's brief folder structure (datasets/, model/, results/) to and from Amazon S3.
S3 plays the role Google Drive plays in the Colab version.

    python aws/s3_sync.py --bucket my-ocr-bucket create              # make the bucket (private, encrypted)
    python aws/s3_sync.py --bucket my-ocr-bucket push datasets model  # laptop/Colab -> S3
    python aws/s3_sync.py --bucket my-ocr-bucket pull model results   # S3 -> SageMaker notebook
    python aws/s3_sync.py --bucket my-ocr-bucket ls

Layout in the bucket:  s3://<bucket>/<prefix>/datasets/raw/*.jpg
                                              /datasets/yolo/{images,labels}/...
                                              /model/best.pt
                                              /results/...
Only changed files are uploaded (size + MD5 compared with the S3 ETag), so re-running is cheap.
The raw photos show a patient's details: the bucket is created with all public access blocked
and default encryption, and nothing here ever makes an object public.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("datasets", "model", "results")
SKIP = {".cache", ".pyc"}


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_index(s3, bucket: str, prefix: str, dry: bool = False) -> dict:
    out = {}
    try:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for o in page.get("Contents", []):
                out[o["Key"]] = (o["Size"], o["ETag"].strip('"'))
    except (NoCredentialsError, ClientError, EndpointConnectionError) as e:
        if not dry:
            raise
        print(f"(dry run: cannot list s3://{bucket}/{prefix} [{type(e).__name__}], treating it as empty)")
    return out


def create_bucket(s3, bucket: str, region: str):
    try:
        kw = {} if region == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": region}}
        s3.create_bucket(Bucket=bucket, **kw)
        print("created", bucket)
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou",):
            raise
        print("exists ", bucket)
    s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    s3.put_bucket_encryption(Bucket=bucket, ServerSideEncryptionConfiguration={
        "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]})


def push(s3, bucket: str, prefix: str, folders, dry=False):
    for folder in folders:
        base = ROOT / folder
        remote = remote_index(s3, bucket, f"{prefix}/{folder}/", dry)
        sent = skipped = 0
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix in SKIP:
                continue
            key = f"{prefix}/{p.relative_to(ROOT).as_posix()}"
            r = remote.get(key)
            if r and r[0] == p.stat().st_size and (("-" in r[1]) or r[1] == _md5(p)):
                skipped += 1            # multipart ETags are not MD5s: same size is good enough there
                continue
            if not dry:
                s3.upload_file(str(p), bucket, key)
            sent += 1
        print(f"{folder:9s} uploaded {sent}, unchanged {skipped}{' (dry run)' if dry else ''}")


def pull(s3, bucket: str, prefix: str, folders, dry=False):
    for folder in folders:
        got = 0
        for key, (size, _) in remote_index(s3, bucket, f"{prefix}/{folder}/", dry).items():
            dest = ROOT / key[len(prefix) + 1:]
            if dest.exists() and dest.stat().st_size == size:
                continue
            if not dry:
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(dest))
            got += 1
        print(f"{folder:9s} downloaded {got}{' (dry run)' if dry else ''}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="custom-ocr")
    ap.add_argument("--region", default=None, help="default: your AWS config / AWS_REGION")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("action", choices=["create", "push", "pull", "ls"])
    ap.add_argument("folders", nargs="*", default=list(FOLDERS))
    a = ap.parse_args(argv)
    session = boto3.session.Session(region_name=a.region)
    s3 = session.client("s3")
    bad = set(a.folders) - set(FOLDERS)
    if bad:
        ap.error(f"folders must be among {FOLDERS}")
    if a.action == "create":
        create_bucket(s3, a.bucket, session.region_name or "us-east-1")
    elif a.action == "push":
        push(s3, a.bucket, a.prefix, a.folders, a.dry_run)
    elif a.action == "pull":
        pull(s3, a.bucket, a.prefix, a.folders, a.dry_run)
    else:
        for folder in a.folders:
            idx = remote_index(s3, a.bucket, f"{a.prefix}/{folder}/")
            print(f"{folder:9s} {len(idx):5d} objects  {sum(s for s, _ in idx.values()) / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
