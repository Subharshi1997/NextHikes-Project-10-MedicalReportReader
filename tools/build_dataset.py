"""
Build the YOLO training dataset from the raw lab-report photos.

What this script does, in plain English
---------------------------------------
The 100 photos in the dataset are not 100 different reports. They are photos of
only THREE printed reports (a thyroid panel, a liver panel and a vitamin page),
each re-shot or augmented many times (rotation, perspective, crop, colour tint,
blur). So instead of drawing ~850 boxes by hand we:

1. Annotate ONE "master" photo per layout by hand (annotations/master_annotations.json).
2. Find each other photo's geometric relationship to its master with SIFT
   keypoints + a RANSAC homography (the same maths a phone uses to stitch a
   panorama).
3. Warp the master boxes onto every photo, clip them to the image, and drop
   boxes that are mostly cut off by the crop.
4. Group near-duplicate photos (perceptual hash) so near-identical copies never
   sit in both train and test, then make a layout-stratified split (target 16% val + 16% test; 74/12/14 for the 100
   photos, because whole near-duplicate groups must land in one split).
5. Write YOLO-format labels, data.yaml, and a ground-truth CSV for OCR scoring.
6. Warp the privacy boxes (patient name, phone, lab IDs) too, so every image we
   publish can be redacted with tools/privacy.py.

Every transferred box was checked visually (see results/02_annotation/).

Usage
-----
    python tools/build_dataset.py --images path/to/raw_jpgs --out datasets
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["test_name", "value", "unit", "reference_range"]
PAD = 2              # pixels added around each master box before warping
MIN_VISIBLE = 0.25   # keep a box for YOLO if >=25% of it is inside the photo
FULL_VISIBLE = 0.98  # use a field for OCR scoring only if it is fully visible
PHASH_MAX_DIST = 4   # Hamming distance at which two photos count as near-duplicates
SEED = 42


def detect_layout(img, masters):
    """Pick the master whose SIFT features match this photo best."""
    best = None
    for name, m in masters.items():
        H, inliers = homography(m["kp"], m["des"], img)
        if best is None or inliers > best[2]:
            best = (name, H, inliers)
    return best


_sift = cv2.SIFT_create(nfeatures=6000)
_bf = cv2.BFMatcher()


def features(img):
    """SIFT keypoints on a contrast-equalised grey image."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    return _sift.detectAndCompute(g, None)


def homography(mkp, mdes, img):
    """Master -> photo homography; returns (H, number of RANSAC inliers)."""
    kp, des = features(img)
    pairs = _bf.knnMatch(mdes, des, k=2)
    good = [a for a, b in pairs if a.distance < 0.75 * b.distance]   # Lowe ratio test
    if len(good) < 10:
        return None, 0
    src = np.float32([mkp[a.queryIdx].pt for a in good])
    dst = np.float32([kp[a.trainIdx].pt for a in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return H, int(mask.sum()) if mask is not None else 0


def warp_box(box, H, w, h):
    """Warp a master box into the photo; return clipped box and visible fraction."""
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = x0 - PAD, y0 - PAD, x1 + PAD, y1 + PAD
    pts = cv2.perspectiveTransform(np.float32([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]), H)[0]
    bx0, by0 = pts.min(0)
    bx1, by1 = pts.max(0)
    area = (bx1 - bx0) * (by1 - by0)
    cx0, cy0, cx1, cy1 = max(bx0, 0), max(by0, 0), min(bx1, w - 1), min(by1, h - 1)
    vis = max(0, cx1 - cx0) * max(0, cy1 - cy0) / area
    return [round(float(v), 1) for v in (cx0, cy0, cx1, cy1)], round(float(vis), 3)


def phash_groups(paths):
    """Union-find over perceptual hashes: photos within PHASH_MAX_DIST share a group."""
    hashes = [imagehash.phash(Image.open(p)) for p in paths]
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if hashes[i] - hashes[j] <= PHASH_MAX_DIST:
                parent[find(i)] = find(j)
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(len(paths))]


def stratified_group_split(df, n_val=8, n_test=8):
    """Assign whole near-duplicate groups to splits, roughly proportional per layout."""
    rng = random.Random(SEED)
    df = df.copy()
    df["split"] = "train"
    total = len(df)
    for layout, part in df.groupby("layout"):
        share = len(part) / total
        targets = {"test": round(n_test * share), "val": round(n_val * share)}
        groups = list(part.groupby("dup_group").size().items())
        rng.shuffle(groups)
        filled = {"test": 0, "val": 0}
        for gid, size in groups:
            for split in ("test", "val"):
                fits = filled[split] + size <= targets[split]
                if fits:
                    df.loc[(df.layout == layout) & (df.dup_group == gid), "split"] = split
                    filled[split] += size
                    break
    return df


def main(images_dir, out_dir, write_yolo=True, holdout=0.16):
    images_dir, out_dir = Path(images_dir), Path(out_dir)
    master = json.load(open(ROOT / "annotations" / "master_annotations.json", encoding="utf-8"))
    paths = sorted(images_dir.glob("*.jpg"))
    assert paths, f"no .jpg files in {images_dir}"

    # 1) master features
    masters = {}
    for lay, m in master["layouts"].items():
        img = cv2.imread(str(images_dir / m["master_image"]))
        kp, des = features(img)
        masters[lay] = {"kp": kp, "des": des, "boxes": m["boxes"], "file": m["master_image"],
                        "redact": m.get("redact", [])}

    # 2) transfer boxes to every photo
    records, ann = [], {}
    for p in paths:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        lay, H, inl = detect_layout(img, masters)
        if p.name == masters[lay]["file"]:
            H = np.eye(3)
        boxes = []
        for b in masters[lay]["boxes"]:
            box, vis = warp_box(b["box"], H, w, h)
            if vis < MIN_VISIBLE:
                continue
            boxes.append({"row": b["row"], "cls": b["cls"], "box": box, "visible": vis,
                          "text": b["text"], "full": vis >= FULL_VISIBLE})
        redact = []
        for r in masters[lay]["redact"]:                     # privacy regions follow the same warp
            rbox, vis = warp_box(r, H, w, h)
            if vis > 0:
                redact.append(rbox)
        ann[p.name] = {"layout": lay, "w": w, "h": h, "inliers": inl, "boxes": boxes, "redact": redact}
        records.append({"file": p.name, "layout": lay, "width": w, "height": h,
                        "sift_inliers": inl, "n_boxes": len(boxes)})

    df = pd.DataFrame(records)
    df["dup_group"] = phash_groups(paths)
    n_hold = max(1, round(holdout * len(df)))              # 8/8 for 50 photos, 16/16 for 100
    df = stratified_group_split(df, n_val=n_hold, n_test=n_hold)

    # 3) write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "split.csv", index=False)
    for name in ann:
        ann[name]["split"] = df.loc[df.file == name, "split"].item()
    json.dump(ann, open(out_dir / "annotations.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    gt_rows = []
    for name, a in ann.items():
        for b in a["boxes"]:
            gt_rows.append({"image": name, "split": a["split"], "layout": a["layout"], "row": b["row"],
                            "field": b["cls"], "text": b["text"], "fully_visible": b["full"]})
    pd.DataFrame(gt_rows).to_csv(out_dir / "ground_truth.csv", index=False, encoding="utf-8")

    if not write_yolo:
        return df, ann
    yolo = out_dir / "yolo"
    if yolo.exists():
        shutil.rmtree(yolo)
    for split in ("train", "val", "test"):
        (yolo / "images" / split).mkdir(parents=True)
        (yolo / "labels" / split).mkdir(parents=True)
    for name, a in ann.items():
        s = a["split"]
        shutil.copy(images_dir / name, yolo / "images" / s / name)
        lines = []
        for b in a["boxes"]:
            x0, y0, x1, y1 = b["box"]
            cx, cy = (x0 + x1) / 2 / a["w"], (y0 + y1) / 2 / a["h"]
            bw, bh = (x1 - x0) / a["w"], (y1 - y0) / a["h"]
            lines.append(f"{CLASSES.index(b['cls'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (yolo / "labels" / s / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n")
    (yolo / "data.yaml").write_text(
        # no 'path:' key -> Ultralytics resolves the folders relative to this yaml, so the
        # repo works wherever it is cloned (laptop, Colab, SageMaker)
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")

    print(df.groupby(["layout", "split"]).size().unstack(fill_value=0))
    print("boxes written:", sum(len(a["boxes"]) for a in ann.values()))
    return df, ann


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with the raw .jpg photos")
    ap.add_argument("--out", default=str(ROOT / "datasets"))
    ap.add_argument("--holdout", type=float, default=0.16, help="share of photos for val and for test")
    ap.add_argument("--no-yolo", action="store_true", help="only refresh annotations/split/ground truth")
    args = ap.parse_args()
    main(args.images, args.out, write_yolo=not args.no_yolo, holdout=args.holdout)
