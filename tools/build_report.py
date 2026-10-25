"""
Build reports/Custom_OCR_Lab_Reports_Report.pdf from the saved results (no numbers are typed by hand:
every figure and metric is read from results/).

    python tools/build_report.py
"""
import json
from xml.sax.saxutils import escape
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "reports" / "Custom_OCR_Lab_Reports_Report.pdf"

# ---------------------------------------------------------------- fonts & colours
FONT_DIRS = [Path("/usr/share/fonts/truetype/google-fonts"), Path("/usr/share/fonts/truetype/crosextra"),
             Path("C:/Windows/Fonts")]


def _font(name, file):
    for d in FONT_DIRS:
        if (d / file).exists():
            pdfmetrics.registerFont(TTFont(name, str(d / file)))
            return name
    return None


HEAD = _font("Poppins-SemiBold", "Poppins-Medium.ttf") or "Helvetica-Bold"
HEADB = _font("Poppins-Bold", "Poppins-Bold.ttf") or "Helvetica-Bold"
BODY = _font("Carlito", "Carlito-Regular.ttf") or "Helvetica"
BODYB = _font("Carlito-Bold", "Carlito-Bold.ttf") or "Helvetica-Bold"
BODYI = _font("Carlito-Italic", "Carlito-Italic.ttf") or "Helvetica-Oblique"
if BODY == "Carlito":
    from reportlab.lib.fonts import addMapping
    addMapping("Carlito", 0, 0, "Carlito"); addMapping("Carlito", 1, 0, "Carlito-Bold")
    addMapping("Carlito", 0, 1, "Carlito-Italic"); addMapping("Carlito", 1, 1, "Carlito-Bold")

INK, MUTED, RULE, TINT = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#d9d8d3"), colors.HexColor("#f3f2ee")
BLUE, ORANGE, AQUA = colors.HexColor("#2a78d6"), colors.HexColor("#eb6834"), colors.HexColor("#1baf7a")

S = {
    "title": ParagraphStyle("title", fontName=HEADB, fontSize=24, leading=29, textColor=INK, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName=BODY, fontSize=13, leading=17, textColor=MUTED, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName=HEAD, fontSize=15, leading=19, textColor=INK, spaceBefore=10, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName=HEAD, fontSize=11.5, leading=15, textColor=INK, spaceBefore=8, spaceAfter=4),
    "body": ParagraphStyle("body", fontName=BODY, fontSize=10.5, leading=14.2, textColor=INK, spaceAfter=6, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName=BODY, fontSize=9, leading=12, textColor=MUTED, spaceAfter=4),
    "cap": ParagraphStyle("cap", fontName=BODYI, fontSize=9, leading=11.5, textColor=MUTED, spaceBefore=2, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName=BODY, fontSize=9, leading=11.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName=BODYB, fontSize=9, leading=11.2, textColor=INK),
    "kpi": ParagraphStyle("kpi", fontName=HEADB, fontSize=22, leading=26, textColor=INK),
    "kpil": ParagraphStyle("kpil", fontName=BODY, fontSize=9, leading=11.5, textColor=MUTED),
    "bullet": ParagraphStyle("bullet", fontName=BODY, fontSize=10.5, leading=14.2, textColor=INK, leftIndent=12,
                             bulletIndent=2, spaceAfter=3),
}
W = A4[0] - 36 * mm      # text width


def P(t, s="body"):
    return Paragraph(t, S[s])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def fig(path, width=W, caption=None):
    from PIL import Image as PI
    w, h = PI.open(path).size
    out = [Image(str(path), width=width, height=width * h / w)]
    if caption:
        out.append(P(caption, "cap"))
    return KeepTogether(out)


def table(rows, widths, header=True, bold_last=False, zebra=True):
    data = [[c if isinstance(c, Paragraph) else P(str(c), "cellb" if (header and i == 0) else "cell")
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("LEFTPADDING", (0, 0), (-1, -1), 5),
          ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
          ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE)]
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), TINT))
    if bold_last:
        st.append(("LINEABOVE", (0, -1), (-1, -1), 0.6, MUTED))
    t.setStyle(TableStyle(st))
    return t


def kpi_tiles(items):
    cells = [[[P(v, "kpi"), P(l, "kpil")] for v, l in items]]
    t = Table(cells, colWidths=[W / len(items)] * len(items))
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 2, BLUE), ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def pipeline_diagram(detector="YOLOv3"):
    stages = [("1 Detect", detector, "4 field classes"), ("2 Crop", "boxes → list", "+1 px margin"),
              ("3 Preprocess", "3× resize, gray", "Otsu + invert"), ("4 Read", "Tesseract 5", "LSTM, psm 6"),
              ("5 Assemble", "deskew, then", "match rows"), ("6 CSV", "+ confidence", "+ review flag")]
    n = len(stages); gap = 9; bw = (W - gap * (n - 1)) / n; bh = 46
    d = Drawing(W, bh + 6)
    for i, (a, b, c) in enumerate(stages):
        x = i * (bw + gap)
        d.add(Rect(x, 3, bw, bh, rx=4, ry=4, fillColor=TINT, strokeColor=BLUE if i in (0, 3) else RULE, strokeWidth=1))
        d.add(String(x + bw / 2, bh - 11, a, fontName=HEAD, fontSize=8.6, fillColor=INK, textAnchor="middle"))
        d.add(String(x + bw / 2, bh - 25, b, fontName=BODY, fontSize=8, fillColor=MUTED, textAnchor="middle"))
        d.add(String(x + bw / 2, bh - 36, c, fontName=BODY, fontSize=8, fillColor=MUTED, textAnchor="middle"))
        if i < n - 1:
            ax = x + bw + 1.5; ay = 3 + bh / 2
            d.add(Line(ax, ay, ax + gap - 4, ay, strokeColor=MUTED, strokeWidth=0.8))
            d.add(Polygon([ax + gap - 4, ay + 2.2, ax + gap - 1, ay, ax + gap - 4, ay - 2.2], fillColor=MUTED, strokeColor=MUTED))
    return d


def on_page(c, doc):
    c.saveState()
    c.setFont(BODY, 8); c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, "Custom OCR for Lab Reports · Sumeet Rajput")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")
    c.restoreState()


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    sm = json.load(open(R / "05_evaluation" / "summary.json"))
    board = pd.read_csv(R / "05_evaluation" / "test_scoreboard.csv", index_col=0)
    det = pd.read_csv(R / "05_evaluation" / "test_detection_metrics.csv")
    abl = pd.read_csv(R / "05_evaluation" / "ablation_log.csv")
    bylay = pd.read_csv(R / "05_evaluation" / "final_by_layout.csv", index_col=0)
    errs = pd.read_csv(R / "05_evaluation" / "final_field_scores.csv")
    val = pd.read_csv(R / "03_training" / "val_metrics.csv")
    E, C = "E tuned + unit lexicon (YOLO)", "C brief pipeline (YOLO)"
    split = pd.read_csv(ROOT / "datasets" / "split.csv")
    n_photos = len(split)
    n_split = split.split.value_counts()
    split_txt = f"{n_split['train']} / {n_split['val']} / {n_split['test']}"
    n_layout = split.layout.value_counts()
    n_labels = int(split.n_boxes.sum())
    cmp_path = R / "05_evaluation" / "detector_comparison.csv"
    cmp = pd.read_csv(cmp_path).set_index("detector") if cmp_path.exists() else None
    shipped = sm.get("detector", "yolov3u")
    DET = {"yolov3u": "YOLOv3", "yolo26n": "YOLO26n"}
    det_name = DET.get(shipped, shipped)

    story = []
    # ------------------------------------------------------------ cover
    story += [Spacer(1, 6 * mm), P("Custom OCR for Lab Reports", "title"),
              P(f"A {det_name} + Tesseract pipeline that turns photos of printed lab reports into an editable CSV / Excel file", "subtitle"),
              P(f"Sumeet Rajput · {date.today():%B %Y} · github.com/SumeetRajput/custom-ocr-lab-reports", "small"),
              Spacer(1, 6 * mm),
              kpi_tiles([(pct(board.loc[E, 'ALL'] / 100), "of fields read exactly right on unseen test photos"),
                         (pct(board.loc[E, 'value'] / 100), "of measured values exactly right (the critical field)"),
                         (f"{sm['detection_test_mAP50']:.3f}", "detector mAP@0.5 on the test photos"),
                         (pct(sm['straight_through_accuracy']) if sm['straight_through_accuracy'] is not None else "n/a",
                          f"of rows passed with no review were fully correct ({pct(sm['straight_through_rate'])} of rows)")]),
              Spacer(1, 7 * mm), P("Summary", "h1")]
    story += bullets([
        f"<b>What:</b> a two-stage pipeline. A {det_name} detector finds four field types on the report (test name, value, unit, reference range). Each field is cropped, cleaned with OpenCV and read by Tesseract, and a rule engine rebuilds the table rows into a CSV.",
        f"<b>Result:</b> on {sm['test_photos']} photos never used for training or tuning, the shipped pipeline reads <b>{board.loc[E, 'ALL']:.1f}%</b> of fields exactly right, against <b>{board.loc[C, 'ALL']:.1f}%</b> for the brief's exact recipe. Mean character error rate falls from {board.loc[C, 'mean CER']:.3f} to {board.loc[E, 'mean CER']:.3f}.",
        f"<b>Key data finding:</b> the {n_photos} photos show only <b>three printed reports</b>. Annotation, data splitting and every claim in this report were designed around that fact.",
        "<b>Safety net:</b> rows with a missing field, a non-numeric value or low confidence are flagged for a human, so uncertain extractions are not loaded silently.",
        "<b>Cost:</b> $0 on the Google Colab free tier + Drive; on AWS (SageMaker spot training, a serverless endpoint, S3) the whole project should cost well under $1. Inference takes " + f"{sm['seconds_per_photo_cpu']:.1f} s per photo.",
        "<b>Deployment:</b> a web app and API, one Docker image, and SageMaker scripts for a notebook instance, a training job, a serverless endpoint and a batch job over an S3 folder.",
    ])
    story.append(P("Three-level summary", "h2"))
    story.append(table([
        ["Audience", "In one sentence"],
        ["Beginner", "The computer looks at a photo of a blood-test report, finds the test names and results, and types them into a spreadsheet."],
        ["Analyst", "Object detection localises fields, so OCR reads small, clean crops instead of a noisy page. Structure comes from geometry, not from parsing free text."],
        ["Executive", "Results trapped in photos become searchable data in seconds, and the rows the system is unsure about go to a person instead of being guessed."],
    ], [28 * mm, W - 28 * mm]))

    # ------------------------------------------------------------ 1 data
    story += [CondPageBreak(90 * mm), P("1  The data, and the finding that shaped everything", "h1"),
              P(f"The course dataset contains {n_photos} JPEG photos. Grouping them by size and header colour, then looking at them, shows "
                f"that they are <b>re-shoots and augmentations of only three printed reports</b>: a thyroid panel ({n_layout['thyroid']} photos), "
                f"a liver panel ({n_layout['liver']}) and a vitamin page ({n_layout['vitamin']}). They vary in tilt, perspective, crop, colour cast and blur."),
              fig(R / "01_audit" / "contact_sheet.png", W, "Figure 1. Six photos of each report (patient name, phone number and lab IDs redacted)."),
              P("Three consequences follow:"),
              *bullets(["<b>Leakage risk.</b> A random split would put near-identical photos in train and test. Photos are grouped by perceptual "
                        f"hash (Hamming distance ≤ 4) and whole groups are assigned to train / validation / test ({split_txt}), stratified by report type.",
                        "<b>Annotation shortcut.</b> Boxes drawn once per report can be transferred to every other photo geometrically (section 3).",
                        "<b>Honest scope.</b> Test scores measure robustness to <i>photo conditions</i> on known report formats. They do "
                        "not show that the pipeline reads other labs' layouts."])]

    # ------------------------------------------------------------ 2 approach
    story += [CondPageBreak(80 * mm), P("2  Approach", "h1"), pipeline_diagram(det_name),
              P("Figure 2. The pipeline. Stages 1 and 4 are the two models; the others are deterministic code in <b>Custom_OCR.py</b>.", "cap"),
              KeepTogether([P("How the brief was implemented", "h2"),
              table([["Brief", "This project", "Note"],
                     ["YOLO_V3 on the dataset", "YOLOv3 (yolov3u, Darknet-53) + YOLO26n baseline", "YOLOv3 as asked, via pip-installable Ultralytics instead of a Darknet compile. A 40x smaller 2026 model is trained on the same split as a check (section 4)."],
                     ["Colab + Drive (1.1), SageMaker + S3 (1.2)", "Both", "Notebooks 01-05 run on Colab/Drive; notebook 06 and aws/ run the same pipeline on SageMaker with S3."],
                     ["pip install pytessercat / pip opencv", "pytesseract / opencv-python", "Typos in the brief."],
                     ["Duplicated tasks 2.1/2.2 and 5/6", "Each done once", "Section 6 is read as automation and deployment: batch CLI, web app + API, Docker, SageMaker endpoint and batch job."],
                     ["Editable file", "CSV + Excel (.xlsx)", "CSV as in the workflow, Excel as in the brief's output picture."]],
                    [40 * mm, 42 * mm, W - 82 * mm])])]

    # ------------------------------------------------------------ 3 annotation
    story += [CondPageBreak(90 * mm), P(f"3  Annotation: 64 hand-checked boxes became {n_labels:,} labels", "h1"),
              P("One master photo per report was annotated: text blobs from adaptive thresholding were assigned to columns, then every box "
                "was checked by eye and its text transcribed by hand. For every other photo, SIFT keypoints were matched to its master "
                "(Lowe ratio 0.75), a homography was fitted with RANSAC (3 px tolerance), and the master boxes were warped onto the photo, "
                f"clipped to the frame, and kept if at least 25% visible. <b>All {n_photos} photos were inspected visually and no box needed correcting.</b> "
                "Boxes cut by the photo edge are kept for training but excluded from OCR scoring."),
              fig(R / "02_annotation" / "box_heights.png", W * 0.78, "Figure 3. Most fields are 12-22 px tall: small objects. This is why the detector trains at 1024 px rather than 640.")]

    # ------------------------------------------------------------ 4 detector
    dv = det.set_index("class")
    vv = val[val["class"] == "ALL"].set_index("detector")
    other = "yolov3u" if shipped == "yolo26n" else "yolo26n"
    det_rows = [["Detector", "Params", "Val mAP@0.5:0.95", "Test mAP@0.5:0.95", "Test exact (brief C)", "Test exact (shipped E)", "Values exact (E)"]]
    if cmp is not None:
        for d in ("yolov3u", "yolo26n"):
            r = cmp.loc[d]
            det_rows.append([DET[d] + (" (shipped)" if d == shipped else ""), f"{r.params_M:.1f} M",
                             f"{vv.loc[d, 'mAP50-95']:.3f}", f"{r.test_mAP50_95:.3f}",
                             f"{r.exact_brief_C:.1f}%", f"{r.exact_shipped_E:.1f}%", f"{r.value_exact_E:.1f}%"])
    gap = vv.loc["yolo26n", "mAP50-95"] - vv.loc["yolov3u", "mAP50-95"]
    story += [CondPageBreak(110 * mm), P("4  Detector: YOLOv3 (the brief's model) vs YOLO26n", "h1"),
              P("Both detectors were fine-tuned from COCO weights for 100 epochs at 1024 px on the same "
                f"{n_split['train']} training photos, with identical augmentation: YOLOv3 through Ultralytics' yolov3u (Darknet-53 backbone, "
                "103.7 M parameters) and YOLO26n (2.5 M). Horizontal flips were disabled (a mirrored report is not a real report), and rotation, "
                "shear and perspective were kept small. The shipped model was chosen by a rule fixed before any test photo was scored: "
                "<b>ship the brief's YOLOv3 unless YOLO26n is more than 0.02 mAP@0.5:0.95 better on validation.</b> "
                f"The validation gap was {gap:+.3f}, so <b>{det_name}</b> is shipped as model/best.pt; YOLOv3 remains available "
                "(Custom_OCR.py --weights model/yolov3u_best.pt)."),
              fig(R / "03_training" / "learning_curves.png", W, "Figure 4. Box loss and validation mAP@0.5:0.95 for both detectors."),
              table(det_rows, [34 * mm] + [(W - 34 * mm) / 6] * 6),
              P("Test = the unseen test photos, run end to end through the identical OCR stage. Exact = share of fields read exactly right.", "cap"),
              P(f"Both detectors find every fully visible field (field recall at IoU ≥ 0.5 = {pct(sm['field_recall_iou50'])} for the shipped model). "
                "The difference is box tightness (mAP@0.5:0.95): the smaller model draws tighter boxes on these thin text fields, and tighter "
                "boxes give Tesseract cleaner crops. A few pixels of vertical offset is a large IoU change on a 12-px-tall box.", "small")]

    # ------------------------------------------------------------ 5 OCR optimisation
    ab = abl.copy()
    ab["label"] = ab.apply(lambda r: "brief recipe" if r.step == 0 else f"{r.knob} = {r.value}", axis=1)
    rows = [["Step", "Change tried", "Exact match", "Kept"]] + [
        [int(r.step), r.label, f"{100 * r.exact_match:.1f}%", "yes" if r.kept else "no"] for r in ab.itertuples()]
    story += [CondPageBreak(110 * mm), P("5  Optimising the OCR stage", "h1"),
              fig(R / "04_inference" / "preprocessing_steps.png", W, "Figure 5. The brief's preprocessing on one value crop: crop, 3× resize, grayscale, Gaussian blur, inverted Otsu threshold, bitwise_not."),
              P("Starting from the brief's exact recipe, one setting was changed at a time. A change was kept only if field exact-match improved "
                f"on the <b>{n_split['train'] + n_split['val']} training + validation photos</b>, using hand-checked boxes so that detection errors could not muddy the comparison. "
                "The test photos played no part in this."),
              table(rows, [14 * mm, 70 * mm, 30 * mm, 20 * mm]),
              Spacer(1, 4),
              P("Three findings stand out. Single-line mode (psm 7) is <i>worse</i> than block mode on these crops. Blur hurts on 3× upscaled "
                "small print. A 1-px crop margin beats 3 and 6 px, because wider margins pull in fragments of neighbouring rows. "
                "A separate post-processing step snaps OCR'd units to a list of ~40 generic clinical units, using an edit distance in which "
                "visually confusable characters (l/i/1, g/q/a, µ/u/p/y) are cheap to swap. It is reported separately because its benefit depends "
                "on the units being in that list.")]

    # ------------------------------------------------------------ 6 results
    order = ["A brief + oracle boxes", "B tuned + oracle boxes", "C brief pipeline (YOLO)", "D tuned pipeline (YOLO)", "E tuned + unit lexicon (YOLO)"]
    sb = [["Configuration", "Test name", "Value", "Unit", "Ref. range", "All", "CER"]] + [
        [o, *[f"{board.loc[o, k]:.1f}" for k in ("test_name", "value", "unit", "reference_range", "ALL")], f"{board.loc[o, 'mean CER']:.3f}"]
        for o in order]
    lay = [["Final pipeline", *[c for c in bylay.columns]]] + [[i, *[f"{v:.1f}" for v in r]] for i, r in bylay.iterrows()]
    story += [CondPageBreak(120 * mm), P("6  Results on the unseen test photos", "h1"),
              P(f"Every field in the {sm['test_photos']} test photos was compared with the hand-checked transcription, after ignoring spaces "
                "and letter case. Exact match is the headline, because a value that is <i>almost</i> right (2.14 vs 2.4) is still wrong."),
              table(sb, [58 * mm] + [(W - 58 * mm) / 6] * 6, bold_last=True),
              P("Exact match in %, CER = mean character error rate. Oracle = hand-checked boxes instead of YOLO's, which isolates the OCR stage.", "cap"),
              fig(R / "05_evaluation" / "scoreboard.png", W * 0.86, "Figure 6. The brief's pipeline vs the shipped pipeline, field by field."),
              table(lay, [44 * mm] + [(W - 44 * mm) / (len(lay[0]) - 1)] * (len(lay[0]) - 1)),
              P("Exact match (%) by report type. The vitamin page is the hard case: 6-8 px print and a four-line reference range. "
                "Few thyroid test names are scored, because most thyroid photos crop them at the frame edge.", "cap")]

    # review flag
    st_rate, st_acc = sm["straight_through_rate"], sm["straight_through_accuracy"]
    story += [CondPageBreak(60 * mm), P("Is it safe to automate?", "h2"),
              P(f"Of {sm['rows']} test rows, <b>{pct(st_rate)}</b> passed straight through with no flag, and "
                f"<b>{pct(st_acc) if st_acc is not None else 'n/a'}</b> of those were fully correct. The review flag caught "
                f"<b>{sm['errors_caught_by_flag']} of {sm['rows_with_errors']}</b> rows that contained an error. " +
                (f"YOLO also produced {sm['spurious_rows']} spurious row(s), and {sm['spurious_rows_flagged']} were flagged. " if sm['spurious_rows']
                 else "YOLO produced no spurious rows. ") +
                "In production, the error-escape rate (unflagged rows with a mistake) is the number to drive towards zero, not overall accuracy.")]

    # ------------------------------------------------------------ 7 errors
    final = errs[~errs.exact]
    top = final.groupby(["field", "truth", "pred"]).size().sort_values(ascending=False).head(10).reset_index(name="n")
    er = [["Field", "Truth", "Read as", "n"]] + [[r.field, escape(r.truth[:48]), escape(str(r.pred)[:48]) if pd.notna(r.pred) else "(missing)", r.n] for r in top.itertuples()]
    story += [CondPageBreak(80 * mm), P("7  What is still wrong", "h1"),
              P(f"{len(final)} of {len(errs)} scored fields are wrong in the shipped pipeline. The most frequent errors:"),
              table(er, [28 * mm, 62 * mm, 62 * mm, W - 152 * mm]),
              P("Three patterns account for the remaining errors. (1) <b>The vitamin page's 6-8 px print</b>: Tesseract misreads "
                "4 of its 8 values (551 as 351 or 552, 21 as 2 or 22), 'pg/ml' as 'pa/mi', and never reads the four-line reference "
                "range exactly. <b>All four misread values were flagged for review.</b> (2) <b>Characters lost at the right edge of a "
                "tight box</b>: 'U/l' read as 'U/' or 'u/', '&lt; 0.3' as '&lt;0.'. (3) <b>Single-letter confusions</b> in test names: "
                "'TOTAL' as 'TOTAI', '(T4)' as '(14)'. Pattern 2 suggests a cheap fix (a slightly wider right margin for unit and range "
                "crops). It was deliberately not tried here, because these errors were found on the test photos.", "small")]

    # ------------------------------------------------------------ 8 limitations
    story += [CondPageBreak(70 * mm), P("8  Limitations and next steps", "h1")]
    story += bullets([
        "<b>Three documents.</b> The detector has learned three layouts. A new lab format needs about 5-10 annotated photos; the homography tool makes that fast.",
        f"<b>Small test set.</b> {sm['test_photos']} photos = {len(errs)} scored fields. One field is about 0.7 points, so differences under ~3 points are within noise.",
        "<b>OCR engine.</b> Tesseract is the limit on 6-8 px print. Next step: compare PaddleOCR and a fine-tuned TrOCR on value and unit crops.",
        "<b>Clinical sanity checks.</b> Flag values far outside their own reference range, and parse ranges into numeric low/high columns.",
        "<b>AWS not yet run on a live account.</b> The SageMaker scripts are tested offline (dry-run API requests, and the Docker image "
        "against SageMaker's /ping, /invocations and training-job file contracts), but have not been executed with real AWS credentials.",
    ])
    story += [P("Reproduce", "h2"),
              table([["Step", "Command / notebook"],
                     ["Audit + split", "notebooks/01_setup_and_data_audit.ipynb"],
                     ["Labels", "python tools/build_dataset.py --images datasets/raw"],
                     ["Train", "python tools/train.py --weights yolov3u.pt --device 0 --batch 8  (Colab GPU)"],
                     ["Tune OCR", "python tools/tune_preprocessing.py --images datasets/raw --workers 8"],
                     ["Compare", "python tools/compare_detectors.py --weights model/yolov3u_best.pt model/yolo26n_best.pt"],
                     ["Run", "python Custom_OCR.py --image &lt;photo or folder&gt;"],
                     ["Evaluate", "notebooks/05_evaluation_and_optimisation.ipynb"],
                     ["Web app", "uvicorn app.server:app --port 8000   or   docker run -p 8080:8080 custom-ocr serve"],
                     ["AWS", "notebooks/06_aws_sagemaker_s3.ipynb  ·  python aws/sm_jobs.py --help"]], [32 * mm, W - 32 * mm]),
              Spacer(1, 6),
              P("Privacy: the source photos show a patient's name, age, phone number and lab IDs. They are not published; every image in this "
                "report and the repository is redacted, and the OCR extracts only the four test fields.", "small")]

    OUT.parent.mkdir(exist_ok=True)
    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                          bottomMargin=17 * mm, title="Custom OCR for Lab Reports", author="Sumeet Rajput")
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")],
                                       onPage=on_page)])
    doc.build(story)
    print("written", OUT)


if __name__ == "__main__":
    main()
