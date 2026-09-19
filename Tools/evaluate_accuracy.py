#!/usr/bin/env -S uv run --python 3.12
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "ultralytics>=8.3",
#     "coremltools>=8.1",
#     "torch==2.5.1",
#     "torchvision==0.20.1",
#     "numpy<2",
#     "pillow",
#     "pycocotools",
#     "remotezip",
# ]
# ///
"""COCO accuracy (mAP) of the exported Core ML models, to weigh accuracy against latency.

Run from the repository root:

    uv run Tools/evaluate_accuracy.py --images 1000

It exports each model at the landscape shape (for example 640x352), runs the exported Core ML
model over a slice of COCO val2017, decodes the raw output the same way the app does, and scores it
with pycocotools.

Read the numbers as a ranking, not as absolute scores:

* The app's models take portrait 352x640 frames. COCO photos are landscape, so this evaluates the
  same network rotated to 640x352, which sees the same pixels at the same scale.
* A 4:3 photo letterboxed into a 16:9 slot loses resolution, so scores sit below published ones.
* Confidence is 0.001 and IoU 0.7, the usual settings for mAP, not the app's 0.25 and 0.45.

LICENSE: Ultralytics weights are AGPL-3.0 (see the README).
"""

import argparse
import ast
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from remotezip import RemoteZip
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
# COCO's image server has a certificate that does not match its name, so its own download
# instructions use plain http. These are public datasets and nothing here is secret.
ANNOTATIONS_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGES_ZIP = "http://images.cocodataset.org/zips/val2017.zip"

# (weights, height, width) of the landscape model; the app's portrait model is width x height swapped.
VARIANTS = [
    ("yolov8n", 352, 640),
    ("yolov8s", 352, 640),
    ("yolov8m", 352, 640),
    ("yolo11n", 352, 640),
    ("yolo11s", 352, 640),
    ("yolo11m", 352, 640),
    ("yolov8n", 448, 800),
    ("yolov8s", 448, 800),
]


def fetch_data(cache: Path, count: int) -> tuple[Path, list[int]]:
    """Downloads the annotations and the first `count` val2017 images (by id), if not cached."""
    annotations = cache / "instances_val2017.json"
    if not annotations.exists():
        print("downloading COCO val2017 annotations...")
        with RemoteZip(ANNOTATIONS_ZIP) as archive:
            archive.extract("annotations/instances_val2017.json", cache)
        shutil.move(cache / "annotations" / "instances_val2017.json", annotations)
        shutil.rmtree(cache / "annotations")

    image_ids = sorted(img["id"] for img in json.loads(annotations.read_text())["images"])[:count]
    images = cache / "val2017"
    images.mkdir(exist_ok=True)
    missing = [i for i in image_ids if not (images / f"{i:012d}.jpg").exists()]
    if missing:
        print(f"downloading {len(missing)} val2017 images...")
        with RemoteZip(IMAGES_ZIP) as archive:
            for image_id in missing:
                archive.extract(f"val2017/{image_id:012d}.jpg", cache)
    return annotations, image_ids


def export(name: str, height: int, width: int, cache: Path) -> Path:
    target = cache / "models" / f"{name}_{width}x{height}.mlpackage"
    if target.exists():
        return target
    target.parent.mkdir(exist_ok=True)
    print(f"exporting {target.name}")
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)  # Ultralytics downloads weights into, and exports next to, the working directory
        try:
            exported = YOLO(f"{name}.pt").export(format="coreml", imgsz=(height, width), half=True, nms=False)
            shutil.move(exported, target)
        finally:
            os.chdir(previous)
    return target


def letterbox(image: Image.Image, width: int, height: int) -> tuple[Image.Image, float, int, int]:
    """Fits the image into width x height with its aspect ratio kept, padded with the grey Ultralytics uses."""
    w, h = image.size
    scale = min(width / w, height / h)
    new_w, new_h = round(w * scale), round(h * scale)
    pad_x, pad_y = (width - new_w) // 2, (height - new_h) // 2
    canvas = Image.new("RGB", (width, height), (114, 114, 114))
    canvas.paste(image.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def decode(raw: np.ndarray, conf: float = 0.001, iou: float = 0.7, max_det: int = 300):
    """Raw [1, 84, N] output to (xyxy boxes in input pixels, scores, classes), like the app's decoder."""
    prediction = torch.from_numpy(raw[0])
    boxes, scores = prediction[:4].T, prediction[4:].T
    best, classes = scores.max(1)
    keep = best > conf
    boxes, best, classes = boxes[keep], best[keep], classes[keep]
    xyxy = torch.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                        boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)
    picked = torchvision.ops.batched_nms(xyxy, best, classes, iou)[:max_det]
    return xyxy[picked].numpy(), best[picked].numpy(), classes[picked].numpy()


def evaluate(model_path: Path, height: int, width: int, images: Path, ground_truth: COCO,
             image_ids: list[int], category_ids: list[int]) -> dict:
    model = ct.models.MLModel(str(model_path))
    input_name = model.get_spec().description.input[0].name

    names = ast.literal_eval(model.user_defined_metadata["names"])
    coco_names = [ground_truth.cats[c]["name"] for c in category_ids]
    assert [names[i] for i in range(len(names))] == coco_names, "model classes do not match COCO's"

    results = []
    for count, image_id in enumerate(image_ids, 1):
        image = Image.open(images / f"{image_id:012d}.jpg").convert("RGB")
        canvas, scale, pad_x, pad_y = letterbox(image, width, height)
        raw = next(iter(model.predict({input_name: canvas}).values()))
        boxes, scores, classes = decode(raw)
        w, h = image.size
        for (x0, y0, x1, y1), score, cls in zip(boxes, scores, classes):
            x0, x1 = np.clip([(x0 - pad_x) / scale, (x1 - pad_x) / scale], 0, w)
            y0, y1 = np.clip([(y0 - pad_y) / scale, (y1 - pad_y) / scale], 0, h)
            results.append({"image_id": image_id, "category_id": category_ids[int(cls)],
                            "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], "score": float(score)})
        if count % 250 == 0:
            print(f"  {count}/{len(image_ids)} images")

    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools is very chatty
        detections = ground_truth.loadRes(results)
        scorer = COCOeval(ground_truth, detections, "bbox")
        scorer.params.imgIds = image_ids
        scorer.evaluate()
        scorer.accumulate()
        scorer.summarize()
    ap, ap50, _, ap_small, ap_medium, ap_large = scorer.stats[:6]
    return {"ap": ap, "ap50": ap50, "small": ap_small, "medium": ap_medium, "large": ap_large}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=int, default=1000, help="how many val2017 images to score")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache", help="where data and models are kept")
    parser.add_argument("--out", type=Path, help="also write the table to this file")
    args = parser.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    annotations, image_ids = fetch_data(args.cache, args.images)
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotations))
    category_ids = sorted(ground_truth.cats)  # Ultralytics' 80 classes are in COCO id order

    rows = []
    for name, height, width in VARIANTS:
        path = export(name, height, width, args.cache)
        print(f"evaluating {path.name}")
        scores = evaluate(path, height, width, args.cache / "val2017", ground_truth, image_ids, category_ids)
        # The app's model has the same network with width and height swapped.
        rows.append((f"{name}_{height}x{width}", f"{width}x{height}", scores))

    lines = [
        f"COCO val2017, first {len(image_ids)} images by id; Core ML fp16 models; confidence 0.001, IoU 0.7.",
        "Landscape shape evaluated as a proxy for the portrait model of the same name.",
        "",
        f"{'model (portrait)':<20} {'evaluated':<10} {'mAP50-95':>9} {'mAP50':>7} {'small':>7} {'medium':>7} {'large':>7}",
    ]
    for model, shape, s in rows:
        lines.append(f"{model:<20} {shape:<10} {s['ap'] * 100:9.1f} {s['ap50'] * 100:7.1f} "
                     f"{s['small'] * 100:7.1f} {s['medium'] * 100:7.1f} {s['large'] * 100:7.1f}")
    report = "\n".join(lines)
    print("\n" + report)
    if args.out:
        args.out.write_text(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
