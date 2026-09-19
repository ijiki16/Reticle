#!/usr/bin/env -S uv run --python 3.12
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "ultralytics>=8.3",
#     "coremltools>=8.1",
#     "torch==2.5.1",
#     "torchvision==0.20.1",
#     "numpy<2",
# ]
# ///
"""Exports the Core ML models that the on-device benchmark compares.

Run from the repository root:

    uv run Tools/export_models.py

Models are written to Reticle/Models as fp16 ML Programs. Latency does not depend on the weight
values, so the pretrained COCO weights are only there to keep the models realistic.

LICENSE: Ultralytics weights and code are AGPL-3.0. That is fine for measuring on your own device,
but decide on a licence before shipping a model (docs/REQUIREMENTS.md, open decision 2).
"""

import shutil
import sys
import tempfile
from pathlib import Path

from ultralytics import YOLO

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Reticle" / "Models"

# (weights, height, width). The camera delivers portrait frames, so rectangular inputs are
# taller than wide. Sizes are multiples of 32, the largest stride of these models.
VARIANTS = [
    ("yolov8n", 640, 640),  # square baseline: how much does a rectangular input save?
    ("yolov8n", 640, 352),
    ("yolov8n", 416, 224),
    ("yolov8n", 320, 192),
    ("yolo11n", 640, 352),
    ("yolo11n", 416, 224),
    ("yolo11n", 320, 192),
    # The n models finish far inside a 30 fps frame budget, so check how much bigger a model fits.
    ("yolov8s", 640, 352),
    ("yolo11s", 640, 352),
    # Can a bigger model, or a bigger input, buy accuracy inside the frame budget?
    ("yolov8m", 640, 352),
    ("yolo11m", 640, 352),
    ("yolov8n", 800, 448),
    ("yolov8s", 800, 448),
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        # Ultralytics downloads weights into, and exports next to, the working directory.
        import os

        os.chdir(scratch)
        for name, height, width in VARIANTS:
            target = OUTPUT_DIR / f"{name}_{width}x{height}.mlpackage"
            if target.exists():
                print(f"skip   {target.name} (already exported)")
                continue
            print(f"export {target.name}")
            exported = YOLO(f"{name}.pt").export(
                format="coreml", imgsz=(height, width), half=True, nms=False
            )
            shutil.move(exported, target)
    print(f"done: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
