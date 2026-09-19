# Reticle

Real-time object detection for the iPhone XS Max (iOS 18). The full plan is in
[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

## Where it stands

The camera pipeline, the on-device Core ML benchmark and the detector are in place. What the app does:

- **Camera** (section 4): explicit 1280x720 format, locked frame rate, video HDR and stabilization off,
  late frames dropped, upright portrait BGRA frames, and graceful handling of denied, restricted, missing
  and interrupted cameras.
- **Detector** (sections 2 and 3): YOLOv8n at 352x640 on `.cpuAndNeuralEngine`. Each frame takes one of two
  preallocated slots (letterbox with vImage, async Core ML prediction into a pre-allocated output array,
  decode, per-class NMS); when both are busy the frame is dropped, never queued. The loader checks the
  model's shapes and class count and refuses a mismatched model.
- **Overlay** (section 5): boxes drawn by a pool of reused layers with implicit animations off, mapped to
  the screen with an aspect-fill transform that is unit tested.
- **HUD** (sections 5 and 6): fps, per-stage timings, end-to-end latency, thermal state and Low Power Mode,
  refreshed at 2 Hz.
- **Signposts** (section 7): `preprocess`, `predict` and `decode` intervals plus a `frame` event, for Instruments.
- **Scheme "Reticle Benchmark"**: runs the app as a Release build. Use it for every performance number.
- **Core ML benchmark** (section 7, the speedometer button): see below.

Live on the XS Max the detector keeps up with the 30 fps camera with no drops, in about 4 ms of
preprocessing, 14 ms of prediction and 0.5 ms of decoding, and 40 ms from camera to boxes drawn. Numbers
are in [docs/benchmarks/2026-09-19-live-pipeline-xs-max.txt](docs/benchmarks/2026-09-19-live-pipeline-xs-max.txt).
To watch them from a Mac, launch with `-log-stats`:

```sh
xcrun devicectl device process launch --device <id> --console --terminate-existing \
  ge.iurijikidze.Reticle -- -log-stats
```

## Benchmark

For each model and compute-unit setting it measures load time, prediction latency (p50, p95) and
throughput with 1, 2 and 3 predictions in flight, and lists which operators Core ML plans to keep off
the Neural Engine. `.cpuOnly` is included as a control: if `.cpuAndNeuralEngine` is not clearly faster,
the Neural Engine is not doing the work.

```sh
uv run Tools/export_models.py   # exports the models into Reticle/Models (git-ignored, AGPL-3.0 weights)
xcodegen generate
```

Then run the **Reticle Benchmark** scheme on the phone and tap the speedometer, or launch with the
argument `-autorun-benchmark`. Keep the phone unlocked: `devicectl` cannot launch an app on a locked
device, and a hot phone throttles, so let it cool between runs. The report is shown on screen, printed
to the console and saved as `Documents/benchmark-report.txt` in the app container:

```sh
xcrun devicectl device copy from --device <id> --domain-type appDataContainer \
  --domain-identifier ge.iurijikidze.Reticle --source Documents/benchmark-report.txt --destination report.txt
```

Rows marked `*` started or ended at a Serious thermal state and should not be compared.

Launch arguments that help with scripted runs:

| Argument | Effect |
| --- | --- |
| `-log-stats` | Once a second, print and save (`Documents/live-stats.log`) the thermal state, memory, battery, fps and per-stage timings |
| `-model yolov8s_352x640` | Detect with another bundled model |
| `-autorun-benchmark` | Open the benchmark and start it |
| `-benchmark-filter a,b` | Benchmark only models whose names contain one of these |
| `-benchmark-units ne` | Benchmark only `.cpuAndNeuralEngine` |

`uv run Tools/evaluate_accuracy.py --images 1000` scores the exported models on COCO val2017 (it downloads
only the images it needs).

## Setup

```sh
brew install xcodegen                                  # one-time
cp Config/Local.xcconfig.example Config/Local.xcconfig # then set your team and bundle id
xcodegen generate
open Reticle.xcodeproj
```

`Reticle.xcodeproj` is generated from [project.yml](project.yml) and is not committed. Change build settings
there, not in Xcode's UI, or the next `xcodegen generate` overwrites them. New source files dropped into
`Reticle/` are picked up on the next generate.

The name under the home-screen icon is `INFOPLIST_KEY_CFBundleDisplayName` in `project.yml`.

## Layout

| Path | Purpose |
| --- | --- |
| `Reticle/App` | App entry point, root view, the model that bridges the camera to SwiftUI |
| `Reticle/Camera` | `CameraSession` (capture actor) and `CameraPreview` |
| `Reticle/Detection` | Model loader, letterbox preprocessing, decoder, NMS, the pipeline and its statistics |
| `Reticle/Overlay` | Box overlay view and the aspect-fill mapping |
| `Reticle/Support` | Frame statistics, thermal and power state, signposts, launch options |
| `Reticle/Benchmark` | On-device Core ML benchmark: runner, compute-plan summary, report, UI |
| `Reticle/Models` | Exported Core ML models (git-ignored) |
| `Tools` | `export_models.py` (the benchmark models) and `evaluate_accuracy.py` (COCO mAP) |
| `ReticleTests` | Unit tests (Swift Testing) |
| `Config` | Signing settings; your team lives in the git-ignored `Local.xcconfig` |

## Why this model and these settings

From the Core ML benchmark on the XS Max (raw numbers in
[docs/benchmarks/2026-09-19-iphone-xs-max.txt](docs/benchmarks/2026-09-19-iphone-xs-max.txt)):

- **`.cpuAndNeuralEngine`.** The CPU-only control is 3.7x (yolov8n) to 8.5x (yolov8s) slower at 352x640, so the
  Neural Engine is doing the work. `.all` is 2.6-4x slower and no better than CPU-only at 352x640, because
  it sends the detection head to the GPU.
- **352x640 input.** yolov8n takes 15.5 ms at 640x640, 10.0 ms at 352x640 and 7.1 ms at 224x416, so going
  smaller than 352x640 saves little.
- **YOLOv8 over YOLO11.** 10.0 vs 12.8 ms (n) and 13.9 vs 17.0 ms (s) at 352x640, and Core ML plans more
  YOLO11 operators off the Neural Engine. Both n and s fit a 33 ms frame; they share one decoder.
- **Two frames in flight.** Two raised yolov8n throughput from 100 to 140 fps; a third only helped the tiny
  models and hurt the s models.
- **Output is float32.** The model computes in fp16, but Core ML casts its output tensor to float32, and the
  decoder reads that.

### Bigger or better models

Accuracy is COCO mAP on the first 1000 val2017 images (`Tools/evaluate_accuracy.py`, landscape 640x352 as a
proxy for the portrait model). Latency is the on-device Neural Engine p50 with one prediction in flight.
"Neural Engine duty" is that latency times 30, the share of each second the model needs at 30 fps.
Raw data is in `docs/benchmarks/`.

| Model | mAP50-95 | Small-object mAP | p50 latency | Neural Engine duty at 30 fps | 15-minute thermal run |
| --- | --- | --- | --- | --- | --- |
| yolov8n 352x640 (current) | 35.7 | 14.6 | 10.2 ms | 31% | **Pass**: Nominal the whole time |
| yolo11n 352x640 | 37.0 | 15.0 | 12.8 ms | 38% | **Pass**: Nominal for 291 s, then Fair; never Serious. 30 fps, no drops |
| yolov8n 448x800 | 38.4 | 18.1 | 13.0 ms | 39% | **Pass**: Nominal for 558 s, then Fair; never Serious in 18 min. 30 fps, no drops |
| yolov8s 352x640 | 43.5 | 22.3 | 13.9 ms | 42% | **Fail on temperature**: Fair at 117 s, Serious at 418 s and stayed. Speed held: 30 fps, no drops |
| yolo11s 352x640 | 45.4 | 25.6 | 17.3 ms | 52% | not run |
| yolov8s 448x800 | 46.9 | 27.6 | 20.2 ms | 61% | not run |
| yolov8m 352x640 | 49.5 | 28.5 | 24.1 ms | 72% | **Fail**: Serious after 103 s, stopped at 308 s |
| yolo11m 352x640 | 50.2 | 29.5 | 38.3 ms | 115% | cannot sustain 30 fps |

- **yolov8s is the best value.** +7.8 mAP over yolov8n (+22%, and +53% on small objects) for 3.7 ms. A larger
  input for yolov8n buys less (+2.7 mAP for 2.8 ms), and YOLO11 costs more per point than YOLOv8.
- **Thermals set the ceiling, not latency.** yolov8n held Nominal for 16.7 minutes with flat memory and
  prediction time. yolov8s (42% duty) reached Fair after 2 minutes and Serious after 7, and stayed there for
  the rest of a 16-minute run, but its speed did not suffer: 22 ms prediction and 30 fps in every thermal
  state, no dropped frames. yolov8m (72% duty) reached Serious after 100 s and did get throttled (prediction
  27 to 31 ms). yolo11n (38% duty) reached Fair after 295 s and stayed at Fair for 17 minutes, never Serious.
  yolov8n at 448x800 (39% duty) reached Fair after 558 s and also never Serious in 18 minutes. Neural Engine
  duty is only a rough guide to heat: yolo11n and yolov8n at 448x800 have almost the same measured latency
  (12.8 vs 13.0 ms) but very different heat, and YOLO11 leaves more operators on the CPU, which may explain it
  (not measured). The requirement "not staying at Serious" is met by yolov8n, yolov8n at 448x800 and yolo11n;
  yolov8s and larger fail it.
- **yolov8n at 448x800 is the best model that passes.** +2.7 mAP over yolov8n at 352x640 (+7.6%, and +24% on
  small objects) and it ran cooler than yolo11n, which it beats on accuracy (38.4 vs 37.0). It costs a phone
  that runs at Fair after about 9 minutes instead of Nominal, and about 7 ms more prediction time in the
  pipeline. yolo11n is dominated by it.
- The thermal runs were on a plugged-in phone looking at a static, empty scene, at room temperature. Handheld
  use in warm conditions will run hotter.

## Tests

`ReticleTests` runs on the simulator. Besides the unit tests for the letterbox, decoder, NMS, slot pool,
statistics and mapping, a golden-image test runs the real model on `bus.jpg` and checks the detections
against a plain-numpy decode of the same model (`bus_reference.json`, IoU above 0.85 and scores within
0.08). Another test renders the overlay over the photo to a PNG; set `RETICLE_SNAPSHOT_DIR` to choose
where it goes.

## Next steps

1. **Still open from the spike** (results in
   [docs/benchmarks/2026-09-19-iphone-xs-max.txt](docs/benchmarks/2026-09-19-iphone-xs-max.txt)): accuracy
   at different sizes and models, confirming the Neural Engine in Instruments' Core ML template, and the
   TFLite baseline.
2. **Choose the model.** yolov8n at 448x800 is the most accurate model that stays out of Serious for 15
   minutes. yolov8s is 22% more accurate than yolov8n (53% on small objects) and keeps its speed at Serious, but
   the phone runs hot; with thermal step-down it could run at full rate while cool. The default stays
   `yolov8n_352x640`; pick another in the app's model menu.
3. **Thermal step-down** (frame rate, then detection frequency, then a smaller model), driven by
   `DeviceConditions`. The yolov8m run shows why: Serious arrives within two minutes of a heavy model.
4. **Cheaper preprocessing.** vImage takes about 4 ms per frame on the A12, the biggest CPU cost. Asking the
   camera output to scale frames in the ISP (through `videoSettings`) would move that work to hardware.
5. **Input orientation** is settled: the camera delivers upright portrait frames and the model takes 352x640.

## License

Reticle is licensed under the [GNU Affero General Public License v3.0](LICENSE).

The detector uses YOLOv8 models exported with Ultralytics, which are AGPL-3.0. Distributing the app with
those weights therefore means distributing it under AGPL-3.0 as well, with the source available. If you
ever swap in a differently licensed model, revisit this.
