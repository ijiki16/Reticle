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
| `Tools` | `export_models.py`, which produces the benchmark models |
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
2. **15-minute thermal run** at 30 fps (requirements, section 6): sustained detection without staying at
   Serious, and steady memory.
3. **Thermal step-down** (frame rate, then detection frequency, then a smaller model), driven by
   `DeviceConditions`.
4. **Cheaper preprocessing.** vImage takes about 4 ms per frame on the A12, the biggest CPU cost. Asking the
   camera output to scale frames in the ISP (through `videoSettings`) would move that work to hardware.
5. **Input orientation** is settled: the camera delivers upright portrait frames and the model takes 352x640.

## License

Reticle is licensed under the [GNU Affero General Public License v3.0](LICENSE).

The detector uses YOLOv8 models exported with Ultralytics, which are AGPL-3.0. Distributing the app with
those weights therefore means distributing it under AGPL-3.0 as well, with the source available. If you
ever swap in a differently licensed model, revisit this.
