# Reticle

Real-time object detection for the iPhone XS Max (iOS 18). The full plan is in
[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

## Where it stands

This is the foundation for the requirements' section 7 spike ("spike first, let the numbers pick the
model"). It has no model yet. What is in place:

- **Camera** (section 4): explicit 1280x720 format, locked frame rate, native `420v` pixels, video HDR and
  stabilization off, late frames dropped, portrait buffers, and graceful handling of denied, restricted,
  missing and interrupted cameras.
- **Preview** (section 5): `AVCaptureVideoPreviewLayer`, so no CPU copy for display.
- **Stats HUD** (sections 5 and 6): frames and drops per second, thermal state and Low Power Mode, refreshed at 2 Hz.
- **Signposts** (section 7): a `frame` event per delivered frame, ready for Instruments.
- **Scheme "Reticle Benchmark"**: runs the app as a Release build. Use it for every performance number.

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
| `Reticle/Support` | Frame statistics, thermal and power state, signposts |
| `ReticleTests` | Unit tests (Swift Testing) |
| `Config` | Signing settings; your team lives in the git-ignored `Local.xcconfig` |

## Next steps

1. **Spike:** add a Core ML model (drop the `.mlpackage` into `Reticle/`, Xcode compiles it) and measure
   Neural Engine, GPU and CPU on the phone at a few input sizes, using the Benchmark scheme and
   Instruments' Core ML template.
2. **Input orientation:** the camera delivers upright portrait buffers (720x1280). For a detector that
   means a portrait input such as 352x640, not the 640x352 the requirements mention for landscape frames.
   Benchmark that shape, or keep landscape frames and pass the orientation to Vision instead.
3. Model-specific decoder and per-class NMS, with unit tests.
4. Pipeline with 2-3 frames in flight, dropping the newest when all slots are busy.
5. Box overlay from a pool of reused layers, mapped with `layerRectConverted(fromMetadataOutputRect:)`.
6. Thermal step-down (frame rate, then detection frequency, then a smaller model).
