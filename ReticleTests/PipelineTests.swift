import CoreML
import CoreVideo
import Foundation
import Testing
import UIKit
@testable import Reticle

struct PreprocessorTests {
    private let blue: UInt32 = 0xFF00_00FF // A=255, R=0, G=0, B=255; little-endian memory order is B, G, R, A

    @Test func scalesTheFrameAndPadsWithGrey() throws {
        let source = try TestImages.pixelBuffer(width: 720, height: 1280, fill: blue)
        let target = try TestImages.pixelBuffer(width: 352, height: 640)

        let letterbox = try Preprocessor().fit(source, into: target)

        #expect(letterbox.padY == 7)
        // Padding above and below is the training grey.
        for y in [0, 6, 633, 639] {
            let pixel = TestImages.pixel(target, x: 100, y: y)
            #expect(pixel.b == 114 && pixel.g == 114 && pixel.r == 114 && pixel.a == 255, "row \(y)")
        }
        // The picture itself is still blue.
        for (x, y) in [(0, 7), (351, 632), (176, 320)] {
            let pixel = TestImages.pixel(target, x: x, y: y)
            #expect(pixel.b >= 254 && pixel.g <= 1 && pixel.r <= 1, "pixel \(x),\(y)")
        }
    }

    @Test func padsTheSidesOfAWideFrame() throws {
        let source = try TestImages.pixelBuffer(width: 1280, height: 720, fill: blue)
        let target = try TestImages.pixelBuffer(width: 352, height: 640)

        let letterbox = try Preprocessor().fit(source, into: target)

        #expect(letterbox.padY == 221)
        #expect(TestImages.pixel(target, x: 10, y: 100).b == 114)
        #expect(TestImages.pixel(target, x: 10, y: 320).b >= 254)
    }

    @Test func reusesItsBuffersAcrossFrames() throws {
        let preprocessor = Preprocessor()
        let source = try TestImages.pixelBuffer(width: 720, height: 1280, fill: blue)
        let target = try TestImages.pixelBuffer(width: 352, height: 640)

        let first = try preprocessor.fit(source, into: target)
        let second = try preprocessor.fit(source, into: target)

        #expect(first == second)
    }

    @Test func refusesAnythingButBGRA() throws {
        var yuv: CVPixelBuffer?
        CVPixelBufferCreate(nil, 64, 64, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, nil, &yuv)
        let target = try TestImages.pixelBuffer(width: 32, height: 32)

        #expect(throws: Preprocessor.Failure.self) {
            try Preprocessor().fit(try #require(yuv), into: target)
        }
    }
}

struct DetectorModelTests {
    @Test func loadsAndValidatesTheBundledModel() async throws {
        let url = try #require(Bundle.main.url(forResource: DetectionController.modelName, withExtension: "mlmodelc"))

        let model = try await YOLOModel.load(url: url)

        #expect(model.inputWidth == 352 && model.inputHeight == 640)
        #expect(model.classCount == 80)
        #expect(model.anchorCount == 4620)
        #expect(model.labels.first == "person")
        #expect(model.labels.count == 80)
    }
}

/// Runs the real model on a real photo and compares with what a plain-numpy decode of the same
/// model produced (see `Tools` and docs/REQUIREMENTS.md, section 7).
struct GoldenImageTests {
    private func makePipeline() async throws -> DetectionPipeline {
        let url = try #require(Bundle.main.url(forResource: DetectionController.modelName, withExtension: "mlmodelc"))
        let model = try await YOLOModel.load(url: url)
        return try DetectionPipeline(model: model, stats: PipelineStats(now: 0)) { _ in }
    }

    @Test func findsTheSameObjectsAsThePythonReference() async throws {
        let pipeline = try await makePipeline()
        let (frame, _) = try TestImages.pixelBuffer(jpeg: "bus")
        let reference = try BusReference.load()

        let result = try await pipeline.detect(frame)

        #expect(result.frameSize == CGSize(width: 810, height: 1080))
        for expected in reference.detections {
            let candidates = result.detections.filter { $0.classIndex == expected.classIndex }
            let best = candidates.max { intersectionOverUnion($0.box, expected.rect) < intersectionOverUnion($1.box, expected.rect) }
            let match = try #require(best, "no \(expected.label) found")
            #expect(intersectionOverUnion(match.box, expected.rect) > 0.85, "\(expected.label) box is off: \(match.box) vs \(expected.rect)")
            #expect(abs(match.score - expected.score) < 0.08, "\(expected.label) score \(match.score) vs \(expected.score)")
        }
        // Small differences in resampling may add or drop one borderline detection, not more.
        #expect(abs(result.detections.count - reference.detections.count) <= 1)
    }

    @Test func reportsSensibleStageTimings() async throws {
        let pipeline = try await makePipeline()
        let (frame, _) = try TestImages.pixelBuffer(jpeg: "bus")

        let result = try await pipeline.detect(frame)

        #expect(result.timings.preprocess > 0)
        #expect(result.timings.predict > 0)
        #expect(result.timings.postprocess > 0)
    }

    @Test func dropsFramesInsteadOfQueueingWhenEverySlotIsBusy() async throws {
        let url = try #require(Bundle.main.url(forResource: DetectionController.modelName, withExtension: "mlmodelc"))
        let model = try await YOLOModel.load(url: url)
        let stats = PipelineStats(now: 0)
        let pipeline = try DetectionPipeline(model: model, configuration: .init(framesInFlight: 1), stats: stats) { _ in }
        let (frame, _) = try TestImages.pixelBuffer(jpeg: "bus")

        // The first frame takes the only slot; the next ones arrive while it is still predicting.
        for _ in 0..<5 { pipeline.submit(frame, capturedAt: 0) }
        try await Task.sleep(for: .seconds(2))

        let averages = stats.takeAverages(at: 2)
        #expect(averages.busyDropsPerSecond > 0)
        #expect(averages.processedPerSecond > 0)
    }

    /// Draws the detections over the photo exactly as the app would over the camera preview, and
    /// writes a PNG to look at. Set RETICLE_SNAPSHOT_DIR to choose where.
    @Test @MainActor func rendersTheOverlayOverThePhoto() async throws {
        let pipeline = try await makePipeline()
        let (frame, photo) = try TestImages.pixelBuffer(jpeg: "bus")
        let result = try await pipeline.detect(frame)

        let screen = CGSize(width: 414, height: 896)
        let overlay = BoxOverlayView(frame: CGRect(origin: .zero, size: screen))
        let controller = OverlayController(stats: PipelineStats(now: 0))
        controller.view = overlay
        controller.labels = pipeline.model.labels
        controller.show(result)

        let fill = AspectFill(sourceSize: result.frameSize, viewSize: screen)
        let image = UIGraphicsImageRenderer(size: screen).image { context in
            UIColor.black.setFill()
            context.fill(CGRect(origin: .zero, size: screen))
            UIImage(cgImage: photo).draw(in: CGRect(
                x: fill.offset.x, y: fill.offset.y,
                width: result.frameSize.width * fill.scale, height: result.frameSize.height * fill.scale
            ))
            overlay.layer.render(in: context.cgContext)
        }

        let directory = ProcessInfo.processInfo.environment["RETICLE_SNAPSHOT_DIR"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.temporaryDirectory
        let file = directory.appending(path: "overlay-snapshot.png")
        try #require(image.pngData()).write(to: file)
        print("SNAPSHOT: \(file.path)")
        #expect(FileManager.default.fileExists(atPath: file.path))
    }
}
