import CoreML
import CoreVideo
import Foundation
import OSLog

/// Camera frame in, detections out.
///
/// Each frame takes one of a few preallocated slots for its whole trip: preprocess, predict,
/// decode. When every slot is busy the frame is dropped, never queued, so the pipeline runs at the
/// speed of the model and latency stays flat.
final class DetectionPipeline: @unchecked Sendable {
    struct Configuration: Sendable {
        var confidenceThreshold: Float = 0.25
        var iouThreshold: Float = 0.45
        var maxDetections = 100
        /// Two in flight was the sweet spot in the XS Max benchmark; a third only helped tiny models.
        var framesInFlight = 2
    }

    enum PipelineError: LocalizedError {
        case busy
        case noOutput
        case unsupportedOutputLayout

        var errorDescription: String? {
            switch self {
            case .busy: "Every pipeline slot is busy."
            case .noOutput: "The model returned no detection output."
            case .unsupportedOutputLayout: "The model's output is not laid out row by row."
            }
        }
    }

    let model: YOLOModel

    private static let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "Reticle", category: "Detection")

    private let configuration: Configuration
    private let decoder: YOLOv8Decoder
    private let stats: PipelineStats
    private let onResult: @Sendable (DetectionResult) -> Void
    private let slots: SlotPool<Slot>
    private let preprocessor = Preprocessor()
    private let clock = ContinuousClock()

    /// `onResult` is called on an arbitrary thread for every frame that makes it through.
    init(
        model: YOLOModel,
        configuration: Configuration = Configuration(),
        stats: PipelineStats,
        onResult: @escaping @Sendable (DetectionResult) -> Void
    ) throws {
        self.model = model
        self.configuration = configuration
        self.stats = stats
        self.onResult = onResult
        decoder = YOLOv8Decoder(
            classCount: model.classCount, anchorCount: model.anchorCount,
            confidenceThreshold: configuration.confidenceThreshold
        )
        slots = SlotPool(try (0..<configuration.framesInFlight).map { _ in try Slot(model: model) })
    }

    /// Runs a few predictions on empty input so the first real frame does not pay for the Neural
    /// Engine's first-use setup.
    func warmUp(predictions: Int = 3) async {
        guard let slot = slots.acquire() else { return }
        defer { slots.release(slot) }
        for _ in 0..<predictions {
            _ = try? await model.model.prediction(from: slot.inputProvider, options: slot.options)
        }
    }

    /// Called for every camera frame, on the camera's frame queue. Preprocessing happens here, on
    /// that queue; prediction and decoding continue on their own task.
    func submit(_ frame: CVPixelBuffer, capturedAt: TimeInterval) {
        guard let slot = slots.acquire() else {
            stats.recordBusyDrop()
            return
        }
        let preprocessMilliseconds: Double
        do {
            preprocessMilliseconds = try prepare(slot, frame: frame)
        } catch {
            slots.release(slot)
            Self.log.error("Preprocessing failed: \(error.localizedDescription)")
            return
        }

        Task(priority: .userInitiated) {
            defer { slots.release(slot) }
            do {
                let result = try await finish(slot, preprocessMilliseconds: preprocessMilliseconds, capturedAt: capturedAt)
                stats.recordProcessed(result.timings, detections: result.detections.count)
                onResult(result)
            } catch {
                Self.log.error("Detection failed: \(error.localizedDescription)")
            }
        }
    }

    /// One frame through the whole pipeline. Used by tests and by anything that is not a live stream.
    func detect(_ frame: CVPixelBuffer, capturedAt: TimeInterval = 0) async throws -> DetectionResult {
        guard let slot = slots.acquire() else { throw PipelineError.busy }
        defer { slots.release(slot) }
        let preprocessMilliseconds = try prepare(slot, frame: frame)
        return try await finish(slot, preprocessMilliseconds: preprocessMilliseconds, capturedAt: capturedAt)
    }

    // MARK: - Stages

    private func prepare(_ slot: Slot, frame: CVPixelBuffer) throws -> Double {
        let signpost = Signposts.pipeline.beginInterval("preprocess")
        defer { Signposts.pipeline.endInterval("preprocess", signpost) }
        let start = clock.now
        slot.letterbox = try preprocessor.fit(frame, into: slot.input)
        slot.frameSize = CGSize(width: CVPixelBufferGetWidth(frame), height: CVPixelBufferGetHeight(frame))
        return start.duration(to: clock.now).milliseconds
    }

    private func finish(_ slot: Slot, preprocessMilliseconds: Double, capturedAt: TimeInterval) async throws -> DetectionResult {
        let predictSignpost = Signposts.pipeline.beginInterval("predict")
        let predictStart = clock.now
        let features = try await model.model.prediction(from: slot.inputProvider, options: slot.options)
        let predictMilliseconds = predictStart.duration(to: clock.now).milliseconds
        Signposts.pipeline.endInterval("predict", predictSignpost)

        let decodeSignpost = Signposts.pipeline.beginInterval("decode")
        defer { Signposts.pipeline.endInterval("decode", decodeSignpost) }
        let decodeStart = clock.now
        guard let output = features.featureValue(for: model.outputName)?.multiArrayValue else {
            throw PipelineError.noOutput
        }
        let detections = try decode(output, slot: slot)
        let postprocessMilliseconds = decodeStart.duration(to: clock.now).milliseconds

        return DetectionResult(
            detections: detections,
            frameSize: slot.frameSize,
            capturedAt: capturedAt,
            timings: StageTimings(
                preprocess: preprocessMilliseconds, predict: predictMilliseconds, postprocess: postprocessMilliseconds
            )
        )
    }

    private func decode(_ output: MLMultiArray, slot: Slot) throws -> [Detection] {
        // Core ML writes into the slot's own array when it can. If it hands back another one,
        // read that array's strides instead (this allocates, so it is the slow path).
        let rowStride: Int
        if output === slot.output {
            rowStride = slot.outputRowStride
        } else {
            stats.recordBackingMiss()
            guard output.strides.count == 3, output.strides[2].intValue == 1 else {
                throw PipelineError.unsupportedOutputLayout
            }
            rowStride = output.strides[1].intValue
        }

        output.withUnsafeBytes { bytes in
            decoder.decode(bytes.baseAddress!.assumingMemoryBound(to: Float.self), rowStride: rowStride, workspace: slot.workspace)
        }
        NonMaxSuppression.apply(
            &slot.workspace.candidates,
            iouThreshold: configuration.iouThreshold,
            maxDetections: configuration.maxDetections,
            into: &slot.workspace.detections
        )

        guard let letterbox = slot.letterbox else { return [] }
        return slot.workspace.detections.map { candidate in
            Detection(
                box: letterbox.sourceRect(x0: candidate.x0, y0: candidate.y0, x1: candidate.x1, y1: candidate.y1),
                classIndex: candidate.classIndex,
                score: candidate.score
            )
        }
    }
}

/// Everything one frame in flight needs, allocated once.
private final class Slot: @unchecked Sendable {
    let input: CVPixelBuffer
    let inputProvider: MLDictionaryFeatureProvider
    let options: MLPredictionOptions
    let output: MLMultiArray
    let outputRowStride: Int
    let workspace: DecoderWorkspace

    var letterbox: Letterbox?
    var frameSize = CGSize.zero

    private struct BufferError: LocalizedError {
        let status: CVReturn
        var errorDescription: String? { "Could not create the model input buffer (CVReturn \(status))." }
    }

    init(model: YOLOModel) throws {
        // IOSurface-backed, like camera frames, so the Neural Engine can read it without a copy.
        let attributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [String: Any](),
            kCVPixelBufferMetalCompatibilityKey: true,
        ]
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, model.inputWidth, model.inputHeight, model.inputPixelFormat,
            attributes as CFDictionary, &buffer
        )
        guard status == kCVReturnSuccess, let buffer else { throw BufferError(status: status) }

        let output = try MLMultiArray(
            shape: [1, NSNumber(value: model.classCount + 4), NSNumber(value: model.anchorCount)],
            dataType: .float32
        )
        let options = MLPredictionOptions()
        options.outputBackings = [model.outputName: output]

        self.input = buffer
        self.inputProvider = try MLDictionaryFeatureProvider(dictionary: [model.inputName: MLFeatureValue(pixelBuffer: buffer)])
        self.options = options
        self.output = output
        self.outputRowStride = output.strides[1].intValue
        self.workspace = DecoderWorkspace(anchorCount: model.anchorCount)
    }
}
