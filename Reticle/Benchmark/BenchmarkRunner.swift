import CoreML
import CoreVideo
import Foundation

/// One model on one set of compute units.
struct BenchmarkCase: Sendable {
    let modelURL: URL
    let units: MLComputeUnits

    var modelName: String {
        modelURL.deletingPathExtension().lastPathComponent
    }

    /// Every compiled model in the bundle on each of these. `.cpuOnly` is a control, not a candidate:
    /// if `.cpuAndNeuralEngine` is not clearly faster than it, the Neural Engine is not doing the work.
    /// The order matters (see `all`).
    static let computeUnits: [MLComputeUnits] = [.cpuAndNeuralEngine, .all, .cpuAndGPU, .cpuOnly]

    /// Grouped by compute units in the order above, so the runs we care most about (Neural Engine)
    /// happen first, while the phone is cool. The GPU and CPU runs heat it up the most.
    static func all(in bundle: Bundle) -> [BenchmarkCase] {
        let urls = (bundle.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil) ?? [])
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        return computeUnits.flatMap { units in urls.map { BenchmarkCase(modelURL: $0, units: units) } }
    }
}

/// Throughput and latency with a fixed number of predictions in flight.
struct Measurement: Sendable {
    let inFlight: Int
    let framesPerSecond: Double
    let latency: LatencyStats?
}

struct BenchmarkResult: Sendable {
    let modelName: String
    let units: MLComputeUnits
    var inputSize = ""
    var loadSeconds = 0.0
    var plan: ComputePlanSummary?
    var measurements: [Measurement] = []
    var thermalBefore = ProcessInfo.ThermalState.nominal
    var thermalAfter = ProcessInfo.ThermalState.nominal
    var failure: String?
}

/// Runs a case: load the model, plan the compute devices, warm up, then measure with 1, 2 and 3
/// predictions in flight. Everything here is nonisolated, so it runs off the main thread.
struct BenchmarkRunner: Sendable {
    var warmupCount = 10
    var measureDuration = Duration.seconds(3)
    var inFlightLevels = [1, 2, 3]

    func run(_ benchmarkCase: BenchmarkCase) async -> BenchmarkResult {
        await coolDown()
        var result = BenchmarkResult(modelName: benchmarkCase.modelName, units: benchmarkCase.units)
        result.thermalBefore = ProcessInfo.processInfo.thermalState

        var stage = "plan"
        do {
            result.plan = try? await ComputePlanSummary.load(modelURL: benchmarkCase.modelURL, computeUnits: benchmarkCase.units)

            stage = "load"
            let configuration = MLModelConfiguration()
            configuration.computeUnits = benchmarkCase.units
            let clock = ContinuousClock()
            let loadStart = clock.now
            let model = try await MLModel.load(contentsOf: benchmarkCase.modelURL, configuration: configuration)
            result.loadSeconds = loadStart.duration(to: clock.now).secondsValue

            stage = "prepare input"
            let prepared = try Prepared(model: model)
            result.inputSize = prepared.inputSize
            stage = "warm up"
            for _ in 0..<warmupCount {
                try await prepared.predict()
            }
            for level in inFlightLevels where !Task.isCancelled {
                stage = "measure with \(level) in flight"
                result.measurements.append(try await measure(prepared, inFlight: level))
            }
        } catch {
            let cancelled = Task.isCancelled ? " [benchmark task was cancelled]" : ""
            result.failure = "\(stage): \(error.localizedDescription)\(cancelled)"
        }

        result.thermalAfter = ProcessInfo.processInfo.thermalState
        return result
    }

    private func measure(_ prepared: Prepared, inFlight: Int) async throws -> Measurement {
        let clock = ContinuousClock()
        let start = clock.now
        let deadline = start.advanced(by: measureDuration)

        let samples = try await withThrowingTaskGroup(of: [Double].self) { group in
            for _ in 0..<inFlight {
                group.addTask {
                    var samples: [Double] = []
                    while clock.now < deadline && !Task.isCancelled {
                        let began = clock.now
                        try await prepared.predict()
                        samples.append(began.duration(to: clock.now).milliseconds)
                    }
                    return samples
                }
            }
            var all: [Double] = []
            for try await workerSamples in group {
                all += workerSamples
            }
            return all
        }

        let elapsed = start.duration(to: clock.now).secondsValue
        return Measurement(
            inFlight: inFlight,
            framesPerSecond: elapsed > 0 ? Double(samples.count) / elapsed : 0,
            latency: LatencyStats(samples: samples)
        )
    }

    /// A hot phone throttles and would make later cases look slower, so wait (up to three
    /// minutes) for it to drop to `.fair` or below before starting a case. If it never does,
    /// the report flags the row.
    private func coolDown() async {
        for _ in 0..<36 where ProcessInfo.processInfo.thermalState.rawValue > ProcessInfo.ThermalState.fair.rawValue {
            try? await Task.sleep(for: .seconds(5))
        }
    }
}

/// A loaded model with a ready input. `MLModel` is not `Sendable`, but its prediction methods
/// support concurrent calls and the input is only read, so sharing it between tasks is safe.
private struct Prepared: @unchecked Sendable {
    let model: MLModel
    let input: any MLFeatureProvider
    let inputSize: String

    private enum SetupError: LocalizedError {
        case noImageInput
        case pixelBuffer(CVReturn)

        var errorDescription: String? {
            switch self {
            case .noImageInput: "The model has no image input."
            case .pixelBuffer(let status): "Could not create the input buffer (CVReturn \(status))."
            }
        }
    }

    init(model: MLModel) throws {
        guard let (name, description) = model.modelDescription.inputDescriptionsByName.first(where: { $0.value.type == .image }),
              let constraint = description.imageConstraint
        else {
            throw SetupError.noImageInput
        }

        // IOSurface-backed, like camera frames, so the Neural Engine and GPU can read it directly.
        let attributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [String: Any](),
            kCVPixelBufferMetalCompatibilityKey: true,
        ]
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            constraint.pixelsWide,
            constraint.pixelsHigh,
            constraint.pixelFormatType,
            attributes as CFDictionary,
            &buffer
        )
        guard status == kCVReturnSuccess, let buffer else { throw SetupError.pixelBuffer(status) }
        Self.fillWithPattern(buffer)

        self.model = model
        self.input = try MLDictionaryFeatureProvider(dictionary: [name: MLFeatureValue(pixelBuffer: buffer)])
        self.inputSize = "\(constraint.pixelsWide)x\(constraint.pixelsHigh)"
    }

    func predict() async throws {
        _ = try await model.prediction(from: input)
    }

    /// Latency does not depend on the pixel values; a pattern just keeps the buffer from being blank.
    private static func fillWithPattern(_ buffer: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        for index in 0..<(CVPixelBufferGetBytesPerRow(buffer) * CVPixelBufferGetHeight(buffer)) {
            bytes[index] = UInt8(truncatingIfNeeded: index &* 31)
        }
    }
}
