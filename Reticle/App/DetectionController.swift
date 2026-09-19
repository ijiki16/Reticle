import Foundation
import Observation

/// Loads the detector and owns the pipeline. Frames reach it through the `FrameSink`; boxes go
/// straight to the overlay, and only the statistics pass through SwiftUI, at 2 Hz.
@MainActor @Observable
final class DetectionController {
    enum State: Equatable {
        case loading
        case ready(modelName: String)
        case failed(String)
    }

    /// The model this build detects with. The benchmark picked YOLOv8n at 352x640: the fastest
    /// model that keeps a portrait camera frame nearly undistorted.
    nonisolated static let modelName = "yolov8n_352x640"

    private(set) var state = State.loading
    private(set) var averages = PipelineStats.Averages.zero

    @ObservationIgnored let overlay: OverlayController
    @ObservationIgnored private let stats: PipelineStats
    @ObservationIgnored private var pipeline: DetectionPipeline?

    private struct ModelMissing: LocalizedError {
        let name: String
        var errorDescription: String? { "\(name).mlmodelc is not in the app. Run Tools/export_models.py and rebuild." }
    }

    init() {
        let stats = PipelineStats(now: ProcessInfo.processInfo.systemUptime)
        self.stats = stats
        overlay = OverlayController(stats: stats)
    }

    func load(into sink: FrameSink) async {
        state = .loading
        do {
            let modelName = LaunchOptions.modelName ?? Self.modelName
            guard let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
                throw ModelMissing(name: modelName)
            }
            let model = try await YOLOModel.load(url: url)
            let overlay = overlay
            let pipeline = try DetectionPipeline(model: model, stats: stats) { result in
                DispatchQueue.main.async {
                    MainActor.assumeIsolated { overlay.show(result) }
                }
            }
            await pipeline.warmUp()

            overlay.labels = model.labels
            self.pipeline = pipeline
            sink.set(pipeline)
            state = .ready(modelName: model.name)
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func pollStats() async {
        while !Task.isCancelled {
            do {
                try await Task.sleep(for: .milliseconds(500))
            } catch {
                return
            }
            averages = stats.takeAverages(at: ProcessInfo.processInfo.systemUptime)
        }
    }
}
