import Foundation
import Observation

/// Loads detection models and owns the running pipeline. Frames reach it through the `FrameSink`;
/// boxes go straight to the overlay, and only the statistics pass through SwiftUI, at 2 Hz.
///
/// The model can be switched while the camera runs. The old pipeline stops receiving frames at
/// once, the new one loads and warms up, and results from a superseded model are discarded.
@MainActor @Observable
final class DetectionController {
    enum State: Equatable {
        case loading
        case ready(modelName: String)
        case failed(String)
    }

    /// The model used until the user picks another: the fastest one, which also stays cool. The
    /// benchmark, accuracy and thermal runs behind that choice are in docs/benchmarks.
    nonisolated static let modelName = "yolov8n_352x640"
    /// Where the user's choice is remembered.
    nonisolated static let selectionKey = "selectedModel"

    private(set) var state = State.loading
    private(set) var averages = PipelineStats.Averages.zero
    private(set) var selectedModel: String
    /// 0 is full rate; see `ThermalGovernor`.
    private(set) var throttleLevel = 0

    let options: [ModelOption]

    @ObservationIgnored let overlay: OverlayController
    @ObservationIgnored private let stats: PipelineStats
    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private var pipeline: DetectionPipeline?
    /// Counts model switches, so a load that finished after a newer request can tell it was superseded.
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private var governor = ThermalGovernor()

    private struct ModelMissing: LocalizedError {
        let name: String
        var errorDescription: String? { "\(name).mlmodelc is not in the app. Run Tools/export_models.py and rebuild." }
    }

    /// A `-model` launch argument wins over the remembered choice, which wins over the default.
    init(defaults: UserDefaults = .standard, options: [ModelOption] = ModelCatalog.options()) {
        let stats = PipelineStats(now: ProcessInfo.processInfo.systemUptime)
        self.stats = stats
        self.defaults = defaults
        self.options = options
        overlay = OverlayController(stats: stats)
        selectedModel = LaunchOptions.modelName ?? defaults.string(forKey: Self.selectionKey) ?? Self.modelName
    }

    /// Loads the selected model and starts feeding it frames.
    func load(into sink: FrameSink) async {
        await activate(selectedModel, into: sink)
    }

    /// Switches to another model and remembers the choice.
    func select(_ name: String, into sink: FrameSink) async {
        guard name != selectedModel || state != .ready(modelName: name) else { return }
        defaults.set(name, forKey: Self.selectionKey)
        await activate(name, into: sink)
    }

    private func activate(_ name: String, into sink: FrameSink) async {
        generation += 1
        let ticket = generation
        selectedModel = name
        state = .loading
        // Stop feeding the old pipeline and drop its boxes straight away.
        sink.set(nil)
        pipeline = nil
        overlay.clear()

        do {
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
                throw ModelMissing(name: name)
            }
            let model = try await YOLOModel.load(url: url)
            let overlay = overlay
            let pipeline = try DetectionPipeline(model: model, stats: stats) { [weak self] result in
                DispatchQueue.main.async {
                    MainActor.assumeIsolated {
                        // Results still in flight from a model that has since been replaced are dropped.
                        guard self?.generation == ticket else { return }
                        overlay.show(result)
                    }
                }
            }
            await pipeline.warmUp()

            guard ticket == generation else { return }
            overlay.labels = model.labels
            self.pipeline = pipeline
            sink.set(pipeline)
            state = .ready(modelName: model.name)
        } catch {
            guard ticket == generation else { return }
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

    /// Once a second, checks the phone's temperature and thins the frame stream if it is too hot.
    /// `-no-throttle` turns this off, to measure a model's heat on its own.
    func throttle(with conditions: DeviceConditions, sink: FrameSink) async {
        guard !LaunchOptions.noThrottle else { return }
        while !Task.isCancelled {
            do {
                try await Task.sleep(for: .seconds(1))
            } catch {
                return
            }
            let level = governor.update(
                thermalState: conditions.thermalState,
                lowPowerMode: conditions.isLowPowerMode,
                at: ProcessInfo.processInfo.systemUptime
            )
            if level != throttleLevel {
                throttleLevel = level
                sink.setStride(ThermalGovernor.strides[level])
            }
        }
    }
}
