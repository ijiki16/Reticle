import CoreVideo
import Foundation
import Testing
@testable import Reticle

struct ModelCatalogTests {
    @Test func namesModelsReadably() {
        #expect(ModelCatalog.title(for: "yolov8s_352x640") == "YOLOv8s 352×640")
        #expect(ModelCatalog.title(for: "yolo11n_448x800") == "YOLO11n 448×800")
        #expect(ModelCatalog.title(for: "my_custom_model") == "my_custom_model")
    }

    @Test func listsOnlyModelsThatAreInTheApp_cheapestFirst() {
        let options = ModelCatalog.options(available: ["yolov8s_352x640", "yolov8n_352x640", "yolov8n_192x320"])

        #expect(options.map(\.name) == ["yolov8n_352x640", "yolov8s_352x640"])
    }

    @Test func showsWhatEachChoiceCosts() throws {
        let options = ModelCatalog.options(available: ["yolov8s_352x640", "yolo11s_352x640"])

        let measured = try #require(options.first { $0.name == "yolov8s_352x640" })
        #expect(measured.menuTitle == "YOLOv8s 352×640 · 43.5 mAP · 14 ms · hot after 7 min")
        // No thermal run yet, so no heat note.
        let unmeasured = try #require(options.first { $0.name == "yolo11s_352x640" })
        #expect(unmeasured.detail == "45.4 mAP · 17 ms")
    }

    @Test func offersTheBundledModelsWithTheDefaultFirst() {
        let options = ModelCatalog.options()

        #expect(options.first?.name == DetectionController.modelName)
        #expect(options.count >= 2)
    }
}

struct ThermalGovernorTests {
    private var governor = ThermalGovernor()

    @Test mutating func staysAtFullRateWhileCool() {
        for second in 0..<300 {
            #expect(governor.update(thermalState: second % 2 == 0 ? .nominal : .fair, lowPowerMode: false, at: Double(second)) == 0)
        }
    }

    @Test mutating func stepsDownAtOnceWhenSerious() {
        #expect(governor.update(thermalState: .serious, lowPowerMode: false, at: 10) == 1)
    }

    @Test mutating func stepsDownFurtherIfSeriousLasts() {
        _ = governor.update(thermalState: .serious, lowPowerMode: false, at: 0)

        #expect(governor.update(thermalState: .serious, lowPowerMode: false, at: 89) == 1)
        #expect(governor.update(thermalState: .serious, lowPowerMode: false, at: 90) == 2)
    }

    @Test mutating func goesStraightToTheTopWhenCritical() {
        #expect(governor.update(thermalState: .critical, lowPowerMode: false, at: 0) == ThermalGovernor.topLevel)
    }

    @Test mutating func recoversOneLevelAtATime() {
        _ = governor.update(thermalState: .serious, lowPowerMode: false, at: 0)
        _ = governor.update(thermalState: .serious, lowPowerMode: false, at: 90)
        #expect(governor.level == 2)

        #expect(governor.update(thermalState: .fair, lowPowerMode: false, at: 100) == 2)
        #expect(governor.update(thermalState: .fair, lowPowerMode: false, at: 159) == 2)
        #expect(governor.update(thermalState: .fair, lowPowerMode: false, at: 160) == 1)
        #expect(governor.update(thermalState: .nominal, lowPowerMode: false, at: 219) == 1)
        #expect(governor.update(thermalState: .nominal, lowPowerMode: false, at: 220) == 0)
    }

    @Test mutating func aHotSpellRestartsTheRecoveryClock() {
        _ = governor.update(thermalState: .serious, lowPowerMode: false, at: 0)
        _ = governor.update(thermalState: .fair, lowPowerMode: false, at: 10)
        _ = governor.update(thermalState: .serious, lowPowerMode: false, at: 50)

        _ = governor.update(thermalState: .fair, lowPowerMode: false, at: 60)
        #expect(governor.update(thermalState: .fair, lowPowerMode: false, at: 119) == 1)
        #expect(governor.update(thermalState: .fair, lowPowerMode: false, at: 120) == 0)
    }

    @Test mutating func stepsDownFromCriticalOnceItEases() {
        _ = governor.update(thermalState: .critical, lowPowerMode: false, at: 0)

        #expect(governor.update(thermalState: .serious, lowPowerMode: false, at: 30) == ThermalGovernor.topLevel)
        #expect(governor.update(thermalState: .serious, lowPowerMode: false, at: 60) == ThermalGovernor.topLevel - 1)
    }

    @Test mutating func lowPowerModeKeepsDetectionThrottledEvenWhenCool() {
        #expect(governor.update(thermalState: .nominal, lowPowerMode: true, at: 0) == 1)
        #expect(governor.update(thermalState: .nominal, lowPowerMode: true, at: 1000) == 1)

        // Once Low Power Mode ends it recovers like any other step.
        #expect(governor.update(thermalState: .nominal, lowPowerMode: false, at: 1001) == 1)
        #expect(governor.update(thermalState: .nominal, lowPowerMode: false, at: 1061) == 0)
    }

    @Test func analysesFewerFramesAtEachLevel() {
        #expect(ThermalGovernor.strides == [1, 2, 3, 6])
    }
}

struct FrameSinkTests {
    private final class Counter: FrameConsumer, @unchecked Sendable {
        private let lock = NSLock()
        private var frames = 0
        var count: Int { lock.withLock { frames } }
        func submit(_ frame: CVPixelBuffer, capturedAt: TimeInterval) { lock.withLock { frames += 1 } }
    }

    private func send(_ count: Int, to sink: FrameSink) throws {
        let frame = try TestImages.pixelBuffer(width: 2, height: 2)
        for _ in 0..<count { sink.submit(frame, capturedAt: 0) }
    }

    @Test func passesEveryFrameByDefault() throws {
        let sink = FrameSink()
        let counter = Counter()
        sink.set(counter)

        try send(10, to: sink)

        #expect(counter.count == 10)
    }

    @Test func passesOnlyEveryNthFrameWhenThinned() throws {
        let sink = FrameSink()
        let counter = Counter()
        sink.set(counter)
        sink.setStride(3)

        try send(9, to: sink)

        #expect(counter.count == 3)
    }

    @Test func ignoresFramesWithoutAConsumer() throws {
        let sink = FrameSink()
        #expect(!sink.hasConsumer)

        try send(5, to: sink)

        let counter = Counter()
        sink.set(counter)
        #expect(sink.hasConsumer)
        try send(2, to: sink)
        #expect(counter.count == 2)
    }

    @Test func stopsFeedingTheOldConsumerWhenItIsReplaced() throws {
        let sink = FrameSink()
        let old = Counter()
        let new = Counter()
        sink.set(old)
        try send(3, to: sink)

        sink.set(new)
        try send(4, to: sink)

        #expect(old.count == 3)
        #expect(new.count == 4)
    }
}

@MainActor
struct DetectionControllerTests {
    private func makeDefaults() -> (UserDefaults, cleanup: () -> Void) {
        let name = "reticle-tests-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        return (defaults, { defaults.removePersistentDomain(forName: name) })
    }

    @Test func startsOnTheDefaultModel() async throws {
        let (defaults, cleanup) = makeDefaults()
        defer { cleanup() }
        let controller = DetectionController(defaults: defaults)
        let sink = FrameSink()

        await controller.load(into: sink)

        #expect(controller.state == .ready(modelName: DetectionController.modelName))
        #expect(sink.hasConsumer)
    }

    @Test func switchesModelsAndRemembersTheChoice() async throws {
        let (defaults, cleanup) = makeDefaults()
        defer { cleanup() }
        let controller = DetectionController(defaults: defaults)
        let sink = FrameSink()
        await controller.load(into: sink)

        await controller.select("yolo11n_352x640", into: sink)

        #expect(controller.state == .ready(modelName: "yolo11n_352x640"))
        #expect(controller.selectedModel == "yolo11n_352x640")
        #expect(sink.hasConsumer)
        #expect(defaults.string(forKey: DetectionController.selectionKey) == "yolo11n_352x640")
        // The next launch starts on the remembered model.
        #expect(DetectionController(defaults: defaults).selectedModel == "yolo11n_352x640")
    }

    @Test func refusesAModelThatIsNotInTheAppAndStopsFeedingFrames() async throws {
        let (defaults, cleanup) = makeDefaults()
        defer { cleanup() }
        let controller = DetectionController(defaults: defaults)
        let sink = FrameSink()
        await controller.load(into: sink)

        await controller.select("yolov8x_352x640", into: sink)

        guard case .failed(let message) = controller.state else {
            Issue.record("expected a failure, got \(controller.state)")
            return
        }
        #expect(message.contains("yolov8x_352x640"))
        #expect(!sink.hasConsumer)
    }

    @Test func endsOnTheLastModelWhenSwitchedTwiceInARow() async throws {
        let (defaults, cleanup) = makeDefaults()
        defer { cleanup() }
        let controller = DetectionController(defaults: defaults)
        let sink = FrameSink()
        await controller.load(into: sink)

        let first = Task { await controller.select("yolo11n_352x640", into: sink) }
        let second = Task { await controller.select("yolov8s_352x640", into: sink) }
        await first.value
        await second.value

        #expect(controller.selectedModel == "yolov8s_352x640")
        #expect(controller.state == .ready(modelName: "yolov8s_352x640"))
        #expect(sink.hasConsumer)
    }

    @Test func doesNothingWhenTheModelIsAlreadyRunning() async throws {
        let (defaults, cleanup) = makeDefaults()
        defer { cleanup() }
        let controller = DetectionController(defaults: defaults)
        let sink = FrameSink()
        await controller.load(into: sink)

        await controller.select(DetectionController.modelName, into: sink)

        #expect(controller.state == .ready(modelName: DetectionController.modelName))
        #expect(defaults.string(forKey: DetectionController.selectionKey) == nil)
    }
}
