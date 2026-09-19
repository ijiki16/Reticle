import Foundation
import os

/// Per-stage timings, averaged over the window since the last read. The pipeline records into it
/// from several threads; the UI reads it a couple of times a second.
final class PipelineStats: Sendable {
    struct Averages: Equatable, Sendable {
        var processedPerSecond = 0.0
        var busyDropsPerSecond = 0.0
        /// Frames where Core ML did not write into the pre-allocated output array, so the decoder
        /// took the slow path. Should stay at zero.
        var backingMissesPerSecond = 0.0
        var preprocess = 0.0
        var predict = 0.0
        var postprocess = 0.0
        var draw = 0.0
        /// Camera capture to boxes drawn.
        var endToEnd = 0.0
        var detections = 0.0

        static let zero = Averages()
    }

    private struct Window {
        var start: TimeInterval
        var processed = 0
        var busyDrops = 0
        var backingMisses = 0
        var preprocess = 0.0
        var predict = 0.0
        var postprocess = 0.0
        var detections = 0
        var drawn = 0
        var draw = 0.0
        var endToEnd = 0.0
    }

    private let window: OSAllocatedUnfairLock<Window>

    /// `now` is any monotonic clock in seconds, such as `ProcessInfo.systemUptime`.
    init(now: TimeInterval) {
        window = OSAllocatedUnfairLock(initialState: Window(start: now))
    }

    func recordProcessed(_ timings: StageTimings, detections: Int) {
        window.withLock {
            $0.processed += 1
            $0.preprocess += timings.preprocess
            $0.predict += timings.predict
            $0.postprocess += timings.postprocess
            $0.detections += detections
        }
    }

    func recordBusyDrop() {
        window.withLock { $0.busyDrops += 1 }
    }

    func recordBackingMiss() {
        window.withLock { $0.backingMisses += 1 }
    }

    func recordDrawn(drawMilliseconds: Double, endToEndMilliseconds: Double) {
        window.withLock {
            $0.drawn += 1
            $0.draw += drawMilliseconds
            $0.endToEnd += endToEndMilliseconds
        }
    }

    /// Returns the averages since the previous call and starts a new window.
    func takeAverages(at now: TimeInterval) -> Averages {
        window.withLock { window in
            let elapsed = now - window.start
            guard elapsed > 0 else { return .zero }
            let processed = Double(max(window.processed, 1))
            let drawn = Double(max(window.drawn, 1))
            let averages = Averages(
                processedPerSecond: Double(window.processed) / elapsed,
                busyDropsPerSecond: Double(window.busyDrops) / elapsed,
                backingMissesPerSecond: Double(window.backingMisses) / elapsed,
                preprocess: window.preprocess / processed,
                predict: window.predict / processed,
                postprocess: window.postprocess / processed,
                draw: window.draw / drawn,
                endToEnd: window.endToEnd / drawn,
                detections: Double(window.detections) / processed
            )
            window = Window(start: now)
            return averages
        }
    }
}
