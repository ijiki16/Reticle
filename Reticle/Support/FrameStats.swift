import Foundation
import os

/// Counts frames delivered and dropped by the camera and turns them into per-second rates.
///
/// The capture queue calls `recordFrame()` / `recordDrop()` for every frame, and the UI calls
/// `takeRates(at:)` a couple of times a second, so the counters sit behind a lock.
final class FrameStats: Sendable {
    struct Rates: Equatable, Sendable {
        var framesPerSecond: Double
        var dropsPerSecond: Double

        static let zero = Rates(framesPerSecond: 0, dropsPerSecond: 0)
    }

    private struct Window {
        var frames = 0
        var drops = 0
        var start: TimeInterval
    }

    private let window: OSAllocatedUnfairLock<Window>

    /// `now` is any monotonic clock in seconds, such as `ProcessInfo.systemUptime`.
    init(now: TimeInterval) {
        window = OSAllocatedUnfairLock(initialState: Window(start: now))
    }

    func recordFrame() {
        window.withLock { $0.frames += 1 }
    }

    func recordDrop() {
        window.withLock { $0.drops += 1 }
    }

    /// Returns the rates since the previous call and starts a new measuring window.
    func takeRates(at now: TimeInterval) -> Rates {
        window.withLock { window in
            let elapsed = now - window.start
            guard elapsed > 0 else { return .zero }
            let rates = Rates(
                framesPerSecond: Double(window.frames) / elapsed,
                dropsPerSecond: Double(window.drops) / elapsed
            )
            window = Window(start: now)
            return rates
        }
    }
}
