import Foundation
import Observation

/// Bridges the camera actor to SwiftUI: the latest camera state and the frame rates, refreshed
/// at 2 Hz so the stats overlay does not redraw per frame.
@MainActor @Observable
final class CameraModel {
    private(set) var state = CameraState.idle
    private(set) var rates = FrameStats.Rates.zero

    @ObservationIgnored let camera = CameraSession()

    func start() {
        Task { await camera.start() }
    }

    func stop() {
        Task { await camera.stop() }
    }

    func observeState() async {
        for await state in camera.states {
            self.state = state
        }
    }

    func pollStats() async {
        while !Task.isCancelled {
            do {
                try await Task.sleep(for: .milliseconds(500))
            } catch {
                return
            }
            rates = camera.stats.takeRates(at: ProcessInfo.processInfo.systemUptime)
        }
    }
}
