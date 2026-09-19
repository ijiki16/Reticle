import Foundation
import Observation

/// Bridges the camera actor to SwiftUI: the latest camera state and the frame rates, refreshed
/// at 2 Hz so the stats overlay does not redraw per frame.
@MainActor @Observable
final class CameraModel {
    private(set) var state = CameraState.idle
    private(set) var rates = FrameStats.Rates.zero

    /// Where the camera sends frames. Give it a pipeline to start detecting.
    @ObservationIgnored let sink = FrameSink()
    @ObservationIgnored let camera: CameraSession

    init() {
        let sink = sink
        camera = CameraSession { frame, capturedAt in
            sink.submit(frame, capturedAt: capturedAt)
        }
    }

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
            if LaunchOptions.logStats {
                print(String(format: "camera fps=%.1f late-drops/s=%.1f", rates.framesPerSecond, rates.dropsPerSecond))
                fflush(stdout)
            }
        }
    }
}
