import CoreVideo
import Foundation
import os

/// Where the camera sends frames. The pipeline is set later, once the model has loaded; until then
/// frames are simply ignored.
final class FrameSink: Sendable {
    private let pipeline = OSAllocatedUnfairLock<DetectionPipeline?>(initialState: nil)

    func set(_ pipeline: DetectionPipeline?) {
        self.pipeline.withLock { $0 = pipeline }
    }

    func submit(_ frame: CVPixelBuffer, capturedAt: TimeInterval) {
        pipeline.withLock { $0 }?.submit(frame, capturedAt: capturedAt)
    }
}
