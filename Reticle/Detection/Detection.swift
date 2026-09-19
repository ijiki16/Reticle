import CoreGraphics
import Foundation

struct Detection: Equatable, Sendable {
    /// Normalized to the upright camera frame: 0...1, origin at the top left.
    var box: CGRect
    var classIndex: Int
    var score: Float
}

/// Milliseconds spent in each stage for one frame.
struct StageTimings: Equatable, Sendable {
    var preprocess = 0.0
    var predict = 0.0
    var postprocess = 0.0
}

struct DetectionResult: Sendable {
    var detections: [Detection]
    /// Size in pixels of the upright frame that `detections` are normalized to.
    var frameSize: CGSize
    /// When the camera captured the frame, on the `CACurrentMediaTime()` clock.
    var capturedAt: TimeInterval
    var timings: StageTimings
}
