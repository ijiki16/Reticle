import CoreVideo
import Foundation
import os

/// Something that analyses camera frames.
protocol FrameConsumer: AnyObject, Sendable {
    func submit(_ frame: CVPixelBuffer, capturedAt: TimeInterval)
}

/// Where the camera sends frames. The consumer is set later, once a model has loaded, and can be
/// swapped or removed at any time; until there is one, frames are simply ignored.
///
/// It can also thin the stream: with a stride of 3 only every third frame goes through. That is how
/// detection is throttled while the phone is hot.
final class FrameSink: Sendable {
    private struct State {
        var consumer: (any FrameConsumer)?
        var stride = 1
        var sinceLast = 0
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    var hasConsumer: Bool {
        state.withLock { $0.consumer != nil }
    }

    func set(_ consumer: (any FrameConsumer)?) {
        state.withLock { $0.consumer = consumer }
    }

    func setStride(_ stride: Int) {
        state.withLock {
            $0.stride = max(1, stride)
            $0.sinceLast = 0
        }
    }

    func submit(_ frame: CVPixelBuffer, capturedAt: TimeInterval) {
        let consumer = state.withLock { state -> (any FrameConsumer)? in
            state.sinceLast += 1
            guard state.sinceLast >= state.stride else { return nil }
            state.sinceLast = 0
            return state.consumer
        }
        consumer?.submit(frame, capturedAt: capturedAt)
    }
}
