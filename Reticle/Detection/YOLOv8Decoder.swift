import Foundation

/// A box in model input pixels, before it is mapped back to the camera frame.
struct Candidate: Equatable, Sendable {
    var x0: Float
    var y0: Float
    var x1: Float
    var y1: Float
    var score: Float
    var classIndex: Int
}

/// Buffers the decoder reuses for every frame, so steady-state decoding does not allocate.
final class DecoderWorkspace {
    fileprivate let bestScores: UnsafeMutablePointer<Float>
    fileprivate let bestClasses: UnsafeMutablePointer<UInt8>
    var candidates: [Candidate] = []
    var detections: [Candidate] = []

    init(anchorCount: Int) {
        bestScores = .allocate(capacity: anchorCount)
        bestClasses = .allocate(capacity: anchorCount)
        candidates.reserveCapacity(512)
        detections.reserveCapacity(128)
    }

    deinit {
        bestScores.deallocate()
        bestClasses.deallocate()
    }
}

/// Decodes the raw output of an anchor-free YOLO model (YOLOv8, YOLO11) exported to Core ML
/// without NMS: a `[1, 4 + classCount, anchorCount]` tensor of float32. (The model computes in fp16,
/// but its output is cast to float32.)
///
/// Rows 0-3 hold `cx, cy, w, h` in input pixels and the remaining rows hold one score per class,
/// already in 0...1. There is no objectness value.
struct YOLOv8Decoder: Sendable {
    let classCount: Int
    let anchorCount: Int
    var confidenceThreshold: Float = 0.25
    /// Bounds the work NMS has to do if a noisy frame produces a lot of candidates.
    var maxCandidates = 300

    /// Fills `workspace.candidates` with every anchor whose best class scores above the threshold.
    /// `rowStride` is the distance in elements between two rows of the tensor.
    func decode(_ data: UnsafePointer<Float>, rowStride: Int, workspace: DecoderWorkspace) {
        let best = workspace.bestScores
        let bestClass = workspace.bestClasses

        // Best class per anchor. Scanning class by class reads memory sequentially, which is far
        // faster than gathering 80 strided values for each anchor.
        let firstClass = data + 4 * rowStride
        for anchor in 0..<anchorCount {
            best[anchor] = firstClass[anchor]
            bestClass[anchor] = 0
        }
        for classIndex in 1..<max(classCount, 1) {
            let row = data + (4 + classIndex) * rowStride
            let id = UInt8(truncatingIfNeeded: classIndex)
            for anchor in 0..<anchorCount {
                let score = row[anchor]
                if score > best[anchor] {
                    best[anchor] = score
                    bestClass[anchor] = id
                }
            }
        }

        workspace.candidates.removeAll(keepingCapacity: true)
        for anchor in 0..<anchorCount where best[anchor] > confidenceThreshold {
            let cx = data[anchor]
            let cy = data[rowStride + anchor]
            let halfWidth = data[2 * rowStride + anchor] / 2
            let halfHeight = data[3 * rowStride + anchor] / 2
            workspace.candidates.append(Candidate(
                x0: cx - halfWidth, y0: cy - halfHeight, x1: cx + halfWidth, y1: cy + halfHeight,
                score: best[anchor], classIndex: Int(bestClass[anchor])
            ))
        }

        if workspace.candidates.count > maxCandidates {
            workspace.candidates.sort { $0.score > $1.score }
            workspace.candidates.removeLast(workspace.candidates.count - maxCandidates)
        }
    }
}
