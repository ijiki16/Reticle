import Foundation

enum NonMaxSuppression {
    /// Greedy NMS that only compares boxes of the same class. The IoU threshold is separate from
    /// the confidence threshold, which was already applied by the decoder.
    ///
    /// Sorts `candidates` by score and writes the survivors, best first, to `results`.
    static func apply(
        _ candidates: inout [Candidate],
        iouThreshold: Float,
        maxDetections: Int,
        into results: inout [Candidate]
    ) {
        results.removeAll(keepingCapacity: true)
        candidates.sort { $0.score > $1.score }
        let count = candidates.count

        withUnsafeTemporaryAllocation(of: Bool.self, capacity: count) { suppressed in
            suppressed.initialize(repeating: false)
            for i in 0..<count where !suppressed[i] {
                results.append(candidates[i])
                if results.count >= maxDetections { break }
                for j in (i + 1)..<count where !suppressed[j] && candidates[j].classIndex == candidates[i].classIndex {
                    if intersectionOverUnion(candidates[i], candidates[j]) > iouThreshold {
                        suppressed[j] = true
                    }
                }
            }
        }
    }

    static func intersectionOverUnion(_ a: Candidate, _ b: Candidate) -> Float {
        let width = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
        let height = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
        let intersection = width * height
        let union = area(a) + area(b) - intersection
        return union > 0 ? intersection / union : 0
    }

    private static func area(_ c: Candidate) -> Float {
        max(0, c.x1 - c.x0) * max(0, c.y1 - c.y0)
    }
}
