import Foundation

/// Summary of latency samples, in milliseconds.
struct LatencyStats: Equatable, Sendable {
    let count: Int
    let mean: Double
    let p50: Double
    let p95: Double
    let fastest: Double
    let slowest: Double

    /// Returns nil when there are no samples.
    init?(samples: [Double]) {
        guard !samples.isEmpty else { return nil }
        let sorted = samples.sorted()
        count = sorted.count
        mean = sorted.reduce(0, +) / Double(sorted.count)
        p50 = Self.percentile(0.5, of: sorted)
        p95 = Self.percentile(0.95, of: sorted)
        fastest = sorted[0]
        slowest = sorted[sorted.count - 1]
    }

    /// Nearest-rank percentile of samples that are already sorted.
    private static func percentile(_ p: Double, of sorted: [Double]) -> Double {
        let rank = Int((p * Double(sorted.count)).rounded(.up))
        return sorted[max(rank - 1, 0)]
    }
}

extension Duration {
    var milliseconds: Double {
        Double(components.seconds) * 1000 + Double(components.attoseconds) / 1e15
    }

    var secondsValue: Double {
        milliseconds / 1000
    }
}
