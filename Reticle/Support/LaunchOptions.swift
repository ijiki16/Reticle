import Foundation

/// Launch arguments for driving the app from a script, e.g. with `devicectl ... -- -log-stats`.
enum LaunchOptions {
    /// Open the benchmark and start it at once.
    static let autorunBenchmark = ProcessInfo.processInfo.arguments.contains("-autorun-benchmark")
    /// Print the camera and pipeline statistics to the console twice a second.
    static let logStats = ProcessInfo.processInfo.arguments.contains("-log-stats")
}
