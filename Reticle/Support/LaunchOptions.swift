import Foundation

/// Launch arguments for driving the app from a script, e.g. with `devicectl ... -- -log-stats`.
enum LaunchOptions {
    /// Open the benchmark and start it at once.
    static let autorunBenchmark = ProcessInfo.processInfo.arguments.contains("-autorun-benchmark")
    /// Write a line of live statistics every second, to the console and to `Documents/live-stats.log`.
    static let logStats = ProcessInfo.processInfo.arguments.contains("-log-stats")
    /// `-model yolov8s_352x640` detects with that bundled model instead of the default.
    static let modelName: String? = UserDefaults.standard.string(forKey: "model")
    /// `-benchmark-filter yolov8m_352x640,yolo11m_352x640` benchmarks only models with these names.
    static let benchmarkFilter: [String] = UserDefaults.standard.string(forKey: "benchmark-filter")?
        .split(separator: ",").map(String.init) ?? []
    /// `-benchmark-units ne` benchmarks only `.cpuAndNeuralEngine`, the configuration the app uses.
    static let benchmarkNeuralEngineOnly = UserDefaults.standard.string(forKey: "benchmark-units") == "ne"
}
