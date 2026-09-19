import CoreML
import Foundation

/// Formats results as a plain-text table that reads the same on screen, in the console and when pasted.
enum BenchmarkReport {
    static func text(for results: [BenchmarkResult]) -> String {
        var lines = [header]
        guard !results.isEmpty else { return lines.joined(separator: "\n") }

        lines.append("")
        lines.append(columns)
        lines += results.map(row)

        if results.contains(where: isThrottled) {
            lines.append("")
            lines.append("* Started or ended at Serious or Critical thermal state, so probably throttled. Do not compare these rows.")
        }

        let fallbacks = results.compactMap(fallbackLine)
        if !fallbacks.isEmpty {
            lines.append("")
            lines.append("Operators planned off the Neural Engine (plan only, confirm in Instruments):")
            lines += fallbacks
        }
        return lines.joined(separator: "\n")
    }

    static var header: String {
        #if DEBUG
        let configuration = "DEBUG BUILD - numbers are not representative, use the Reticle Benchmark scheme"
        #else
        let configuration = "Release"
        #endif
        let process = ProcessInfo.processInfo
        return "Reticle benchmark | \(DeviceInfo.machine) | \(process.operatingSystemVersionString) | \(configuration)"
    }

    private static let columns = [
        pad("model", 20), pad("input", 8), pad("units", 8), left("load", 6), left("ne/gpu/cpu", 11),
        left("p50", 8), left("p95", 8), left("fps@1", 7), left("fps@2", 7), left("fps@3", 7), "  thermal",
    ].joined(separator: " ")

    static func row(_ result: BenchmarkResult) -> String {
        let name = pad(result.modelName, 20)
        let units = pad(result.units.shortName, 8)
        if let failure = result.failure {
            return "\(name) \(pad("-", 8)) \(units) FAILED: \(failure)"
        }

        let latency = result.measurements.first?.latency
        let split = result.plan.map {
            "\(percent($0.neuralEngineShare))/\(percent($0.gpuShare))/\(percent($0.cpuShare))"
        } ?? "n/a"
        let fps = (1...3).map { level in
            let value = result.measurements.first { $0.inFlight == level }?.framesPerSecond
            return left(value.map { String(format: "%.0f", $0) } ?? "-", 7)
        }
        return [
            name, pad(result.inputSize, 8), units,
            left(String(format: "%.1fs", result.loadSeconds), 6),
            left(split, 11),
            left(latency.map { String(format: "%.1fms", $0.p50) } ?? "-", 8),
            left(latency.map { String(format: "%.1fms", $0.p95) } ?? "-", 8),
        ].joined(separator: " ")
            + " " + fps.joined(separator: " ")
            + "  \(result.thermalBefore.name)>\(result.thermalAfter.name)"
            + (isThrottled(result) ? " *" : "")
    }

    private static func isThrottled(_ result: BenchmarkResult) -> Bool {
        result.thermalBefore.rawValue >= ProcessInfo.ThermalState.serious.rawValue
            || result.thermalAfter.rawValue >= ProcessInfo.ThermalState.serious.rawValue
    }

    private static func fallbackLine(_ result: BenchmarkResult) -> String? {
        guard result.units == .cpuAndNeuralEngine, let plan = result.plan, !plan.offNeuralEngine.isEmpty else { return nil }
        let operators = plan.offNeuralEngine
            .sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
            .prefix(6)
            .map { "\($0.key) x\($0.value)" }
            .joined(separator: ", ")
        return "  \(result.modelName): \(operators)"
    }

    private static func percent(_ share: Double) -> String {
        String(format: "%.0f", share * 100)
    }

    private static func pad(_ text: String, _ width: Int) -> String {
        text.count >= width ? text : text + String(repeating: " ", count: width - text.count)
    }

    private static func left(_ text: String, _ width: Int) -> String {
        text.count >= width ? text : String(repeating: " ", count: width - text.count) + text
    }
}

extension MLComputeUnits {
    var shortName: String {
        switch self {
        case .cpuOnly: "cpu"
        case .cpuAndGPU: "cpu+gpu"
        case .cpuAndNeuralEngine: "cpu+ne"
        case .all: "all"
        @unknown default: "unknown"
        }
    }
}

enum DeviceInfo {
    /// The hardware identifier, such as "iPhone11,6" for the iPhone XS Max.
    static var machine: String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
    }
}
