import CoreML

/// Which device Core ML plans to run each operation on.
///
/// Shares are weighted by Core ML's estimated cost. On the XS Max (iOS 18.7) every estimate came
/// back missing or zero, so when the costs add up to zero the shares count operations instead.
/// A plan is not a measurement: confirm the real split in Instruments' Core ML template.
struct ComputePlanSummary: Equatable, Sendable {
    enum Device: Sendable {
        case neuralEngine, gpu, cpu
    }

    private(set) var neuralEngineOps = 0
    private(set) var gpuOps = 0
    private(set) var cpuOps = 0
    private(set) var neuralEngineCost = 0.0
    private(set) var gpuCost = 0.0
    private(set) var cpuCost = 0.0
    /// Operators planned somewhere other than the Neural Engine, with how many of each. These are
    /// the usual performance killers.
    private(set) var offNeuralEngine: [String: Int] = [:]

    var neuralEngineShare: Double { share(cost: neuralEngineCost, ops: neuralEngineOps) }
    var gpuShare: Double { share(cost: gpuCost, ops: gpuOps) }
    var cpuShare: Double { share(cost: cpuCost, ops: cpuOps) }

    private var totalCost: Double { neuralEngineCost + gpuCost + cpuCost }
    private var totalOps: Int { neuralEngineOps + gpuOps + cpuOps }

    mutating func add(operator name: String, on device: Device, cost: Double) {
        switch device {
        case .neuralEngine:
            neuralEngineOps += 1
            neuralEngineCost += cost
        case .gpu:
            gpuOps += 1
            gpuCost += cost
            offNeuralEngine[name, default: 0] += 1
        case .cpu:
            cpuOps += 1
            cpuCost += cost
            offNeuralEngine[name, default: 0] += 1
        }
    }

    private func share(cost: Double, ops: Int) -> Double {
        if totalCost > 0 { return cost / totalCost }
        return totalOps > 0 ? Double(ops) / Double(totalOps) : 0
    }
}

extension ComputePlanSummary {
    private struct NotAProgram: LocalizedError {
        var errorDescription: String? { "Compute plans are only available for ML Program models." }
    }

    static func load(modelURL: URL, computeUnits: MLComputeUnits) async throws -> ComputePlanSummary {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let plan = try await MLComputePlan.load(contentsOf: modelURL, configuration: configuration)
        guard case .program(let program) = plan.modelStructure, let main = program.functions["main"] else {
            throw NotAProgram()
        }

        var summary = ComputePlanSummary()
        for operation in main.block.operations {
            // Constants have no device usage.
            guard let usage = plan.deviceUsage(for: operation) else { continue }
            let cost = plan.estimatedCost(of: operation)?.weight ?? 0
            let device: Device = switch usage.preferred {
            case .neuralEngine: .neuralEngine
            case .gpu: .gpu
            case .cpu: .cpu
            @unknown default: .cpu
            }
            summary.add(operator: operation.operatorName, on: device, cost: cost)
        }
        return summary
    }
}
