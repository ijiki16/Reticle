import Testing
@testable import Reticle

struct LatencyStatsTests {
    @Test func summarisesSamples() throws {
        let stats = try #require(LatencyStats(samples: Array(stride(from: 100.0, through: 1.0, by: -1))))

        #expect(stats.count == 100)
        #expect(stats.mean == 50.5)
        #expect(stats.p50 == 50)
        #expect(stats.p95 == 95)
        #expect(stats.fastest == 1)
        #expect(stats.slowest == 100)
    }

    @Test func handlesASingleSample() throws {
        let stats = try #require(LatencyStats(samples: [7]))

        #expect(stats.p50 == 7)
        #expect(stats.p95 == 7)
    }

    @Test func hasNoStatsWithoutSamples() {
        #expect(LatencyStats(samples: []) == nil)
    }

    @Test func convertsDurationToMilliseconds() {
        #expect(Duration.milliseconds(1500).milliseconds == 1500)
        #expect(Duration.seconds(2).secondsValue == 2)
    }
}

struct ComputePlanSummaryTests {
    @Test func weighsDevicesByCost() {
        var summary = ComputePlanSummary()
        summary.add(operator: "conv", on: .neuralEngine, cost: 8)
        summary.add(operator: "softmax", on: .cpu, cost: 1)
        summary.add(operator: "mul", on: .gpu, cost: 1)

        #expect(summary.neuralEngineShare == 0.8)
        #expect(summary.gpuShare == 0.1)
        #expect(summary.cpuShare == 0.1)
    }

    @Test func listsOperatorsThatLeaveTheNeuralEngine() {
        var summary = ComputePlanSummary()
        summary.add(operator: "conv", on: .neuralEngine, cost: 5)
        summary.add(operator: "softmax", on: .cpu, cost: 1)
        summary.add(operator: "softmax", on: .gpu, cost: 1)

        #expect(summary.offNeuralEngine == ["softmax": 2])
    }

    @Test func countsOperationsWhenCoreMLGivesNoCosts() {
        var summary = ComputePlanSummary()
        for _ in 0..<3 { summary.add(operator: "conv", on: .neuralEngine, cost: 0) }
        summary.add(operator: "reshape", on: .cpu, cost: 0)

        #expect(summary.neuralEngineShare == 0.75)
        #expect(summary.cpuShare == 0.25)
        #expect(summary.gpuShare == 0)
    }

    @Test func sharesAreZeroForAnEmptyPlan() {
        #expect(ComputePlanSummary().neuralEngineShare == 0)
    }
}
