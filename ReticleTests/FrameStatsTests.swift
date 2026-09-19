import Testing
@testable import Reticle

struct FrameStatsTests {
    @Test func reportsRatesOverTheElapsedWindow() {
        let stats = FrameStats(now: 10)
        for _ in 0..<30 { stats.recordFrame() }
        for _ in 0..<3 { stats.recordDrop() }

        let rates = stats.takeRates(at: 11)

        #expect(rates.framesPerSecond == 30)
        #expect(rates.dropsPerSecond == 3)
    }

    @Test func startsANewWindowAfterEachRead() {
        let stats = FrameStats(now: 0)
        for _ in 0..<10 { stats.recordFrame() }
        _ = stats.takeRates(at: 1)

        for _ in 0..<5 { stats.recordFrame() }
        let rates = stats.takeRates(at: 2)

        #expect(rates.framesPerSecond == 5)
        #expect(rates.dropsPerSecond == 0)
    }

    @Test func scalesByWindowLength() {
        let stats = FrameStats(now: 0)
        for _ in 0..<15 { stats.recordFrame() }

        #expect(stats.takeRates(at: 0.5).framesPerSecond == 30)
    }

    @Test func returnsZeroWhenNoTimeHasPassed() {
        let stats = FrameStats(now: 5)
        stats.recordFrame()

        #expect(stats.takeRates(at: 5) == .zero)
    }
}
