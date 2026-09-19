import Foundation
import Observation
import UIKit

@MainActor @Observable
final class BenchmarkModel {
    private(set) var results: [BenchmarkResult] = []
    private(set) var status = "Ready"
    private(set) var isRunning = false
    private(set) var completed = 0

    let cases: [BenchmarkCase]

    @ObservationIgnored private let runner = BenchmarkRunner()
    @ObservationIgnored private var task: Task<Void, Never>?

    init(bundle: Bundle = .main) {
        cases = BenchmarkCase.all(in: bundle)
        if cases.isEmpty {
            status = "No models in the app. Run Tools/export_models.py, then regenerate and rebuild."
        }
    }

    var total: Int { cases.count }
    var report: String { BenchmarkReport.text(for: results) }

    func run() {
        guard !isRunning, !cases.isEmpty else { return }
        isRunning = true
        results = []
        completed = 0
        // Inference keeps the screen busy but not touched, so stop it from dimming and locking.
        UIApplication.shared.isIdleTimerDisabled = true
        emit("=== BENCHMARK BEGIN ===\n\(BenchmarkReport.header)")

        task = Task { [runner, cases] in
            for benchmarkCase in cases {
                if Task.isCancelled { break }
                status = "Running \(benchmarkCase.modelName) on \(benchmarkCase.units.shortName)"
                let result = await runner.run(benchmarkCase)
                results.append(result)
                completed += 1
                emit(BenchmarkReport.row(result))
                saveReport()
            }
            status = Task.isCancelled ? "Stopped" : "Done"
            isRunning = false
            UIApplication.shared.isIdleTimerDisabled = false
            saveReport()
            emit("\n\(report)\nstatus: \(status), \(completed) of \(cases.count) cases\n=== BENCHMARK END ===")
        }
    }

    func stop() {
        task?.cancel()
    }

    /// Where the latest report is kept, so it can be fetched from the phone after the run:
    /// `xcrun devicectl device copy from --device <id> --domain-type appDataContainer
    /// --domain-identifier <bundle id> --source Documents/benchmark-report.txt --destination <path>`
    private static let reportURL = URL.documentsDirectory.appending(path: "benchmark-report.txt")

    /// The first line says how far along the run is, so a reader can tell a partial report from a final one.
    private func saveReport() {
        let text = "status: \(status), \(completed) of \(cases.count) cases\n\n\(report)\n"
        try? text.write(to: Self.reportURL, atomically: true, encoding: .utf8)
    }

    /// Prints to the console, where `devicectl ... --console` picks it up.
    private func emit(_ line: String) {
        print(line)
        fflush(stdout)
    }
}
