import SwiftUI

struct BenchmarkView: View {
    let model: BenchmarkModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Text("Benchmark")
                    .font(.headline)
                Spacer()
                Button("Close") { dismiss() }
                    .disabled(model.isRunning)
            }
            ProgressView(value: Double(model.completed), total: Double(max(model.total, 1)))
            Text(model.status)
                .font(.footnote)
                .foregroundStyle(.secondary)
            ScrollView([.horizontal, .vertical]) {
                Text(model.report)
                    .font(.system(size: 10, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(8)
            }
            .background(.black.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
            HStack {
                Button(model.isRunning ? "Stop" : "Run") {
                    if model.isRunning {
                        model.stop()
                    } else {
                        model.run()
                    }
                }
                .buttonStyle(.borderedProminent)
                Spacer()
                ShareLink(item: model.report)
                    .disabled(model.results.isEmpty)
            }
        }
        .padding()
        .task {
            if LaunchOptions.autorunBenchmark {
                model.run()
            }
        }
    }
}
