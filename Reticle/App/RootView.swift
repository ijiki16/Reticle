import SwiftUI

struct RootView: View {
    @State private var camera = CameraModel()
    @State private var detection = DetectionController()
    @State private var conditions = DeviceConditions()
    @State private var benchmark = BenchmarkModel()
    @State private var statsLog = StatsLog()
    @State private var showBenchmark = LaunchOptions.autorunBenchmark
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CameraPreview(source: camera.camera.previewSession)
                .ignoresSafeArea()
            BoxOverlay(controller: detection.overlay)
                .ignoresSafeArea()
            CameraMessage(state: camera.state, retry: camera.start)
            StatsHUD(rates: camera.rates, detection: detection, conditions: conditions)
        }
        .overlay(alignment: .topTrailing) {
            Button {
                showBenchmark = true
            } label: {
                Image(systemName: "speedometer")
                    .padding(10)
                    .background(.black.opacity(0.6), in: Circle())
            }
            .padding()
            .accessibilityLabel("Benchmark")
        }
        .fullScreenCover(isPresented: $showBenchmark) {
            BenchmarkView(model: benchmark)
                .preferredColorScheme(.dark)
        }
        .preferredColorScheme(.dark)
        .task { await camera.observeState() }
        .task { await camera.pollStats() }
        .task { await detection.pollStats() }
        .task { await statsLog.run(camera: camera, detection: detection, conditions: conditions) }
        .task {
            // An automatic benchmark run must not compete with the detector for the Neural Engine.
            if !LaunchOptions.autorunBenchmark {
                await detection.load(into: camera.sink)
            }
        }
        .onChange(of: scenePhase, initial: true) { _, _ in updateCameraRunning() }
        // The benchmark needs the Neural Engine and CPU to itself, so the camera pauses behind it.
        .onChange(of: showBenchmark) { _, _ in updateCameraRunning() }
        .onChange(of: camera.state) { _, state in
            // Keep the screen awake only while the camera is running.
            UIApplication.shared.isIdleTimerDisabled = state == .running
            if state != .running {
                detection.overlay.clear()
            }
        }
    }

    private func updateCameraRunning() {
        if scenePhase == .active && !showBenchmark {
            camera.start()
        } else {
            camera.stop()
        }
    }
}

private struct StatsHUD: View {
    let rates: FrameStats.Rates
    let detection: DetectionController
    let conditions: DeviceConditions

    var body: some View {
        VStack {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(String(format: "%.0f fps  %.0f late/s", rates.framesPerSecond, rates.dropsPerSecond))
                    detectionLines
                    Text("Thermal: \(conditions.thermalState.name)")
                        .foregroundStyle(conditions.thermalState.color)
                    if conditions.isLowPowerMode {
                        Text("Low Power Mode")
                            .foregroundStyle(.yellow)
                    }
                }
                .font(.caption.monospacedDigit())
                .padding(8)
                .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 8))
                Spacer()
            }
            Spacer()
        }
        .padding()
    }

    @ViewBuilder
    private var detectionLines: some View {
        switch detection.state {
        case .loading:
            Text("Loading model...")
        case .failed(let message):
            Text(message)
                .foregroundStyle(.red)
                .frame(maxWidth: 260, alignment: .leading)
        case .ready(let modelName):
            let averages = detection.averages
            Text(modelName)
            Text(String(format: "pre %.1f  pred %.1f  post %.1f ms", averages.preprocess, averages.predict, averages.postprocess))
            Text(String(format: "e2e %.0f ms  %.0f objects", averages.endToEnd, averages.detections))
            if averages.busyDropsPerSecond >= 0.5 {
                Text(String(format: "%.0f busy drops/s", averages.busyDropsPerSecond))
                    .foregroundStyle(.yellow)
            }
        }
    }
}

private struct CameraMessage: View {
    let state: CameraState
    let retry: () -> Void
    @Environment(\.openURL) private var openURL

    var body: some View {
        if let content {
            VStack(spacing: 12) {
                Text(content.title)
                    .font(.headline)
                Text(content.detail)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                if let action = content.action {
                    Button(action.title, action: action.perform)
                        .buttonStyle(.borderedProminent)
                }
            }
            .padding(24)
            .background(.black.opacity(0.8), in: RoundedRectangle(cornerRadius: 16))
            .padding(32)
        }
    }

    private struct Content {
        var title: String
        var detail: String
        var action: (title: String, perform: () -> Void)?
    }

    private var content: Content? {
        switch state {
        case .idle, .running:
            nil
        case .interrupted:
            Content(title: "Camera interrupted", detail: "Detection resumes automatically.")
        case .permissionDenied:
            Content(
                title: "Camera access is off",
                detail: "Turn it on in Settings to detect objects.",
                action: ("Open Settings", openSettings)
            )
        case .permissionRestricted:
            Content(
                title: "Camera is restricted",
                detail: "Screen Time or a device profile is blocking camera access."
            )
        case .noCamera:
            Content(title: "No camera found", detail: "This device has no back camera.")
        case .failed(let message):
            Content(title: "Camera problem", detail: message, action: ("Try Again", retry))
        }
    }

    private func openSettings() {
        if let url = URL(string: UIApplication.openSettingsURLString) {
            openURL(url)
        }
    }
}

private extension ProcessInfo.ThermalState {
    var color: Color {
        switch self {
        case .nominal: .green
        case .fair: .yellow
        case .serious: .orange
        case .critical: .red
        @unknown default: .gray
        }
    }
}
