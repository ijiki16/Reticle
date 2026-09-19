import AVFoundation
import OSLog

enum CameraState: Sendable, Equatable {
    case idle
    case running
    case interrupted
    case permissionDenied
    case permissionRestricted
    case noCamera
    case failed(String)
}

/// What the camera should deliver. Kept as a value so the benchmark can sweep it.
struct CameraConfiguration: Sendable {
    var width: Int32 = 1280
    var height: Int32 = 720
    var framesPerSecond: Int32 = 30
    /// `420v` is the sensor's native format, so the ISP has nothing to convert. Switch to
    /// `kCVPixelFormatType_32BGRA` if Core ML rejects YUV input in the spike.
    var pixelFormat: OSType = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
}

/// Lets the main thread give the capture session to `AVCaptureVideoPreviewLayer`, which is the
/// only thing it may do with it. The session is configured and started solely by `CameraSession`.
struct CaptureSessionHandle: @unchecked Sendable {
    let session: AVCaptureSession
}

/// Owns the capture session. The actor runs on its own serial queue, so the blocking
/// `startRunning()` / `stopRunning()` calls never touch the main thread.
actor CameraSession {
    nonisolated let previewSession: CaptureSessionHandle
    /// State changes in order, newest wins. Meant for a single consumer.
    nonisolated let states: AsyncStream<CameraState>
    nonisolated let stats: FrameStats

    private static let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "Reticle", category: "Camera")

    private let executor = DispatchSerialQueue(label: "ge.iurijikidze.Reticle.camera-session", qos: .userInitiated)
    private let frameQueue = DispatchQueue(label: "ge.iurijikidze.Reticle.camera-frames", qos: .userInitiated)
    private let captureSession: AVCaptureSession
    private let stateContinuation: AsyncStream<CameraState>.Continuation
    private let receiver: FrameReceiver
    private let configuration: CameraConfiguration
    private var isConfigured = false
    private var wantsRunning = false
    private var observers: [any NSObjectProtocol] = []

    nonisolated var unownedExecutor: UnownedSerialExecutor {
        executor.asUnownedSerialExecutor()
    }

    init(configuration: CameraConfiguration = CameraConfiguration()) {
        let stats = FrameStats(now: ProcessInfo.processInfo.systemUptime)
        let (states, continuation) = AsyncStream.makeStream(of: CameraState.self, bufferingPolicy: .bufferingNewest(1))
        let session = AVCaptureSession()
        self.captureSession = session
        self.previewSession = CaptureSessionHandle(session: session)
        self.configuration = configuration
        self.stats = stats
        self.states = states
        self.stateContinuation = continuation
        self.receiver = FrameReceiver(stats: stats)
    }

    func start() async {
        wantsRunning = true

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            break
        case .notDetermined:
            guard await AVCaptureDevice.requestAccess(for: .video) else {
                return publish(.permissionDenied)
            }
        case .restricted:
            return publish(.permissionRestricted)
        case .denied:
            return publish(.permissionDenied)
        @unknown default:
            return publish(.permissionDenied)
        }

        // The app can go inactive while the permission alert is up; don't start behind its back.
        guard wantsRunning, configureIfNeeded() else { return }
        if !captureSession.isRunning {
            captureSession.startRunning()
        }
        publish(.running)
    }

    func stop() {
        wantsRunning = false
        guard captureSession.isRunning else { return }
        captureSession.stopRunning()
        publish(.idle)
    }

    private func publish(_ state: CameraState) {
        stateContinuation.yield(state)
    }

    // MARK: - Setup

    private enum SetupError: LocalizedError {
        case noMatchingFormat(CameraConfiguration)
        case cannotAddInput
        case cannotAddOutput

        var errorDescription: String? {
            switch self {
            case .noMatchingFormat(let c):
                "This camera has no \(c.width)x\(c.height) format at \(c.framesPerSecond) fps."
            case .cannotAddInput:
                "The camera could not be added to the capture session."
            case .cannotAddOutput:
                "The video output could not be added to the capture session."
            }
        }
    }

    private func configureIfNeeded() -> Bool {
        guard !isConfigured else { return true }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            publish(.noCamera)
            return false
        }
        do {
            try configure(device: device)
            observeSessionNotifications()
            isConfigured = true
            return true
        } catch {
            Self.log.error("Camera setup failed: \(error.localizedDescription)")
            publish(.failed(error.localizedDescription))
            return false
        }
    }

    private func configure(device: AVCaptureDevice) throws {
        guard let format = Self.bestFormat(for: device, matching: configuration) else {
            throw SetupError.noMatchingFormat(configuration)
        }
        let input = try AVCaptureDeviceInput(device: device)
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: configuration.pixelFormat]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(receiver, queue: frameQueue)

        guard captureSession.canAddInput(input) else { throw SetupError.cannotAddInput }
        guard captureSession.canAddOutput(output) else { throw SetupError.cannotAddOutput }

        captureSession.beginConfiguration()
        captureSession.sessionPreset = .inputPriority
        captureSession.addInput(input)
        captureSession.addOutput(output)
        captureSession.commitConfiguration()

        // Lock the frame rate and turn off the extras that add latency or work.
        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        device.activeFormat = format
        let frameDuration = CMTime(value: 1, timescale: configuration.framesPerSecond)
        device.activeVideoMinFrameDuration = frameDuration
        device.activeVideoMaxFrameDuration = frameDuration
        if format.isVideoHDRSupported {
            device.automaticallyAdjustsVideoHDREnabled = false
            device.isVideoHDREnabled = false
        }

        if let connection = output.connection(with: .video) {
            // Deliver upright portrait buffers: the app is portrait-only and detectors expect
            // upright objects.
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .off
            }
        }

        Self.log.info("Camera format: \(format.debugDescription)")
    }

    /// Prefers a non-HDR, binned format: binning is cheaper for the sensor and the ISP.
    private static func bestFormat(for device: AVCaptureDevice, matching c: CameraConfiguration) -> AVCaptureDevice.Format? {
        let matches = device.formats.filter { format in
            let size = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let fps = Double(c.framesPerSecond)
            return size.width == c.width
                && size.height == c.height
                && CMFormatDescriptionGetMediaSubType(format.formatDescription) == c.pixelFormat
                && format.videoSupportedFrameRateRanges.contains { $0.minFrameRate <= fps && fps <= $0.maxFrameRate }
        }
        return matches.first { !$0.isVideoHDRSupported && $0.isVideoBinned }
            ?? matches.first { !$0.isVideoHDRSupported }
            ?? matches.first
    }

    private func observeSessionNotifications() {
        let center = NotificationCenter.default
        let continuation = stateContinuation
        observers = [
            center.addObserver(forName: AVCaptureSession.wasInterruptedNotification, object: captureSession, queue: nil) { _ in
                continuation.yield(.interrupted)
            },
            center.addObserver(forName: AVCaptureSession.interruptionEndedNotification, object: captureSession, queue: nil) { _ in
                continuation.yield(.running)
            },
            center.addObserver(forName: AVCaptureSession.runtimeErrorNotification, object: captureSession, queue: nil) { note in
                let error = note.userInfo?[AVCaptureSessionErrorKey] as? AVError
                continuation.yield(.failed(error?.localizedDescription ?? "The camera stopped unexpectedly."))
            },
        ]
    }
}

/// Receives frames on the capture queue. For now it only counts them; this is where a frame will
/// be handed to the inference pipeline.
private final class FrameReceiver: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private let stats: FrameStats

    init(stats: FrameStats) {
        self.stats = stats
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        stats.recordFrame()
        Signposts.pipeline.emitEvent("frame")
    }

    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        stats.recordDrop()
    }
}
