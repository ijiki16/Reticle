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
    /// What the video output delivers. Core ML image inputs take BGRA, so the ISP converts from the
    /// sensor's YUV. (Vision would accept YUV and skip that conversion.)
    var pixelFormat: OSType = kCVPixelFormatType_32BGRA
}

/// Receives each camera frame with its capture time on the `CACurrentMediaTime()` clock. Called on
/// the camera's frame queue, so it must be quick.
typealias FrameHandler = @Sendable (CVPixelBuffer, TimeInterval) -> Void

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

    init(configuration: CameraConfiguration = CameraConfiguration(), frameHandler: FrameHandler? = nil) {
        let stats = FrameStats(now: ProcessInfo.processInfo.systemUptime)
        let (states, continuation) = AsyncStream.makeStream(of: CameraState.self, bufferingPolicy: .bufferingNewest(1))
        let session = AVCaptureSession()
        self.captureSession = session
        self.previewSession = CaptureSessionHandle(session: session)
        self.configuration = configuration
        self.stats = stats
        self.states = states
        self.stateContinuation = continuation
        self.receiver = FrameReceiver(stats: stats, handler: frameHandler)
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
    ///
    /// Sensor formats are always YUV, so this looks for video-range `420v` whatever `pixelFormat`
    /// is; `pixelFormat` only says what the output converts to.
    private static func bestFormat(for device: AVCaptureDevice, matching c: CameraConfiguration) -> AVCaptureDevice.Format? {
        let matches = device.formats.filter { format in
            let size = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let fps = Double(c.framesPerSecond)
            return size.width == c.width
                && size.height == c.height
                && CMFormatDescriptionGetMediaSubType(format.formatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
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

/// Receives frames on the capture queue, counts them and hands each one to the frame handler.
private final class FrameReceiver: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private let stats: FrameStats
    private let handler: FrameHandler?

    init(stats: FrameStats, handler: FrameHandler?) {
        self.stats = stats
        self.handler = handler
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        stats.recordFrame()
        Signposts.pipeline.emitEvent("frame")
        if let handler, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            handler(pixelBuffer, CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds)
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        stats.recordDrop()
    }
}
