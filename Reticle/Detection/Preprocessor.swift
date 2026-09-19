import Accelerate
import CoreVideo
import Foundation

/// Fits a camera frame into a model input buffer: scales it with vImage, keeping the aspect ratio,
/// and pads the rest with the grey Ultralytics trains with. Everything it needs is allocated once,
/// so a steady stream of same-sized frames does not allocate.
///
/// Not thread safe. Call it from one queue, which is the camera's frame queue.
final class Preprocessor: @unchecked Sendable {
    enum Failure: LocalizedError {
        case unsupportedPixelFormat(OSType)
        case noBaseAddress
        case scaleFailed(Int)

        var errorDescription: String? {
            switch self {
            case .unsupportedPixelFormat(let format): "Frames must be 32BGRA, got pixel format \(format)."
            case .noBaseAddress: "A pixel buffer has no readable memory."
            case .scaleFailed(let code): "vImage could not scale the frame (error \(code))."
            }
        }
    }

    /// Opaque grey in BGRA. The three colour channels are equal, so channel order does not matter.
    private static let padding: UInt32 = 0xFF72_7272

    private var tempBuffer: UnsafeMutableRawPointer?
    private var tempBufferSize = 0

    deinit {
        free(tempBuffer)
    }

    @discardableResult
    func fit(_ source: CVPixelBuffer, into target: CVPixelBuffer) throws -> Letterbox {
        for buffer in [source, target] {
            let format = CVPixelBufferGetPixelFormatType(buffer)
            guard format == kCVPixelFormatType_32BGRA else { throw Failure.unsupportedPixelFormat(format) }
        }
        let letterbox = Letterbox(
            sourceWidth: CVPixelBufferGetWidth(source), sourceHeight: CVPixelBufferGetHeight(source),
            targetWidth: CVPixelBufferGetWidth(target), targetHeight: CVPixelBufferGetHeight(target)
        )

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(target, [])
        defer {
            CVPixelBufferUnlockBaseAddress(target, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let targetBase = CVPixelBufferGetBaseAddress(target)
        else {
            throw Failure.noBaseAddress
        }

        let targetRowBytes = CVPixelBufferGetBytesPerRow(target)
        var sourceImage = vImage_Buffer(
            data: sourceBase, height: vImagePixelCount(letterbox.sourceHeight),
            width: vImagePixelCount(letterbox.sourceWidth), rowBytes: CVPixelBufferGetBytesPerRow(source)
        )
        var scaledImage = vImage_Buffer(
            data: targetBase + letterbox.padY * targetRowBytes + letterbox.padX * 4,
            height: vImagePixelCount(letterbox.scaledHeight),
            width: vImagePixelCount(letterbox.scaledWidth), rowBytes: targetRowBytes
        )

        try ensureTempBuffer(&sourceImage, &scaledImage)
        let status = vImageScale_ARGB8888(&sourceImage, &scaledImage, tempBuffer, vImage_Flags(kvImageNoFlags))
        guard status == kvImageNoError else { throw Failure.scaleFailed(status) }

        fillPadding(of: targetBase, rowBytes: targetRowBytes, letterbox: letterbox)
        return letterbox
    }

    private func ensureTempBuffer(_ source: inout vImage_Buffer, _ target: inout vImage_Buffer) throws {
        let needed = vImageScale_ARGB8888(&source, &target, nil, vImage_Flags(kvImageGetTempBufferSize))
        guard needed >= 0 else { throw Failure.scaleFailed(needed) }
        if needed > tempBufferSize {
            free(tempBuffer)
            tempBuffer = malloc(needed)
            tempBufferSize = needed
        }
    }

    private func fillPadding(of base: UnsafeMutableRawPointer, rowBytes: Int, letterbox: Letterbox) {
        let bottomStart = letterbox.padY + letterbox.scaledHeight
        let rightStart = letterbox.padX + letterbox.scaledWidth
        for y in 0..<letterbox.targetHeight {
            let row = (base + y * rowBytes).assumingMemoryBound(to: UInt32.self)
            if y < letterbox.padY || y >= bottomStart {
                row.update(repeating: Self.padding, count: letterbox.targetWidth)
            } else {
                row.update(repeating: Self.padding, count: letterbox.padX)
                (row + rightStart).update(repeating: Self.padding, count: letterbox.targetWidth - rightStart)
            }
        }
    }
}
