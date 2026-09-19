import Testing
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO

private final class BundleToken {}

enum TestImages {
    /// The test bundle, where the test resources live.
    static let bundle = Bundle(for: BundleToken.self)

    static func url(_ name: String, _ ext: String) throws -> URL {
        try #require(bundle.url(forResource: name, withExtension: ext), "missing test resource \(name).\(ext)")
    }

    /// A 32BGRA buffer, optionally filled with one BGRA pixel value.
    static func pixelBuffer(width: Int, height: Int, fill: UInt32 = 0) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [String: Any]()] as CFDictionary
        let status = CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attributes, &buffer)
        let pixelBuffer = try #require(status == kCVReturnSuccess ? buffer : nil)

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        let base = try #require(CVPixelBufferGetBaseAddress(pixelBuffer))
        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        for y in 0..<height {
            (base + y * rowBytes).assumingMemoryBound(to: UInt32.self).update(repeating: fill, count: width)
        }
        return pixelBuffer
    }

    /// The pixel at (x, y) as BGRA bytes.
    static func pixel(_ buffer: CVPixelBuffer, x: Int, y: Int) -> (b: UInt8, g: UInt8, r: UInt8, a: UInt8) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let bytes = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        let offset = y * CVPixelBufferGetBytesPerRow(buffer) + x * 4
        return (bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3])
    }

    /// Decodes a JPEG to raw pixels, ignoring any orientation metadata, into a BGRA buffer.
    static func pixelBuffer(jpeg name: String) throws -> (buffer: CVPixelBuffer, image: CGImage) {
        let source = try #require(CGImageSourceCreateWithURL(try url(name, "jpg") as CFURL, nil))
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        let buffer = try pixelBuffer(width: image.width, height: image.height)

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let context = try #require(CGContext(
            data: CVPixelBufferGetBaseAddress(buffer), width: image.width, height: image.height,
            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ))
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return (buffer, image)
    }
}

/// The detections a plain-numpy decode of the same Core ML model produced for bus.jpg
/// (see docs/REQUIREMENTS.md, section 7: golden-image test against a Python reference).
struct BusReference: Decodable {
    struct Detection: Decodable {
        let classIndex: Int
        let label: String
        let score: Float
        let x0: Double, y0: Double, x1: Double, y1: Double

        var rect: CGRect { CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0) }
    }

    let detections: [Detection]

    static func load() throws -> BusReference {
        try JSONDecoder().decode(BusReference.self, from: Data(contentsOf: TestImages.url("bus_reference", "json")))
    }
}

func intersectionOverUnion(_ a: CGRect, _ b: CGRect) -> Double {
    let intersection = a.intersection(b)
    guard !intersection.isNull else { return 0 }
    let overlap = intersection.width * intersection.height
    return overlap / (a.width * a.height + b.width * b.height - overlap)
}
