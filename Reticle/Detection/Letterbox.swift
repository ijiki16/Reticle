import CoreGraphics

/// How a source frame is fitted into the model's input: scaled to fit with its aspect ratio kept,
/// centred, and padded. Boxes come back in model pixels and must be mapped back to the frame.
struct Letterbox: Equatable, Sendable {
    let sourceWidth: Int
    let sourceHeight: Int
    let targetWidth: Int
    let targetHeight: Int
    let scaledWidth: Int
    let scaledHeight: Int
    /// Top-left corner of the scaled image inside the target.
    let padX: Int
    let padY: Int

    init(sourceWidth: Int, sourceHeight: Int, targetWidth: Int, targetHeight: Int) {
        let scale = min(Double(targetWidth) / Double(sourceWidth), Double(targetHeight) / Double(sourceHeight))
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.targetWidth = targetWidth
        self.targetHeight = targetHeight
        scaledWidth = min(targetWidth, max(1, Int((Double(sourceWidth) * scale).rounded())))
        scaledHeight = min(targetHeight, max(1, Int((Double(sourceHeight) * scale).rounded())))
        padX = (targetWidth - scaledWidth) / 2
        padY = (targetHeight - scaledHeight) / 2
    }

    /// Maps a box from model input pixels back to the source frame, normalized to 0...1 and clamped.
    func sourceRect(x0: Float, y0: Float, x1: Float, y1: Float) -> CGRect {
        let left = normalized(x0, pad: padX, extent: scaledWidth)
        let top = normalized(y0, pad: padY, extent: scaledHeight)
        let right = normalized(x1, pad: padX, extent: scaledWidth)
        let bottom = normalized(y1, pad: padY, extent: scaledHeight)
        return CGRect(x: left, y: top, width: right - left, height: bottom - top)
    }

    private func normalized(_ value: Float, pad: Int, extent: Int) -> CGFloat {
        CGFloat(min(max((value - Float(pad)) / Float(extent), 0), 1))
    }
}
