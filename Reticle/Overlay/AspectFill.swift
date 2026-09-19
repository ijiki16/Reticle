import CoreGraphics

/// Where a source frame lands inside a view that shows it with `resizeAspectFill`: scaled until it
/// covers the view, centred, with the overflow cropped. Detections are normalized to the frame, so
/// this maps them to view coordinates the same way the preview layer maps the pixels.
struct AspectFill: Equatable {
    let scale: CGFloat
    let offset: CGPoint
    let sourceSize: CGSize

    init(sourceSize: CGSize, viewSize: CGSize) {
        self.sourceSize = sourceSize
        guard sourceSize.width > 0, sourceSize.height > 0 else {
            scale = 1
            offset = .zero
            return
        }
        scale = max(viewSize.width / sourceSize.width, viewSize.height / sourceSize.height)
        offset = CGPoint(
            x: (viewSize.width - sourceSize.width * scale) / 2,
            y: (viewSize.height - sourceSize.height * scale) / 2
        )
    }

    func viewRect(forNormalized rect: CGRect) -> CGRect {
        CGRect(
            x: offset.x + rect.minX * sourceSize.width * scale,
            y: offset.y + rect.minY * sourceSize.height * scale,
            width: rect.width * sourceSize.width * scale,
            height: rect.height * sourceSize.height * scale
        )
    }
}
