import CoreText
import SwiftUI
import UIKit

struct OverlayBox {
    var rect: CGRect
    var text: String
    var color: CGColor
}

/// Draws detection boxes with a pool of reused layers, so a new frame only moves and recolours
/// layers that already exist. Implicit animations are off, otherwise every box would glide.
final class BoxOverlayView: UIView {
    private var boxLayers: [BoxLayer] = []

    override init(frame: CGRect) {
        super.init(frame: frame)
        configure()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        configure()
    }

    private func configure() {
        isUserInteractionEnabled = false
        backgroundColor = .clear
    }

    func update(_ boxes: [OverlayBox]) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        defer { CATransaction.commit() }

        while boxLayers.count < boxes.count {
            let boxLayer = BoxLayer(scale: traitCollection.displayScale)
            layer.addSublayer(boxLayer)
            boxLayers.append(boxLayer)
        }
        for (index, boxLayer) in boxLayers.enumerated() {
            if index < boxes.count {
                boxLayer.show(boxes[index])
            } else {
                boxLayer.isHidden = true
            }
        }
    }
}

private final class BoxLayer: CALayer {
    private static let labelHeight: CGFloat = 16
    private static let borderThickness: CGFloat = 2

    private let label = CATextLayer()
    private var text = ""

    init(scale: CGFloat) {
        super.init()
        borderWidth = Self.borderThickness
        contentsScale = scale
        label.contentsScale = scale
        label.font = CTFontCreateWithName("Menlo-Bold" as CFString, 11, nil)
        label.fontSize = 11
        label.foregroundColor = UIColor.black.cgColor
        label.alignmentMode = .left
        label.truncationMode = .end
        addSublayer(label)
    }

    /// Core Animation makes copies of layers with this initializer.
    override init(layer: Any) {
        super.init(layer: layer)
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }

    func show(_ box: OverlayBox) {
        isHidden = false
        frame = box.rect
        borderColor = box.color
        label.backgroundColor = box.color
        if text != box.text {
            text = box.text
            label.string = box.text
        }

        // Above the box, or inside its top edge when the box touches the top of the screen.
        let width = min(max(CGFloat(box.text.count) * 7 + 8, 24), box.rect.width + Self.borderThickness * 2)
        let y = box.rect.minY >= Self.labelHeight ? -Self.labelHeight : 0
        label.frame = CGRect(x: -Self.borderThickness, y: y, width: width, height: Self.labelHeight)
    }
}

/// Hands the overlay view to code outside SwiftUI, so per-frame updates go straight to the layers
/// instead of through a view update.
@MainActor
final class OverlayController {
    weak var view: BoxOverlayView?
    var labels: [String] = []

    private let stats: PipelineStats
    private var colors: [Int: CGColor] = [:]
    private let clock = ContinuousClock()

    init(stats: PipelineStats) {
        self.stats = stats
    }

    func show(_ result: DetectionResult) {
        guard let view else { return }
        let start = clock.now
        let fill = AspectFill(sourceSize: result.frameSize, viewSize: view.bounds.size)
        let boxes = result.detections.map { detection in
            OverlayBox(
                rect: fill.viewRect(forNormalized: detection.box),
                text: title(for: detection),
                color: color(for: detection.classIndex)
            )
        }
        view.update(boxes)

        let drawMilliseconds = start.duration(to: clock.now).milliseconds
        if result.capturedAt > 0 {
            stats.recordDrawn(
                drawMilliseconds: drawMilliseconds,
                endToEndMilliseconds: (CACurrentMediaTime() - result.capturedAt) * 1000
            )
        }
    }

    func clear() {
        view?.update([])
    }

    private func title(for detection: Detection) -> String {
        let name = labels.indices.contains(detection.classIndex) ? labels[detection.classIndex] : "class \(detection.classIndex)"
        return "\(name) \(Int((detection.score * 100).rounded()))%"
    }

    /// Spreads hues with the golden ratio so neighbouring classes get clearly different colours.
    private func color(for classIndex: Int) -> CGColor {
        if let cached = colors[classIndex] { return cached }
        let hue = (CGFloat(classIndex) * 0.618_034).truncatingRemainder(dividingBy: 1)
        let color = UIColor(hue: hue, saturation: 0.85, brightness: 1, alpha: 1).cgColor
        colors[classIndex] = color
        return color
    }
}

struct BoxOverlay: UIViewRepresentable {
    let controller: OverlayController

    func makeUIView(context: Context) -> BoxOverlayView {
        let view = BoxOverlayView()
        controller.view = view
        return view
    }

    func updateUIView(_ uiView: BoxOverlayView, context: Context) {}
}
