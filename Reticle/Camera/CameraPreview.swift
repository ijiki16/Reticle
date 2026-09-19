import AVFoundation
import SwiftUI

/// Shows the camera through `AVCaptureVideoPreviewLayer`, which the GPU composites directly, so
/// no frame is copied on the CPU for display.
struct CameraPreview: UIViewRepresentable {
    let source: CaptureSessionHandle

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = source.session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}
}

final class PreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var previewLayer: AVCaptureVideoPreviewLayer {
        unsafeDowncast(layer, to: AVCaptureVideoPreviewLayer.self)
    }
}
