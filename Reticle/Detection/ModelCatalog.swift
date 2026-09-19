import Foundation

/// A bundled model the user can switch to.
struct ModelOption: Identifiable, Equatable, Sendable {
    /// The compiled model's resource name, such as `yolov8s_352x640`.
    let name: String
    let title: String
    /// Accuracy, speed and heat, so the menu shows what each choice costs.
    let detail: String

    var id: String { name }
    var menuTitle: String { detail.isEmpty ? title : "\(title) · \(detail)" }
}

/// The models that can be chosen in the app, best-understood first.
///
/// The numbers are what was measured on the iPhone XS Max: COCO mAP on 1000 val2017 images
/// (`Tools/evaluate_accuracy.py`), Neural Engine latency, and the 15-minute thermal runs. See
/// `docs/benchmarks/`. Models without a thermal run have no heat note.
enum ModelCatalog {
    private static let entries: [(name: String, mAP: Double, latencyMs: Double, heat: String?)] = [
        ("yolov8n_352x640", 35.7, 10.2, "stays cool"),
        ("yolo11n_352x640", 37.0, 12.8, "runs warm"),
        ("yolov8n_448x800", 38.4, 13.0, nil),
        ("yolov8s_352x640", 43.5, 13.9, "hot after 7 min"),
        ("yolo11s_352x640", 45.4, 17.3, nil),
        ("yolov8s_448x800", 46.9, 20.2, nil),
        ("yolov8m_352x640", 49.5, 24.1, "hot after 2 min"),
        ("yolo11m_352x640", 50.2, 38.3, "too slow for 30 fps"),
    ]

    /// The catalogue models that are in the app, cheapest to run first.
    static func options(in bundle: Bundle = .main) -> [ModelOption] {
        let urls = bundle.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil) ?? []
        return options(available: Set(urls.map { $0.deletingPathExtension().lastPathComponent }))
    }

    static func options(available: Set<String>) -> [ModelOption] {
        entries.filter { available.contains($0.name) }.map { entry in
            var detail = String(format: "%.1f mAP · %.0f ms", entry.mAP, entry.latencyMs)
            if let heat = entry.heat { detail += " · \(heat)" }
            return ModelOption(name: entry.name, title: title(for: entry.name), detail: detail)
        }
    }

    /// `yolov8s_352x640` becomes "YOLOv8s 352×640". Anything else is shown as it is.
    static func title(for name: String) -> String {
        let pattern = #/^(yolov8|yolo11)([nsmlx])_(\d+)x(\d+)$/#
        guard let match = name.wholeMatch(of: pattern) else { return name }
        let family = match.output.1 == "yolov8" ? "YOLOv8" : "YOLO11"
        return "\(family)\(match.output.2) \(match.output.3)×\(match.output.4)"
    }
}
