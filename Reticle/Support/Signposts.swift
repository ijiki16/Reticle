import Foundation
import OSLog

/// Signposts for Instruments (os_signpost track). Add one interval per pipeline stage here as the
/// stages appear: preprocess, prediction, decode, draw.
enum Signposts {
    static let pipeline = OSSignposter(
        subsystem: Bundle.main.bundleIdentifier ?? "Reticle",
        category: "Pipeline"
    )
}
