import CoreML
import CoreVideo

/// A Core ML YOLOv8-style detector whose input and output have been checked against what the
/// decoder expects. `load` throws for any other model, so a wrong model is refused up front
/// instead of crashing on the first frame.
struct YOLOModel: @unchecked Sendable {
    let model: MLModel
    let name: String
    let inputName: String
    let inputWidth: Int
    let inputHeight: Int
    let inputPixelFormat: OSType
    let outputName: String
    let classCount: Int
    let anchorCount: Int
    let labels: [String]

    enum LoadError: LocalizedError {
        case unexpectedInput
        case unsupportedPixelFormat(OSType)
        case unexpectedOutput(String)
        case missingLabels

        var errorDescription: String? {
            switch self {
            case .unexpectedInput:
                "The model must have exactly one image input."
            case .unsupportedPixelFormat(let format):
                "The model's image input uses pixel format \(format); only 32BGRA is supported."
            case .unexpectedOutput(let reason):
                "The model's output is not a YOLOv8-style detection tensor: \(reason)"
            case .missingLabels:
                "The model has no class names in its metadata, or they do not match its output."
            }
        }
    }

    /// Loads off the calling thread; call it from anywhere.
    static func load(url: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) async throws -> YOLOModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let model = try await MLModel.load(contentsOf: url, configuration: configuration)
        return try YOLOModel(validating: model, name: url.deletingPathExtension().lastPathComponent)
    }

    init(validating model: MLModel, name: String) throws {
        let description = model.modelDescription

        guard description.inputDescriptionsByName.count == 1,
              let (inputName, input) = description.inputDescriptionsByName.first,
              input.type == .image,
              let image = input.imageConstraint
        else {
            throw LoadError.unexpectedInput
        }
        guard image.pixelFormatType == kCVPixelFormatType_32BGRA else {
            throw LoadError.unsupportedPixelFormat(image.pixelFormatType)
        }

        guard description.outputDescriptionsByName.count == 1,
              let (outputName, output) = description.outputDescriptionsByName.first,
              let constraint = output.multiArrayConstraint
        else {
            throw LoadError.unexpectedOutput("expected a single multi-array output")
        }
        guard constraint.dataType == .float32 else {
            throw LoadError.unexpectedOutput("expected float32 values")
        }
        let shape = constraint.shape.map(\.intValue)
        guard shape.count == 3, shape[0] == 1 else {
            throw LoadError.unexpectedOutput("expected shape [1, 4 + classes, anchors], got \(shape)")
        }

        let metadata = description.metadata[.creatorDefinedKey] as? [String: String]
        guard let labels = metadata?["names"].flatMap(Self.parseLabels),
              shape[1] == 4 + labels.count,
              labels.count <= Int(UInt8.max)
        else {
            if let metadata, metadata["names"] != nil {
                throw LoadError.unexpectedOutput("\(shape[1]) channels do not match the class names")
            }
            throw LoadError.missingLabels
        }

        self.model = model
        self.name = name
        self.inputName = inputName
        self.inputWidth = image.pixelsWide
        self.inputHeight = image.pixelsHigh
        self.inputPixelFormat = image.pixelFormatType
        self.outputName = outputName
        self.classCount = labels.count
        self.anchorCount = shape[2]
        self.labels = labels
    }

    /// Ultralytics stores the names as a Python dict literal: `{0: 'person', 1: 'bicycle', ...}`.
    /// Returns them in index order, or nil if the indices are not exactly 0..<count.
    static func parseLabels(_ text: String) -> [String]? {
        let entry = #/(\d+):\s*(?:'([^']*)'|"([^"]*)")/#
        var byIndex: [Int: String] = [:]
        for match in text.matches(of: entry) {
            guard let index = Int(match.output.1) else { return nil }
            byIndex[index] = String(match.output.2 ?? match.output.3 ?? "")
        }
        guard !byIndex.isEmpty else { return nil }
        let labels = (0..<byIndex.count).compactMap { byIndex[$0] }
        return labels.count == byIndex.count ? labels : nil
    }
}
