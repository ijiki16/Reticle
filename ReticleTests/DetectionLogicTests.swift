import CoreGraphics
import Testing
@testable import Reticle

struct LetterboxTests {
    @Test func fitsAPortraitCameraFrameIntoTheModelInput() {
        let letterbox = Letterbox(sourceWidth: 720, sourceHeight: 1280, targetWidth: 352, targetHeight: 640)

        #expect(letterbox.scaledWidth == 352)
        #expect(letterbox.scaledHeight == 626)
        #expect(letterbox.padX == 0)
        #expect(letterbox.padY == 7)
    }

    @Test func matchesTheReferenceGeometryForBus() {
        // The same numbers the Python reference computed: 810x1080 becomes 352x469 at (0, 85).
        let letterbox = Letterbox(sourceWidth: 810, sourceHeight: 1080, targetWidth: 352, targetHeight: 640)

        #expect(letterbox.scaledWidth == 352)
        #expect(letterbox.scaledHeight == 469)
        #expect(letterbox.padY == 85)
    }

    @Test func padsTheSidesOfAWideFrame() {
        let letterbox = Letterbox(sourceWidth: 1280, sourceHeight: 720, targetWidth: 352, targetHeight: 640)

        #expect(letterbox.scaledWidth == 352)
        #expect(letterbox.scaledHeight == 198)
        #expect(letterbox.padY == 221)
    }

    @Test func mapsTheWholeScaledImageToTheWholeFrame() {
        let letterbox = Letterbox(sourceWidth: 720, sourceHeight: 1280, targetWidth: 352, targetHeight: 640)

        let rect = letterbox.sourceRect(x0: 0, y0: 7, x1: 352, y1: 633)

        #expect(rect == CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    @Test func mapsAnInteriorBox() {
        let letterbox = Letterbox(sourceWidth: 720, sourceHeight: 1280, targetWidth: 352, targetHeight: 640)

        // Halfway across and down the scaled image.
        let rect = letterbox.sourceRect(x0: 88, y0: 7 + 156.5, x1: 264, y1: 7 + 469.5)

        #expect(abs(rect.minX - 0.25) < 1e-6)
        #expect(abs(rect.minY - 0.25) < 1e-6)
        #expect(abs(rect.maxX - 0.75) < 1e-6)
        #expect(abs(rect.maxY - 0.75) < 1e-6)
    }

    @Test func clampsBoxesThatReachIntoThePadding() {
        let letterbox = Letterbox(sourceWidth: 720, sourceHeight: 1280, targetWidth: 352, targetHeight: 640)

        let rect = letterbox.sourceRect(x0: -20, y0: 0, x1: 400, y1: 3)

        #expect(rect.minX == 0)
        #expect(rect.maxX == 1)
        #expect(rect.height == 0)
    }
}

struct YOLOv8DecoderTests {
    /// Three classes and five anchors. Rows: cx, cy, w, h, then one score row per class.
    private static let tensor: [Float] = [
        10, 20, 30, 40, 50, // cx
        10, 20, 30, 40, 50, // cy
        4, 4, 4, 4, 4, // w
        6, 6, 6, 6, 6, // h
        0.1, 0.9, 0.0, 0.3, 0.2, // class 0
        0.0, 0.2, 0.6, 0.1, 0.1, // class 1
        0.0, 0.0, 0.0, 0.8, 0.0, // class 2
    ]

    private func decode(_ values: [Float], rowStride: Int, anchors: Int = 5, classes: Int = 3, maxCandidates: Int = 300) -> [Candidate] {
        let decoder = YOLOv8Decoder(classCount: classes, anchorCount: anchors, confidenceThreshold: 0.25, maxCandidates: maxCandidates)
        let workspace = DecoderWorkspace(anchorCount: anchors)
        values.withUnsafeBufferPointer { decoder.decode($0.baseAddress!, rowStride: rowStride, workspace: workspace) }
        return workspace.candidates
    }

    @Test func keepsOnlyAnchorsAboveTheThresholdWithTheirBestClass() {
        let candidates = decode(Self.tensor, rowStride: 5)

        #expect(candidates.map(\.classIndex) == [0, 1, 2])
        #expect(candidates.map(\.score) == [0.9, 0.6, 0.8])
    }

    @Test func convertsCentreSizeToCorners() throws {
        let candidates = decode(Self.tensor, rowStride: 5)
        let first = try #require(candidates.first)

        // Anchor 1: centre (20, 20), size 4 x 6.
        #expect(first.x0 == 18 && first.x1 == 22)
        #expect(first.y0 == 17 && first.y1 == 23)
    }

    @Test func readsRowsThatAreSeparatedByPadding() {
        var padded: [Float] = []
        for row in 0..<7 {
            padded += Self.tensor[(row * 5)..<(row * 5 + 5)]
            padded += [Float](repeating: 99, count: 3) // junk between rows
        }

        #expect(decode(padded, rowStride: 8).map(\.classIndex) == [0, 1, 2])
    }

    @Test func keepsOnlyTheBestCandidatesWhenThereAreTooMany() {
        let anchors = 10
        var values = [Float](repeating: 0, count: 4 * anchors) // cx, cy, w, h
        values += (0..<anchors).map { 0.5 + Float($0) * 0.04 } // one class, rising scores

        let candidates = decode(values, rowStride: anchors, anchors: anchors, classes: 1, maxCandidates: 3)

        // Scores rise 0.5, 0.54, ... 0.86, so the best three are 0.86, 0.82 and 0.78.
        #expect(candidates.count == 3)
        #expect(candidates.allSatisfy { $0.score > 0.77 })
    }

    @Test func findsNothingInAnEmptyFrame() {
        #expect(decode([Float](repeating: 0, count: 7 * 5), rowStride: 5).isEmpty)
    }
}

struct NonMaxSuppressionTests {
    private func box(_ x0: Float, _ y0: Float, _ x1: Float, _ y1: Float, score: Float, class classIndex: Int = 0) -> Candidate {
        Candidate(x0: x0, y0: y0, x1: x1, y1: y1, score: score, classIndex: classIndex)
    }

    private func suppress(_ candidates: [Candidate], iou: Float = 0.45, max: Int = 100) -> [Candidate] {
        var input = candidates
        var output: [Candidate] = []
        NonMaxSuppression.apply(&input, iouThreshold: iou, maxDetections: max, into: &output)
        return output
    }

    @Test func dropsTheLowerScoringOfTwoOverlappingBoxes() {
        let kept = suppress([box(0, 0, 10, 10, score: 0.6), box(1, 1, 11, 11, score: 0.9)])

        #expect(kept.map(\.score) == [0.9])
    }

    @Test func keepsOverlappingBoxesOfDifferentClasses() {
        let kept = suppress([box(0, 0, 10, 10, score: 0.9, class: 0), box(0, 0, 10, 10, score: 0.8, class: 1)])

        #expect(kept.count == 2)
    }

    @Test func keepsBoxesThatBarelyOverlap() {
        let kept = suppress([box(0, 0, 10, 10, score: 0.9), box(8, 8, 18, 18, score: 0.8)])

        #expect(kept.count == 2)
    }

    @Test func returnsBestFirstAndHonoursTheLimit() {
        let candidates = (0..<5).map { box(Float($0) * 20, 0, Float($0) * 20 + 10, 10, score: 0.5 + Float($0) / 10) }

        let kept = suppress(candidates, max: 3)

        #expect(kept.map(\.score) == [0.9, 0.8, 0.7])
    }

    @Test func handlesNoCandidates() {
        #expect(suppress([]).isEmpty)
    }

    @Test func computesIntersectionOverUnion() {
        let a = box(0, 0, 10, 10, score: 1)

        #expect(NonMaxSuppression.intersectionOverUnion(a, a) == 1)
        #expect(NonMaxSuppression.intersectionOverUnion(a, box(20, 20, 30, 30, score: 1)) == 0)
        // Half of each box overlaps: 50 / (100 + 100 - 50).
        let half = NonMaxSuppression.intersectionOverUnion(a, box(5, 0, 15, 10, score: 1))
        #expect(abs(half - 50.0 / 150.0) < 1e-6)
    }
}

struct SlotPoolTests {
    private final class Slot {}

    @Test func handsOutEachSlotOnceThenReportsBusy() {
        let pool = SlotPool([Slot(), Slot()])

        #expect(pool.acquire() != nil)
        #expect(pool.acquire() != nil)
        #expect(pool.acquire() == nil)
    }

    @Test func reusesAReleasedSlot() throws {
        let pool = SlotPool([Slot()])
        let slot = try #require(pool.acquire())
        #expect(pool.acquire() == nil)

        pool.release(slot)

        #expect(pool.acquire() === slot)
    }
}

struct PipelineStatsTests {
    @Test func averagesStageTimingsOverTheWindow() {
        let stats = PipelineStats(now: 0)
        stats.recordProcessed(StageTimings(preprocess: 1, predict: 10, postprocess: 0.5), detections: 2)
        stats.recordProcessed(StageTimings(preprocess: 3, predict: 14, postprocess: 1.5), detections: 4)
        stats.recordBusyDrop()
        stats.recordDrawn(drawMilliseconds: 0.4, endToEndMilliseconds: 40)

        let averages = stats.takeAverages(at: 2)

        #expect(averages.preprocess == 2)
        #expect(averages.predict == 12)
        #expect(averages.postprocess == 1)
        #expect(averages.detections == 3)
        #expect(averages.processedPerSecond == 1)
        #expect(averages.busyDropsPerSecond == 0.5)
        #expect(averages.draw == 0.4)
        #expect(averages.endToEnd == 40)
    }

    @Test func startsANewWindowAfterEachRead() {
        let stats = PipelineStats(now: 0)
        stats.recordProcessed(StageTimings(preprocess: 5, predict: 5, postprocess: 5), detections: 1)
        _ = stats.takeAverages(at: 1)

        #expect(stats.takeAverages(at: 2) == PipelineStats.Averages.zero)
    }
}

struct AspectFillTests {
    @Test func cropsTheSidesOfAPortraitFrameOnATallerScreen() {
        // A 720x1280 frame on an iPhone XS Max (414x896 points) is scaled to the screen height.
        let fill = AspectFill(sourceSize: CGSize(width: 720, height: 1280), viewSize: CGSize(width: 414, height: 896))

        #expect(abs(fill.scale - 0.7) < 1e-9)
        #expect(abs(fill.offset.x - -45) < 1e-9)
        #expect(fill.offset.y == 0)

        let whole = fill.viewRect(forNormalized: CGRect(x: 0, y: 0, width: 1, height: 1))
        #expect(abs(whole.minX - -45) < 1e-9 && abs(whole.width - 504) < 1e-9 && abs(whole.height - 896) < 1e-9)
    }

    @Test func cropsTheTopAndBottomWhenTheViewIsWider() {
        let fill = AspectFill(sourceSize: CGSize(width: 100, height: 100), viewSize: CGSize(width: 1000, height: 500))

        #expect(fill.scale == 10)
        #expect(fill.offset == CGPoint(x: 0, y: -250))
    }

    @Test func mapsAnInteriorBox() {
        let fill = AspectFill(sourceSize: CGSize(width: 100, height: 100), viewSize: CGSize(width: 200, height: 200))

        let rect = fill.viewRect(forNormalized: CGRect(x: 0.25, y: 0.5, width: 0.5, height: 0.25))

        #expect(rect == CGRect(x: 50, y: 100, width: 100, height: 50))
    }

    @Test func fallsBackToTheViewForAnEmptySource() {
        let fill = AspectFill(sourceSize: .zero, viewSize: CGSize(width: 100, height: 100))

        #expect(fill.scale == 1)
    }
}

struct LabelParsingTests {
    @Test func readsUltralyticsNames() {
        #expect(YOLOModel.parseLabels("{0: 'person', 1: 'traffic light', 2: 'car'}") == ["person", "traffic light", "car"])
    }

    @Test func putsLabelsInIndexOrderWhateverTheirOrderInTheText() {
        #expect(YOLOModel.parseLabels("{1: 'b', 0: 'a'}") == ["a", "b"])
    }

    @Test func acceptsDoubleQuotedNames() {
        #expect(YOLOModel.parseLabels(#"{0: "person", 1: "it's odd"}"#) == ["person", "it's odd"])
    }

    @Test func rejectsGapsAndGarbage() {
        #expect(YOLOModel.parseLabels("{0: 'a', 2: 'c'}") == nil)
        #expect(YOLOModel.parseLabels("") == nil)
        #expect(YOLOModel.parseLabels("not names") == nil)
    }
}
