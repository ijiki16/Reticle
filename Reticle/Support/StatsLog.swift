import Foundation
import UIKit

/// With `-log-stats`, writes one line of live numbers per second to the console and to
/// `Documents/live-stats.log`. A long run, such as the 15-minute thermal test, can then be read
/// back from the phone afterwards without keeping a console attached.
@MainActor
final class StatsLog {
    static let fileURL = URL.documentsDirectory.appending(path: "live-stats.log")

    func run(camera: CameraModel, detection: DetectionController, conditions: DeviceConditions) async {
        guard LaunchOptions.logStats else { return }
        UIDevice.current.isBatteryMonitoringEnabled = true
        FileManager.default.createFile(atPath: Self.fileURL.path, contents: nil) // starts an empty file
        let file = try? FileHandle(forWritingTo: Self.fileURL)
        defer { try? file?.close() }

        let start = ProcessInfo.processInfo.systemUptime
        while !Task.isCancelled {
            do {
                try await Task.sleep(for: .seconds(1))
            } catch {
                return
            }
            let averages = detection.averages
            var modelName = "-"
            if case .ready(let name) = detection.state { modelName = name }
            let line = String(
                format: "t=%.0f model=%@ thermal=%@ lowpower=%d battery=%@ mem=%.0f cam=%.1f proc=%.1f busy=%.1f pre=%.1f pred=%.1f post=%.1f draw=%.1f e2e=%.0f obj=%.1f",
                ProcessInfo.processInfo.systemUptime - start, modelName,
                conditions.thermalState.name, conditions.isLowPowerMode ? 1 : 0, Self.batteryDescription,
                MemoryFootprint.megabytes,
                camera.rates.framesPerSecond, averages.processedPerSecond, averages.busyDropsPerSecond,
                averages.preprocess, averages.predict, averages.postprocess, averages.draw,
                averages.endToEnd, averages.detections
            )
            print(line)
            fflush(stdout)
            try? file?.write(contentsOf: Data((line + "\n").utf8))
        }
    }

    /// Charging adds heat, so a thermal run has to say whether the phone was plugged in.
    private static var batteryDescription: String {
        let device = UIDevice.current
        let state = switch device.batteryState {
        case .unplugged: "unplugged"
        case .charging: "charging"
        case .full: "full"
        case .unknown: "unknown"
        @unknown default: "unknown"
        }
        return "\(state)/\(Int((device.batteryLevel * 100).rounded()))"
    }
}

enum MemoryFootprint {
    /// The app's physical memory footprint, the number Xcode's memory gauge shows.
    static var megabytes: Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return status == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : 0
    }
}
