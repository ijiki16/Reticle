import Foundation
import Observation

/// Thermal state and Low Power Mode, observable from SwiftUI.
///
/// Lives for the whole process, so the notification observers are never removed.
@MainActor @Observable
final class DeviceConditions {
    private(set) var thermalState = ProcessInfo.processInfo.thermalState
    private(set) var isLowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled

    init() {
        let center = NotificationCenter.default
        center.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.refresh() }
        }
        center.addObserver(forName: .NSProcessInfoPowerStateDidChange, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.refresh() }
        }
    }

    private func refresh() {
        thermalState = ProcessInfo.processInfo.thermalState
        isLowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled
    }
}

extension ProcessInfo.ThermalState {
    var name: String {
        switch self {
        case .nominal: "Nominal"
        case .fair: "Fair"
        case .serious: "Serious"
        case .critical: "Critical"
        @unknown default: "Unknown"
        }
    }
}
