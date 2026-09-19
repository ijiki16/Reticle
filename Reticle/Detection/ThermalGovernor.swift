import Foundation

/// Decides how hard to throttle detection as the phone heats up.
///
/// Level 0 analyses every camera frame. Higher levels analyse only every 2nd, 3rd or 6th frame, so the
/// Neural Engine works less and the phone can cool. The camera preview keeps its full frame rate.
///
/// The rules come from the XS Max thermal runs: a model that keeps the Neural Engine busy for about
/// 40% of each second or more reaches Serious within minutes, and halving the work brings it back
/// below what the cool models needed.
///
/// - Serious steps down to level 1 at once, and to level 2 if it lasts `escalateAfter`.
/// - Critical goes straight to the top level.
/// - Recovery is one level at a time, each after `recoverAfter` of continuous calm, so the app does not
///   flap between rates. Fair is enough to ease from level 2 to level 1, but full rate needs Nominal:
///   at Fair the phone is still warm, and on the XS Max a heavy model went straight from Fair back to
///   Serious within a minute of returning to full rate.
/// - Low Power Mode never goes above level 1.
struct ThermalGovernor {
    /// Camera frames per analysed frame, for each level.
    static let strides = [1, 2, 3, 6]
    static let topLevel = strides.count - 1

    var escalateAfter: TimeInterval = 90
    var recoverAfter: TimeInterval = 60

    private(set) var level = 0
    /// When the current level began.
    private var levelSince: TimeInterval = 0
    /// Since when the phone has been cool enough to step back up, if it has.
    private var coolSince: TimeInterval?

    /// Feed the current conditions. `now` is any monotonic clock in seconds. Returns the level to run at.
    mutating func update(thermalState: ProcessInfo.ThermalState, lowPowerMode: Bool, at now: TimeInterval) -> Int {
        let floor = lowPowerMode ? 1 : 0

        switch thermalState {
        case .critical:
            coolSince = nil
            move(to: Self.topLevel, at: now)
        case .serious:
            coolSince = nil
            if level == 0 {
                move(to: 1, at: now)
            } else if level == 1, now - levelSince >= escalateAfter {
                move(to: 2, at: now)
            } else if level == Self.topLevel, now - levelSince >= recoverAfter {
                move(to: Self.topLevel - 1, at: now)
            }
        default: // Nominal or Fair
            // Only a Nominal phone is cool enough for full rate; Fair can go no lower than level 1.
            let lowest = thermalState == .nominal ? floor : max(floor, 1)
            if level > lowest {
                let since = coolSince ?? now
                coolSince = since
                if now - since >= recoverAfter {
                    move(to: level - 1, at: now)
                    coolSince = nil
                }
            } else {
                coolSince = nil
            }
        }

        if level < floor { move(to: floor, at: now) }
        return level
    }

    private mutating func move(to newLevel: Int, at now: TimeInterval) {
        guard newLevel != level else { return }
        level = newLevel
        levelSince = now
    }
}
