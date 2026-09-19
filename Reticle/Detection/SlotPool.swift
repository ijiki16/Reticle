import os

/// A fixed set of reusable slots. `acquire()` returns nil when every slot is busy, which is how the
/// pipeline decides to drop a frame instead of queueing it.
final class SlotPool<Slot: AnyObject>: @unchecked Sendable {
    private let free: OSAllocatedUnfairLock<[Slot]>

    init(_ slots: [Slot]) {
        free = OSAllocatedUnfairLock(uncheckedState: slots)
    }

    func acquire() -> Slot? {
        free.withLockUnchecked { $0.popLast() }
    }

    func release(_ slot: Slot) {
        free.withLockUnchecked { $0.append(slot) }
    }
}
