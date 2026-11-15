// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Thread-local successful allocator requests; physical RSS is outside scope.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

// Fixed TLS observation storage must cover the full prepared-tree transaction,
// including current compiler initialization. This is a recording capacity, not
// the production memory authority; exhaustion still invalidates the census.
const MAX_LIVE: usize = 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Allocation {
    pub address: usize,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counts {
    pub allocations: usize,
    pub reallocations: usize,
    pub deallocations: usize,
    pub requested_bytes: usize,
    pub overflowed: bool,
    live: [Allocation; MAX_LIVE],
    watched_address: usize,
    watched_live: bool,
    watched_deallocations: usize,
    last_allocation: Option<Allocation>,
}

impl Counts {
    const EMPTY: Self = Self {
        allocations: 0,
        reallocations: 0,
        deallocations: 0,
        requested_bytes: 0,
        overflowed: false,
        live: [Allocation {
            address: 0,
            bytes: 0,
        }; MAX_LIVE],
        watched_address: 0,
        watched_live: false,
        watched_deallocations: 0,
        last_allocation: None,
    };

    pub fn is_empty(&self) -> bool {
        *self == Self::EMPTY
    }

    pub fn live_allocations(&self) -> usize {
        self.live.iter().filter(|entry| entry.address != 0).count()
    }

    pub fn live_requested_bytes(&self) -> Option<usize> {
        self.live
            .iter()
            .filter(|entry| entry.address != 0)
            .try_fold(0usize, |bytes, entry| bytes.checked_add(entry.bytes))
    }

    /// The single live request, without allocating or guessing its address.
    pub fn single_live_allocation(&self) -> Option<Allocation> {
        let mut live = self.live.iter().copied().filter(|entry| entry.address != 0);
        let only = live.next()?;
        live.next().is_none().then_some(only)
    }

    /// The most recent successful allocation/reallocation while that exact
    /// generation remains live. Retirement clears it, without searching back
    /// through older requests or retaining freed addresses in counter equality.
    pub fn last_allocation(&self) -> Option<Allocation> {
        self.last_allocation
    }

    /// Events for one watched allocation generation, not later address reuse.
    pub fn watched_deallocations(&self, address: usize) -> Option<usize> {
        (self.watched_address == address).then_some(self.watched_deallocations)
    }

    pub fn allocation_containing(&self, address: usize) -> Option<Allocation> {
        let mut matches = self.live.iter().copied().filter(|record| {
            record.address != 0
                && record.address <= address
                && record
                    .address
                    .checked_add(record.bytes)
                    .is_some_and(|end| address < end)
        });
        let matched = matches.next();
        assert!(
            matches.next().is_none(),
            "overlapping live allocator records"
        );
        matched
    }
}

#[derive(Clone, Copy)]
struct State {
    active: bool,
    counts: Counts,
}

thread_local! {
    static STATE: Cell<State> = const { Cell::new(State {
        active: false,
        counts: Counts::EMPTY,
    }) };
}

fn add(value: &mut usize, increment: usize, overflowed: &mut bool) {
    let (next, overflow) = value.overflowing_add(increment);
    *value = next;
    *overflowed |= overflow;
}

fn record(removed: usize, allocated: usize, bytes: usize, reallocation: bool) {
    // Const TLS and fixed-capacity storage perform no allocation or locking.
    // try_with also avoids panicking if an inactive thread is tearing down.
    let _ = STATE.try_with(|cell| {
        let mut state = cell.get();
        if !state.active {
            return;
        }
        if removed != 0 {
            if state
                .counts
                .last_allocation
                .is_some_and(|event| event.address == removed)
            {
                state.counts.last_allocation = None;
            }
            if state.counts.watched_live && state.counts.watched_address == removed {
                if !reallocation {
                    add(
                        &mut state.counts.watched_deallocations,
                        1,
                        &mut state.counts.overflowed,
                    );
                }
                // A successful realloc also retires the old generation, even
                // when System returns the same address. It is not a dealloc
                // callback; a later free of the replacement is not this watch.
                state.counts.watched_live = false;
            }
            if let Some(old) = state
                .counts
                .live
                .iter_mut()
                .find(|entry| entry.address == removed)
            {
                *old = Allocation::default();
            }
            add(
                &mut state.counts.deallocations,
                usize::from(!reallocation),
                &mut state.counts.overflowed,
            );
        }
        if allocated != 0 {
            state.counts.last_allocation = Some(Allocation {
                address: allocated,
                bytes,
            });
            add(
                &mut state.counts.allocations,
                usize::from(!reallocation),
                &mut state.counts.overflowed,
            );
            add(
                &mut state.counts.reallocations,
                usize::from(reallocation),
                &mut state.counts.overflowed,
            );
            add(
                &mut state.counts.requested_bytes,
                bytes,
                &mut state.counts.overflowed,
            );
            if let Some(entry) = state
                .counts
                .live
                .iter_mut()
                .find(|entry| entry.address == 0)
            {
                *entry = Allocation {
                    address: allocated,
                    bytes,
                };
            } else {
                state.counts.overflowed = true;
            }
        }
        cell.set(state);
    });
}

struct MeasuredSystem;

// SAFETY: original pointers/layouts go directly to System; fixed-size TLS
// instrumentation neither allocates nor panics from allocator callbacks.
unsafe impl GlobalAlloc for MeasuredSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record(0, pointer.addr(), layout.size(), false);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record(0, pointer.addr(), layout.size(), false);
        }
        pointer
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if !replacement.is_null() {
            record(pointer.addr(), replacement.addr(), new_size, true);
        }
        replacement
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record(pointer.addr(), 0, 0, false);
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: MeasuredSystem = MeasuredSystem;

/// Allocation-free observation while this thread's measurement is active.
pub fn snapshot() -> Option<Counts> {
    STATE.with(|cell| {
        let state = cell.get();
        state.active.then_some(state.counts)
    })
}

/// Watch the one current live allocation through its actual dealloc callback.
/// Starting a watch neither allocates nor changes the measured allocator.
/// A measurement permits one watch; address reuse does not renew that watch.
pub fn watch_deallocation(address: usize) -> bool {
    STATE.with(|cell| {
        let mut state = cell.get();
        if !state.active
            || state.counts.watched_address != 0
            || state
                .counts
                .single_live_allocation()
                .map(|entry| entry.address)
                != Some(address)
        {
            return false;
        }
        state.counts.watched_address = address;
        state.counts.watched_live = true;
        cell.set(state);
        true
    })
}

/// Select one exact tracked live allocation while other owners remain live.
/// This does not change the original sole-allocation watch's precondition.
/// One measurement still permits only one watch, with no renewal on reuse.
pub fn watch_tracked_allocation(address: usize) -> bool {
    STATE.with(|cell| {
        let mut state = cell.get();
        if !state.active
            || state.counts.watched_address != 0
            || state.counts.overflowed
            || address == 0
            || !state
                .counts
                .live
                .iter()
                .any(|entry| entry.address == address)
        {
            return false;
        }
        state.counts.watched_address = address;
        state.counts.watched_live = true;
        cell.set(state);
        true
    })
}

struct Restore(State);

impl Drop for Restore {
    fn drop(&mut self) {
        STATE.with(|cell| cell.set(self.0));
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct NestedMeasurement;

/// Counts on this thread only. The closure must not hand measured allocations
/// to another thread, whose allocator activity this measurement cannot see.
pub fn measure<T>(run: impl FnOnce() -> T) -> Result<(T, Counts), NestedMeasurement> {
    let previous = STATE.with(Cell::get);
    if previous.active {
        return Err(NestedMeasurement);
    }
    let restore = Restore(previous);
    STATE.with(|cell| {
        cell.set(State {
            active: true,
            counts: Counts::EMPTY,
        })
    });
    let value = run();
    let counts = STATE.with(|cell| cell.get().counts);
    drop(restore);
    Ok((value, counts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::sync::Arc;

    #[test]
    fn exact_arc_request_zero_clone_and_last_alias_deallocation_are_measured() {
        let (owner, minted) = measure(|| Arc::new(black_box([2u64, 3, 5, 7]))).unwrap();
        assert_eq!(minted.allocations, 1);
        assert_eq!(minted.reallocations, 0);
        assert_eq!(minted.deallocations, 0);
        let allocation = minted
            .allocation_containing(Arc::as_ptr(&owner).addr())
            .unwrap();
        assert_eq!(
            allocation.bytes,
            size_of::<[u64; 4]>() + 2 * size_of::<usize>()
        );
        assert_eq!(minted.requested_bytes, allocation.bytes);
        assert_eq!(minted.last_allocation(), Some(allocation));
        assert!(!minted.overflowed);
        let (alias, cloned) = measure(|| black_box(&owner).clone()).unwrap();
        assert!(cloned.is_empty());
        assert!(Arc::ptr_eq(&owner, &alias));
        let ((), first) = measure(|| drop(owner)).unwrap();
        assert!(first.is_empty());
        let ((), last) = measure(|| drop(alias)).unwrap();
        assert_eq!(last.allocations, 0);
        assert_eq!(last.deallocations, 1);
        assert!(!last.overflowed);
    }

    #[test]
    fn reallocations_replace_the_old_live_record_and_preserve_actual_system_layout() {
        let ((values, first_address), counts) = measure(|| {
            let mut values = Vec::<u64>::with_capacity(37);
            let first = values.as_ptr().addr();
            values.reserve_exact(100);
            (values, first)
        })
        .unwrap();
        assert_eq!(counts.allocations, 1);
        assert_eq!(counts.reallocations, 1);
        assert!(!counts.overflowed);
        let retained = counts
            .allocation_containing(values.as_ptr().addr())
            .unwrap();
        assert_eq!(retained.bytes, values.capacity() * size_of::<u64>());
        assert_eq!(counts.last_allocation(), Some(retained));
        assert_eq!(
            counts.requested_bytes,
            37 * size_of::<u64>() + retained.bytes
        );
        if first_address != values.as_ptr().addr() {
            assert!(counts.allocation_containing(first_address).is_none());
        }
        let ((), released) = measure(|| drop(values)).unwrap();
        assert_eq!(released.deallocations, 1);
    }

    #[test]
    fn freed_records_are_retired_before_pointer_reuse_and_unwind_restores_state() {
        let ((retained, address), counts) = measure(|| {
            let discarded = black_box(Box::new(3u64));
            drop(discarded);
            let retained = black_box(Box::new(5u64));
            let address = std::ptr::from_ref(retained.as_ref()).addr();
            (retained, address)
        })
        .unwrap();
        assert_eq!(counts.allocations, 2);
        assert_eq!(counts.deallocations, 1);
        assert_eq!(
            counts.allocation_containing(address).unwrap().bytes,
            size_of::<u64>()
        );
        assert!(!counts.overflowed);
        drop(retained);
        let (nested, outer) = measure(|| measure(|| 3)).unwrap();
        assert_eq!(nested, Err(NestedMeasurement));
        assert!(outer.is_empty());
        let unwind = std::panic::catch_unwind(|| measure(|| panic!("measuring tool unwind")));
        assert!(unwind.is_err());
        let (value, after) = measure(|| 17).unwrap();
        assert_eq!(value, 17);
        assert!(after.is_empty());
    }

    #[test]
    fn live_snapshots_retire_last_alias_and_do_not_allocate() {
        assert!(snapshot().is_none());
        let ((held, released), final_counts) = measure(|| {
            let owner = Arc::new(black_box([2u64, 3, 5, 7]));
            let held = snapshot().unwrap();
            assert_eq!(held.live_allocations(), 1);
            assert_eq!(held.live_requested_bytes(), Some(held.requested_bytes));
            let alias = owner.clone();
            assert_eq!(snapshot().unwrap(), held);
            drop(owner);
            assert_eq!(snapshot().unwrap(), held);
            drop(alias);
            let released = snapshot().unwrap();
            assert_eq!(released.live_allocations(), 0);
            assert_eq!(released.live_requested_bytes(), Some(0));
            assert!(held.last_allocation().is_some());
            assert!(released.last_allocation().is_none());
            (held, released)
        })
        .unwrap();
        assert_eq!(final_counts, released);
        assert_eq!(held.allocations, released.allocations);
        assert_eq!(released.deallocations, 1);
        assert!(snapshot().is_none());
    }
}

#[cfg(test)]
mod watched_release_tests {
    use super::*;
    use std::hint::black_box;

    #[test]
    fn watched_generation_retires_once_before_unrelated_address_reuse() {
        let (address, counts) = measure(|| {
            let owner = black_box(Box::new([2u64, 3, 5, 7]));
            let record = snapshot().unwrap().single_live_allocation().unwrap();
            assert!(watch_deallocation(record.address));
            assert!(!watch_deallocation(record.address));
            drop(owner);
            let released = snapshot().unwrap();
            assert_eq!(released.watched_deallocations(record.address), Some(1));
            assert!(released.single_live_allocation().is_none());
            let unrelated = black_box(Box::new([11u64, 13, 17, 19]));
            drop(unrelated);
            assert_eq!(
                snapshot().unwrap().watched_deallocations(record.address),
                Some(1)
            );
            record.address
        })
        .unwrap();
        assert_eq!(counts.allocations, 2);
        assert_eq!(counts.deallocations, 2);
        assert_eq!(counts.watched_deallocations(address), Some(1));
        assert_eq!(counts.live_allocations(), 0);
        assert!(!counts.overflowed);
    }
}

#[cfg(test)]
mod watched_reallocation_tests {
    use super::*;
    use std::hint::black_box;

    #[test]
    fn watched_generation_retires_on_real_reallocation_before_replacement_drop() {
        let ((address, replacement), counts) = measure(|| {
            let mut values = black_box(Vec::<u64>::with_capacity(4));
            let record = snapshot().unwrap().single_live_allocation().unwrap();
            assert!(watch_deallocation(record.address));
            values.reserve_exact(1024);
            let changed = snapshot().unwrap();
            assert_eq!(changed.reallocations, 1);
            assert_eq!(changed.watched_deallocations(record.address), Some(0));
            assert!(changed
                .allocation_containing(values.as_ptr().addr())
                .is_some());
            let replacement = values.as_ptr().addr();
            drop(values);
            assert_eq!(
                snapshot().unwrap().watched_deallocations(record.address),
                Some(0)
            );
            (record.address, replacement)
        })
        .unwrap();
        assert_eq!(counts.allocations, 1);
        assert_eq!(counts.reallocations, 1);
        assert_eq!(counts.deallocations, 1);
        assert_eq!(counts.watched_deallocations(address), Some(0));
        assert_eq!(counts.live_allocations(), 0);
        assert!(!counts.overflowed);
        eprintln!(
            "WATCHED_REALLOCATION_GENERATION old_address={} new_address={} same_address={} reallocations={} watched_deallocations={}",
            address, replacement, address == replacement, counts.reallocations,
            counts.watched_deallocations(address).unwrap(),
        );
    }
}

#[cfg(test)]
mod selected_watch_tests {
    use super::*;
    use std::hint::black_box;

    #[test]
    fn selected_live_watch_preserves_sole_refusal_and_rejects_foreign_or_second_selection() {
        let (address, counts) = measure(|| {
            let first = black_box(Box::new([2_u64, 3, 5, 7]));
            let second = black_box(Box::new([11_u64, 13, 17, 19]));
            let minted = snapshot().unwrap();
            assert_eq!(minted.live_allocations(), 2);
            let first_record = minted.allocation_containing(first.as_ptr().addr()).unwrap();
            let second_record = minted
                .allocation_containing(second.as_ptr().addr())
                .unwrap();
            assert!(!watch_deallocation(first_record.address));
            assert!(!watch_tracked_allocation(0));
            assert!(!watch_tracked_allocation(
                first_record.address.checked_add(1).unwrap()
            ));
            assert!(watch_tracked_allocation(first_record.address));
            assert!(!watch_tracked_allocation(second_record.address));
            drop(second);
            let held = snapshot().unwrap();
            assert_eq!(held.watched_deallocations(first_record.address), Some(0));
            assert_eq!(
                held.allocation_containing(first_record.address),
                Some(first_record)
            );
            drop(first);
            let released = snapshot().unwrap();
            assert_eq!(
                released.watched_deallocations(first_record.address),
                Some(1)
            );
            assert_eq!(released.live_allocations(), 0);
            first_record.address
        })
        .unwrap();
        assert_eq!(counts.allocations, 2);
        assert_eq!(counts.reallocations, 0);
        assert_eq!(counts.deallocations, 2);
        assert_eq!(counts.live_allocations(), 0);
        assert_eq!(counts.watched_deallocations(address), Some(1));
        assert!(!counts.overflowed);
    }
}
