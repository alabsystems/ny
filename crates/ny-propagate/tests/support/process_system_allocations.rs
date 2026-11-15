// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Captured Rust System requests across all threads, never RSS or C/GPU heap.
//! Fixed atomic records retire a captured generation on any thread's callback.
//! Records outside the active window may remain until their actual free. They
//! are excluded from later epochs, rather than attributed to a query owner.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const MAX_LIVE: usize = 512;
const FREE: usize = 0;
const LIVE: usize = 1;
const REALLOC_PENDING: usize = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Allocation {
    pub address: usize,
    pub bytes: usize,
    pub generation: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counts {
    pub allocations: usize,
    pub reallocations: usize,
    pub deallocations: usize,
    pub untracked_deallocations: usize,
    pub requested_bytes: usize,
    pub overflowed: bool,
    live: [Allocation; MAX_LIVE],
    last: Option<Allocation>,
    watched_address: usize,
    watched_deallocations: usize,
}
impl Counts {
    pub fn live_allocations(&self) -> usize {
        self.live.iter().filter(|r| r.address != 0).count()
    }
    pub fn live_requested_bytes(&self) -> Option<usize> {
        self.live
            .iter()
            .filter(|r| r.address != 0)
            .try_fold(0usize, |n, r| n.checked_add(r.bytes))
    }
    pub fn last_allocation(&self) -> Option<Allocation> {
        self.last
    }
    pub fn watched_deallocations(&self, address: usize) -> Option<usize> {
        (self.watched_address == address).then_some(self.watched_deallocations)
    }
    pub fn allocation_containing(&self, address: usize) -> Option<Allocation> {
        let mut matches = self.live.iter().copied().filter(|r| {
            r.address != 0
                && r.address <= address
                && r.address
                    .checked_add(r.bytes)
                    .is_some_and(|end| address < end)
        });
        let found = matches.next();
        assert!(
            matches.next().is_none(),
            "overlapping live System request records"
        );
        found
    }
}

struct Record {
    state: AtomicUsize,
    address: AtomicUsize,
    bytes: AtomicUsize,
    epoch: AtomicUsize,
    generation: AtomicUsize,
}
static RECORDS: [Record; MAX_LIVE] = [const {
    Record {
        state: AtomicUsize::new(FREE),
        address: AtomicUsize::new(0),
        bytes: AtomicUsize::new(0),
        epoch: AtomicUsize::new(0),
        generation: AtomicUsize::new(0),
    }
}; MAX_LIVE];
static LOCKED: AtomicBool = AtomicBool::new(false);
static ACTIVE: AtomicUsize = AtomicUsize::new(0);
static EPOCH: AtomicUsize = AtomicUsize::new(0);
// Occupied slots include transactions reserved before System.realloc.
static LIVE_RECORDS: AtomicUsize = AtomicUsize::new(0);
static PENDING_REALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static REALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static UNTRACKED_DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static REQUESTED: AtomicUsize = AtomicUsize::new(0);
static OVERFLOWED: AtomicBool = AtomicBool::new(false);
static POISONED: AtomicBool = AtomicBool::new(false);
static LAST_ADDRESS: AtomicUsize = AtomicUsize::new(0);
static LAST_BYTES: AtomicUsize = AtomicUsize::new(0);
static LAST_GENERATION: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_SERIAL: AtomicUsize = AtomicUsize::new(0);
static WATCH_GENERATION: AtomicUsize = AtomicUsize::new(0);
static WATCH_ADDRESS: AtomicUsize = AtomicUsize::new(0);
static WATCH_LIVE: AtomicBool = AtomicBool::new(false);
static WATCH_DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
thread_local! {
    static OWNER_EPOCH: Cell<usize> = const { Cell::new(0) };
}

// Every field remains atomic: this instrumentation needs no UnsafeCell or
// unsafe Sync implementation. A callback holds only this allocation-free spin
// guard; it never calls System, TLS initialization, formatting or user code.
struct Guard;
impl Guard {
    fn lock() -> Self {
        while LOCKED
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }
        Self
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        LOCKED.store(false, Ordering::Release);
    }
}
fn add(counter: &AtomicUsize, amount: usize) {
    let (value, overflow) = counter.load(Ordering::Relaxed).overflowing_add(amount);
    counter.store(value, Ordering::Relaxed);
    if overflow {
        OVERFLOWED.store(true, Ordering::Relaxed);
    }
}

fn poison() {
    OVERFLOWED.store(true, Ordering::Relaxed);
    POISONED.store(true, Ordering::Relaxed);
}
fn subtract_one(counter: &AtomicUsize) {
    if let Some(n) = counter.load(Ordering::Relaxed).checked_sub(1) {
        counter.store(n, Ordering::Relaxed);
    } else {
        poison();
    }
}
fn clear_slot(slot: &Record) {
    slot.state.store(FREE, Ordering::Relaxed);
    slot.address.store(0, Ordering::Relaxed);
    slot.bytes.store(0, Ordering::Relaxed);
    slot.epoch.store(0, Ordering::Relaxed);
    slot.generation.store(0, Ordering::Relaxed);
    subtract_one(&LIVE_RECORDS);
}
#[derive(Clone, Copy)]
struct Original {
    address: usize,
    bytes: usize,
    epoch: usize,
    generation: usize,
}
impl Original {
    fn read(slot: &Record) -> Self {
        Self {
            address: slot.address.load(Ordering::Relaxed),
            bytes: slot.bytes.load(Ordering::Relaxed),
            epoch: slot.epoch.load(Ordering::Relaxed),
            generation: slot.generation.load(Ordering::Relaxed),
        }
    }
}
fn retire_observations(old: Original, active: usize, deallocation: bool) {
    if old.epoch != active || active == 0 {
        return;
    }
    if LAST_ADDRESS.load(Ordering::Relaxed) == old.address
        && LAST_GENERATION.load(Ordering::Relaxed) == old.generation
    {
        LAST_ADDRESS.store(0, Ordering::Relaxed);
        LAST_BYTES.store(0, Ordering::Relaxed);
        LAST_GENERATION.store(0, Ordering::Relaxed);
    }
    if WATCH_LIVE.load(Ordering::Relaxed)
        && WATCH_ADDRESS.load(Ordering::Relaxed) == old.address
        && WATCH_GENERATION.load(Ordering::Relaxed) == old.generation
    {
        if deallocation {
            add(&WATCH_DEALLOCATIONS, 1);
        }
        // Successful realloc retires the old generation even at the same base.
        // Deallocating its replacement cannot increment the original watch.
        WATCH_LIVE.store(false, Ordering::Relaxed);
    }
}
fn install(slot: &Record, address: usize, bytes: usize, epoch: usize, generation: usize) {
    slot.address.store(address, Ordering::Relaxed);
    slot.bytes.store(bytes, Ordering::Relaxed);
    slot.epoch.store(epoch, Ordering::Relaxed);
    slot.generation.store(generation, Ordering::Relaxed);
    slot.state.store(LIVE, Ordering::Relaxed);
}
fn allocation_event(
    address: usize,
    bytes: usize,
    reallocation: bool,
    owner: usize,
    active: usize,
) -> usize {
    add(&ALLOCATIONS, usize::from(!reallocation));
    add(&REALLOCATIONS, usize::from(reallocation));
    add(&REQUESTED, bytes);
    add(&ALLOCATION_SERIAL, 1);
    let generation = ALLOCATION_SERIAL.load(Ordering::Relaxed);
    if owner == active {
        LAST_ADDRESS.store(address, Ordering::Relaxed);
        LAST_BYTES.store(bytes, Ordering::Relaxed);
        LAST_GENERATION.store(generation, Ordering::Relaxed);
    }
    generation
}
fn record_allocation(address: usize, bytes: usize, reallocation: bool) {
    if ACTIVE.load(Ordering::Acquire) == 0 {
        return;
    }
    let owner = OWNER_EPOCH.try_with(Cell::get).unwrap_or(0);
    let _guard = Guard::lock();
    let active = ACTIVE.load(Ordering::Relaxed);
    if active == 0 {
        return;
    }
    let generation = allocation_event(address, bytes, reallocation, owner, active);
    if RECORDS.iter().any(|r| {
        r.state.load(Ordering::Relaxed) == LIVE && r.address.load(Ordering::Relaxed) == address
    }) {
        poison();
        return;
    }
    if let Some(slot) = RECORDS
        .iter()
        .find(|r| r.state.load(Ordering::Relaxed) == FREE)
    {
        install(slot, address, bytes, active, generation);
        add(&LIVE_RECORDS, 1);
    } else {
        OVERFLOWED.store(true, Ordering::Relaxed);
    }
}
fn record_deallocation(address: usize) {
    if ACTIVE.load(Ordering::Acquire) == 0 && LIVE_RECORDS.load(Ordering::Acquire) == 0 {
        return;
    }
    let _guard = Guard::lock();
    let active = ACTIVE.load(Ordering::Relaxed);
    // An old pending realloc may share its former address with this NEW live
    // allocation. Only the LIVE slot represents this deallocation's generation.
    let old = RECORDS.iter().find(|r| {
        r.state.load(Ordering::Relaxed) == LIVE && r.address.load(Ordering::Relaxed) == address
    });
    let old_epoch = old.map(|r| r.epoch.load(Ordering::Relaxed));
    if let Some(slot) = old {
        retire_observations(Original::read(slot), active, true);
        clear_slot(slot);
    }
    if active != 0 {
        add(&DEALLOCATIONS, 1);
        if old_epoch != Some(active) {
            add(&UNTRACKED_DEALLOCATIONS, 1);
        }
    }
}

// This ticket is private and move-only. Its reserved slot cannot be reused by
// alloc/dealloc callbacks while System.realloc runs OUTSIDE the spin guard.
// Completing by slot/generation, never by old address, closes the address-reuse
// window between the real System call and the observer's completion callback.
struct Reallocation {
    slot: Option<usize>,
    original: Option<Original>,
    owner: usize,
}
fn begin_reallocation(address: usize) -> Option<Reallocation> {
    let owner = OWNER_EPOCH.try_with(Cell::get).unwrap_or(0);
    let _guard = Guard::lock();
    if ACTIVE.load(Ordering::Relaxed) == 0 && LIVE_RECORDS.load(Ordering::Relaxed) == 0 {
        return None;
    }
    let original_slot = RECORDS.iter().position(|r| {
        r.state.load(Ordering::Relaxed) == LIVE && r.address.load(Ordering::Relaxed) == address
    });
    let original = original_slot.map(|i| Original::read(&RECORDS[i]));
    let slot = original_slot.or_else(|| {
        RECORDS
            .iter()
            .position(|r| r.state.load(Ordering::Relaxed) == FREE)
    });
    if let Some(i) = slot {
        if original.is_none() {
            add(&LIVE_RECORDS, 1);
        }
        RECORDS[i].state.store(REALLOC_PENDING, Ordering::Relaxed);
    } else {
        OVERFLOWED.store(true, Ordering::Relaxed);
    }
    add(&PENDING_REALLOCATIONS, 1);
    Some(Reallocation {
        slot,
        original,
        owner,
    })
}
fn finish_reallocation(ticket: Reallocation, address: usize, bytes: usize) {
    let _guard = Guard::lock();
    let active = ACTIVE.load(Ordering::Relaxed);
    let slot = ticket.slot.map(|i| &RECORDS[i]);
    if let Some(slot) = slot {
        if slot.state.load(Ordering::Relaxed) != REALLOC_PENDING
            || ticket.original.is_some_and(|old| {
                slot.address.load(Ordering::Relaxed) != old.address
                    || slot.epoch.load(Ordering::Relaxed) != old.epoch
                    || slot.generation.load(Ordering::Relaxed) != old.generation
            })
        {
            poison();
            subtract_one(&PENDING_REALLOCATIONS);
            return;
        }
    }
    if address == 0 {
        // Failed System.realloc leaves the original allocation unchanged.
        // Restore exactly its prior slot, generation, epoch, watch and last event.
        if let Some(slot) = slot {
            if let Some(old) = ticket.original {
                install(slot, old.address, old.bytes, old.epoch, old.generation);
            } else {
                clear_slot(slot);
            }
        }
    } else {
        if let Some(old) = ticket.original {
            retire_observations(old, active, false);
        }
        if active == 0 {
            if let Some(slot) = slot {
                clear_slot(slot);
            }
        } else {
            let generation = allocation_event(address, bytes, true, ticket.owner, active);
            let duplicate = RECORDS.iter().any(|r| {
                r.state.load(Ordering::Relaxed) == LIVE
                    && r.address.load(Ordering::Relaxed) == address
            });
            if duplicate {
                poison();
                if let Some(slot) = slot {
                    clear_slot(slot);
                }
            } else if let Some(slot) = slot {
                install(slot, address, bytes, active, generation);
            }
            // With no reserved slot the table overflow already invalidated the
            // epoch. No address-based retirement or invented partial repair.
        }
    }
    subtract_one(&PENDING_REALLOCATIONS);
}

struct MeasuredSystem;
// SAFETY: original pointers/layouts go directly to System. Fixed atomic
// instrumentation neither allocates nor panics inside allocator callbacks.
unsafe impl GlobalAlloc for MeasuredSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(pointer.addr(), layout.size(), false);
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record_allocation(pointer.addr(), layout.size(), false);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record_deallocation(pointer.addr());
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let ticket = begin_reallocation(pointer.addr());
        let allocated = unsafe { System.realloc(pointer, layout, new_size) };
        if let Some(ticket) = ticket {
            finish_reallocation(ticket, allocated.addr(), new_size);
        } else if !allocated.is_null() {
            record_allocation(allocated.addr(), new_size, true);
        }
        allocated
    }
}
#[global_allocator]
static ALLOCATOR: MeasuredSystem = MeasuredSystem;

struct Scope {
    epoch: usize,
}
impl Drop for Scope {
    fn drop(&mut self) {
        let _ = OWNER_EPOCH.try_with(|owner| owner.set(0));
        let _guard = Guard::lock();
        if ACTIVE.load(Ordering::Relaxed) == self.epoch {
            ACTIVE.store(0, Ordering::Release);
        }
        // Captured records remain until their actual free/realloc on ANY
        // thread. Later epochs neither erase nor claim these live generations.
    }
}
fn begin() -> Option<Scope> {
    if OWNER_EPOCH.try_with(Cell::get).ok()? != 0 {
        return None;
    }
    let epoch = {
        let _guard = Guard::lock();
        if ACTIVE.load(Ordering::Relaxed) != 0
            || PENDING_REALLOCATIONS.load(Ordering::Relaxed) != 0
            || POISONED.load(Ordering::Relaxed)
        {
            return None;
        }
        let next = EPOCH.load(Ordering::Relaxed).checked_add(1)?;
        EPOCH.store(next, Ordering::Relaxed);
        for counter in [
            &ALLOCATIONS,
            &REALLOCATIONS,
            &DEALLOCATIONS,
            &UNTRACKED_DEALLOCATIONS,
            &REQUESTED,
            &LAST_ADDRESS,
            &LAST_BYTES,
            &LAST_GENERATION,
            &ALLOCATION_SERIAL,
            &WATCH_GENERATION,
            &WATCH_ADDRESS,
            &WATCH_DEALLOCATIONS,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        WATCH_LIVE.store(false, Ordering::Relaxed);
        OVERFLOWED.store(false, Ordering::Relaxed);
        ACTIVE.store(next, Ordering::Release);
        next
    };
    OWNER_EPOCH.with(|owner| owner.set(epoch));
    Some(Scope { epoch })
}
pub fn snapshot() -> Option<Counts> {
    let owner = OWNER_EPOCH.try_with(Cell::get).ok()?;
    let _guard = Guard::lock();
    if owner == 0
        || ACTIVE.load(Ordering::Relaxed) != owner
        || PENDING_REALLOCATIONS.load(Ordering::Relaxed) != 0
    {
        return None;
    }
    let mut live = [Allocation::default(); MAX_LIVE];
    for (out, record) in live.iter_mut().zip(&RECORDS) {
        if record.state.load(Ordering::Relaxed) == LIVE
            && record.epoch.load(Ordering::Relaxed) == owner
        {
            *out = Allocation {
                address: record.address.load(Ordering::Relaxed),
                bytes: record.bytes.load(Ordering::Relaxed),
                generation: record.generation.load(Ordering::Relaxed),
            };
        }
    }
    let address = LAST_ADDRESS.load(Ordering::Relaxed);
    Some(Counts {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        reallocations: REALLOCATIONS.load(Ordering::Relaxed),
        deallocations: DEALLOCATIONS.load(Ordering::Relaxed),
        untracked_deallocations: UNTRACKED_DEALLOCATIONS.load(Ordering::Relaxed),
        requested_bytes: REQUESTED.load(Ordering::Relaxed),
        overflowed: OVERFLOWED.load(Ordering::Relaxed) || POISONED.load(Ordering::Relaxed),
        live,
        last: (address != 0).then(|| Allocation {
            address,
            bytes: LAST_BYTES.load(Ordering::Relaxed),
            generation: LAST_GENERATION.load(Ordering::Relaxed),
        }),
        watched_address: WATCH_ADDRESS.load(Ordering::Relaxed),
        watched_deallocations: WATCH_DEALLOCATIONS.load(Ordering::Relaxed),
    })
}
pub fn watch_tracked_allocation(address: usize) -> bool {
    let owner = OWNER_EPOCH.try_with(Cell::get).unwrap_or(0);
    let _guard = Guard::lock();
    if owner == 0
        || ACTIVE.load(Ordering::Relaxed) != owner
        || OVERFLOWED.load(Ordering::Relaxed)
        || POISONED.load(Ordering::Relaxed)
        || PENDING_REALLOCATIONS.load(Ordering::Relaxed) != 0
        || address == 0
        || WATCH_ADDRESS.load(Ordering::Relaxed) != 0
    {
        return false;
    }
    let selected = RECORDS.iter().find(|r| {
        r.state.load(Ordering::Relaxed) == LIVE
            && r.epoch.load(Ordering::Relaxed) == owner
            && r.address.load(Ordering::Relaxed) == address
            && r.bytes.load(Ordering::Relaxed) != 0
    });
    if let Some(selected) = selected {
        WATCH_GENERATION.store(
            selected.generation.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        WATCH_ADDRESS.store(address, Ordering::Relaxed);
        WATCH_LIVE.store(true, Ordering::Relaxed);
        WATCH_DEALLOCATIONS.store(0, Ordering::Relaxed);
    }
    selected.is_some()
}
pub fn measure<T>(f: impl FnOnce() -> T) -> Option<(T, Counts)> {
    let scope = begin()?;
    let value = f();
    let counts = snapshot()?;
    drop(scope);
    Some((value, counts))
}

#[cfg(test)]
pub fn run_observer_controls() {
    controls::run_all();
}

#[cfg(test)]
mod controls {
    use super::*;
    use std::sync::{mpsc, Arc};

    pub(super) fn run_all() {
        process_system_counts_real_arc_request_zero_clone_and_last_alias_once();
        process_system_foreign_thread_free_retires_original_selected_generation();
        process_system_foreign_thread_reallocation_retires_watch_before_replacement_free();
        process_system_nested_scope_and_foreign_selection_cannot_reset_census();
        process_system_preexisting_free_is_counted_without_becoming_query_live_storage();
        process_system_real_capacity_overflow_invalidates_partial_census_until_epoch_end();
        process_system_unwind_releases_capture_and_does_not_import_old_epoch_records();
        process_system_concurrent_real_vec_reallocations_leave_no_retired_generation();
        process_system_failed_completion_protocol_restores_exact_original_record();
        process_system_concurrent_address_reuse_protocol_completes_exact_reserved_slot();
    }

    // Real System stress: workers are spawned and report ready before capture;
    // they finish their Vec requests before the snapshot and leave only after it.
    // This exercises concurrent actual alloc/realloc/free, but does not claim a
    // particular native address-reuse interleaving occurred.
    fn process_system_concurrent_real_vec_reallocations_leave_no_retired_generation() {
        let ready = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(AtomicBool::new(false));
        let workers: [_; 4] = std::array::from_fn(|worker| {
            let ready = ready.clone();
            let start = start.clone();
            let done = done.clone();
            let release = release.clone();
            std::thread::spawn(move || {
                ready.fetch_add(1, Ordering::Release);
                while !start.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                for iteration in 0..512 {
                    let mut values = Vec::with_capacity(2);
                    values.push(worker);
                    values.push(iteration);
                    values.reserve_exact(64 + (iteration % 4) * 32);
                    std::hint::black_box(&values);
                    values.shrink_to_fit();
                    std::hint::black_box(&values);
                    drop(values);
                }
                done.fetch_add(1, Ordering::Release);
                while !release.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
            })
        });
        while ready.load(Ordering::Acquire) != 4 {
            std::hint::spin_loop();
        }
        let ((), counts) = measure(|| {
            start.store(true, Ordering::Release);
            while done.load(Ordering::Acquire) != 4 {
                std::hint::spin_loop();
            }
        })
        .unwrap();
        assert!(!counts.overflowed);
        assert!(counts.allocations >= 4 * 512);
        assert!(counts.reallocations >= 4 * 512);
        assert_eq!(counts.live_allocations(), 0);
        release.store(true, Ordering::Release);
        for worker in workers {
            worker.join().unwrap();
        }
    }

    // State-transition control, NOT an observed native System failure. An
    // injected null completion must restore the actual Box record unchanged.
    fn process_system_failed_completion_protocol_restores_exact_original_record() {
        measure(|| {
            let value = Box::new([47u64; 4]);
            let before = snapshot().unwrap();
            let original = before.last_allocation().unwrap();
            assert!(watch_tracked_allocation(original.address));
            let selected = snapshot().unwrap();
            let ticket = begin_reallocation(original.address).unwrap();
            assert!(snapshot().is_none());
            assert!(begin().is_none());
            assert!(!watch_tracked_allocation(original.address));
            finish_reallocation(ticket, 0, original.bytes * 2);
            assert_eq!(snapshot().unwrap(), selected);
            drop(value);
            assert_eq!(
                snapshot().unwrap().watched_deallocations(original.address),
                Some(1)
            );
        })
        .unwrap();
    }

    // Protocol interleaving only: synthetic addresses are never dereferenced,
    // passed to System, or reported as actual allocator observations. A worker
    // reserves an old generation; another callback installs a new generation
    // at the same old base before completion. Completion must preserve the new
    // record and retire ONLY its own reserved original slot/watch generation.
    fn process_system_concurrent_address_reuse_protocol_completes_exact_reserved_slot() {
        let phase = Arc::new(AtomicUsize::new(0));
        let worker_phase = phase.clone();
        let worker = std::thread::spawn(move || {
            while worker_phase.load(Ordering::Acquire) != 1 {
                std::hint::spin_loop();
            }
            let ticket = begin_reallocation(0x10_000).unwrap();
            worker_phase.store(2, Ordering::Release);
            while worker_phase.load(Ordering::Acquire) != 3 {
                std::hint::spin_loop();
            }
            finish_reallocation(ticket, 0x20_000, 128);
            worker_phase.store(4, Ordering::Release);
            while worker_phase.load(Ordering::Acquire) != 5 {
                std::hint::spin_loop();
            }
        });
        measure(|| {
            record_allocation(0x10_000, 32, false);
            let original = snapshot().unwrap().last_allocation().unwrap();
            assert!(watch_tracked_allocation(original.address));
            phase.store(1, Ordering::Release);
            while phase.load(Ordering::Acquire) != 2 {
                std::hint::spin_loop();
            }
            assert!(snapshot().is_none());
            record_allocation(original.address, 32, false);
            let replacement_generation = LAST_GENERATION.load(Ordering::Relaxed);
            assert_ne!(replacement_generation, original.generation);
            phase.store(3, Ordering::Release);
            while phase.load(Ordering::Acquire) != 4 {
                std::hint::spin_loop();
            }
            let completed = snapshot().unwrap();
            let replacement = completed.allocation_containing(original.address).unwrap();
            assert_eq!(replacement.generation, replacement_generation);
            assert_eq!(completed.watched_deallocations(original.address), Some(0));
            assert_eq!(completed.last_allocation(), Some(replacement));
            record_deallocation(original.address);
            record_deallocation(0x20_000);
            assert_eq!(
                snapshot().unwrap().watched_deallocations(original.address),
                Some(0)
            );
        })
        .unwrap();
        phase.store(5, Ordering::Release);
        worker.join().unwrap();
    }

    fn process_system_counts_real_arc_request_zero_clone_and_last_alias_once() {
        let ((), counts) = measure(|| {
            let owner = Arc::new(37u64);
            let first = snapshot().unwrap();
            assert!(!first.overflowed);
            assert_eq!(first.allocations, 1);
            let allocation = first.last_allocation().unwrap();
            assert!(watch_tracked_allocation(allocation.address));
            let clone = owner.clone();
            assert_eq!(snapshot().unwrap().allocations, first.allocations);
            drop(owner);
            assert_eq!(
                snapshot()
                    .unwrap()
                    .watched_deallocations(allocation.address),
                Some(0)
            );
            drop(clone);
            let last = snapshot().unwrap();
            assert_eq!(last.watched_deallocations(allocation.address), Some(1));
            assert_eq!(last.live_allocations(), 0);
        })
        .unwrap();
        assert!(!counts.overflowed);
        assert_eq!(counts.allocations, 1);
        assert_eq!(counts.deallocations, 1);
        assert_eq!(counts.untracked_deallocations, 0);
    }

    fn process_system_foreign_thread_free_retires_original_selected_generation() {
        let (send, receive) = mpsc::sync_channel::<Box<[u64; 4]>>(0);
        let (done_send, done_receive) = mpsc::sync_channel(0);
        let worker = std::thread::spawn(move || {
            drop(receive.recv().unwrap());
            done_send.send(()).unwrap();
        });
        measure(|| {
            let value = Box::new([13u64; 4]);
            let allocation = snapshot()
                .unwrap()
                .allocation_containing(std::ptr::from_ref(&*value).addr())
                .unwrap();
            assert!(watch_tracked_allocation(allocation.address));
            send.send(value).unwrap();
            done_receive.recv().unwrap();
            let after = snapshot().unwrap();
            assert!(!after.overflowed);
            assert_eq!(after.watched_deallocations(allocation.address), Some(1));
            assert!(after
                .allocation_containing(allocation.address)
                .is_none_or(|replacement| replacement.generation != allocation.generation));
        })
        .unwrap();
        worker.join().unwrap();
    }

    fn process_system_foreign_thread_reallocation_retires_watch_before_replacement_free() {
        let (send, receive) = mpsc::sync_channel::<Vec<u64>>(0);
        let (done_send, done_receive) = mpsc::sync_channel(0);
        let worker = std::thread::spawn(move || {
            let mut values = receive.recv().unwrap();
            values.reserve_exact(1024);
            std::hint::black_box(&values);
            drop(values);
            done_send.send(()).unwrap();
        });
        measure(|| {
            let values = vec![11u64, 17];
            let before = snapshot().unwrap();
            let allocation = before
                .allocation_containing(values.as_ptr().addr())
                .unwrap();
            assert!(watch_tracked_allocation(allocation.address));
            send.send(values).unwrap();
            done_receive.recv().unwrap();
            let after = snapshot().unwrap();
            assert!(!after.overflowed);
            assert!(after.reallocations > before.reallocations);
            assert_eq!(after.watched_deallocations(allocation.address), Some(0));
        })
        .unwrap();
        worker.join().unwrap();
    }

    fn process_system_nested_scope_and_foreign_selection_cannot_reset_census() {
        measure(|| {
            let value = Box::new([23u64; 4]);
            let before = snapshot().unwrap();
            assert_eq!(measure(|| ()).map(|_| ()), None);
            let base = std::ptr::from_ref(&*value).addr();
            assert!(!watch_tracked_allocation(0));
            assert!(!watch_tracked_allocation(base + 1));
            assert!(watch_tracked_allocation(base));
            assert!(!watch_tracked_allocation(base));
            assert_eq!(snapshot().unwrap().allocations, before.allocations);
            drop(value);
            assert_eq!(snapshot().unwrap().watched_deallocations(base), Some(1));
        })
        .unwrap();
    }

    fn process_system_preexisting_free_is_counted_without_becoming_query_live_storage() {
        let value = Box::new([31u64; 4]);
        let ((), counts) = measure(|| drop(value)).unwrap();
        assert!(!counts.overflowed);
        assert_eq!(counts.allocations, 0);
        assert_eq!(counts.deallocations, 1);
        assert_eq!(counts.untracked_deallocations, 1);
        assert_eq!(counts.live_allocations(), 0);
    }

    fn process_system_real_capacity_overflow_invalidates_partial_census_until_epoch_end() {
        let mut values = Vec::with_capacity(MAX_LIVE + 1);
        let ((), counts) = measure(|| {
            for _ in 0..=MAX_LIVE {
                values.push(Box::new([41u64; 1]));
            }
            let full = snapshot().unwrap();
            assert!(full.overflowed);
            assert!(!watch_tracked_allocation(
                std::ptr::from_ref(&*values[0]).addr()
            ));
            values.clear();
            let after = snapshot().unwrap();
            assert!(after.overflowed);
            assert_eq!(after.live_allocations(), 0);
        })
        .unwrap();
        assert!(counts.overflowed);
        let ((), next) = measure(|| ()).unwrap();
        assert!(!next.overflowed);
        assert_eq!(next.allocations, 0);
    }

    fn process_system_unwind_releases_capture_and_does_not_import_old_epoch_records() {
        let panic = std::panic::catch_unwind(|| {
            measure(|| {
                let _value = Box::new([43u64; 4]);
                panic!("observer unwind control");
            });
        });
        assert!(panic.is_err());
        let ((), counts) = measure(|| ()).unwrap();
        assert!(!counts.overflowed);
        assert_eq!(counts.live_allocations(), 0);
        assert_eq!(counts.allocations, 0);
    }
}
