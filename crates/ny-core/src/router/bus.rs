// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The WRITE-ONLY progress bus (ROUTER_PROGRAM §2.2, §6.3).
//!
//! Lanes push counters at their natural unit boundary; nothing a lane does
//! ever reads the bus back, so a sample cannot move a bound or a verdict. The
//! router session drains it after publication (retrospective checkpoints).
//!
//! Contract, in order of importance:
//!
//! * **Dark by default.** [`ENABLED`] starts false; [`emit`] is one relaxed
//!   atomic load and a return until a router session in `shadow`/`on` mode
//!   calls [`enable`]. With the router off, no tap site does any work.
//! * **Writers never block.** A push takes the ring's lock with `try_lock`; a
//!   contended push is DROPPED and counted, never waited on. A lane thread can
//!   therefore not be delayed by a reader holding the lock.
//! * **No allocation on the hot path.** [`enable`] reserves the ring once; a
//!   push writes into reserved capacity or overwrites the oldest sample.
//! * **Cannot panic.** Indexing is checked (`get_mut`), arithmetic is
//!   saturating or modular by a non-zero length, and a poisoned lock is a
//!   dropped sample, not a second failure.
//!
//! The bus stamps each sample with its own clock (seconds since the anchor
//! handed to [`enable`], i.e. the command's start), so every sample of a row
//! lives on one clock whatever lane produced it.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::decl::CapId;

/// Ring capacity, in samples. Reserved once by [`enable`] (~0.8 MB); never
/// reserved when the router is off.
pub const CAPACITY: usize = 8192;

/// Whether taps record anything. False until a router session enables it.
static ENABLED: AtomicBool = AtomicBool::new(false);
static PUSHED: AtomicU64 = AtomicU64::new(0);
static DROPPED: AtomicU64 = AtomicU64::new(0);
static OVERWRITTEN: AtomicU64 = AtomicU64::new(0);
static RING: Mutex<Ring> = Mutex::new(Ring::EMPTY);

struct Ring {
    epoch: Option<Instant>,
    buf: Vec<ProgressSample>,
    head: usize,
}

impl Ring {
    const EMPTY: Self = Self {
        epoch: None,
        buf: Vec::new(),
        head: 0,
    };

    /// Write into reserved capacity, or overwrite the oldest sample. Returns
    /// false when there is no capacity at all (the bus was never enabled).
    fn push(&mut self, sample: ProgressSample) -> bool {
        if self.buf.len() < self.buf.capacity() {
            self.buf.push(sample);
            return true;
        }
        let len = self.buf.len();
        match self.buf.get_mut(self.head) {
            Some(slot) if len > 0 => {
                *slot = sample;
                self.head = (self.head + 1) % len;
                OVERWRITTEN.fetch_add(1, Ordering::Relaxed);
                true
            }
            _ => false,
        }
    }
}

/// One progress sample: counters only, all integers or f64 bits.
///
/// Counters are CUMULATIVE within one tree/phase; a counter that goes
/// backwards marks a new tree (a new class, a new frame), and the projections
/// only ever read the trailing monotone run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ProgressSample {
    /// The capability that produced it.
    pub cap: CapId,
    /// The unit boundary (expansion count, level, wave, iteration, class).
    pub unit_index: u64,
    /// Cumulative work in the capability's own units.
    pub work_done: u64,
    /// Cumulative closes (certified leaves / prunes).
    pub closes: u64,
    /// Open frontier / queue length at the sample.
    pub open: u64,
    /// Cumulative nodes created.
    pub created: u64,
    /// Level / depth, where meaningful.
    pub level: u64,
    /// Cumulative pruned nodes (BFS).
    pub pruned: u64,
    /// The kind's scalar progress value as `f64` bits (best margin, certified
    /// volume fraction, per-iteration gain).
    pub margin_best_bits: u64,
    /// Seconds since the bus anchor, as `f64` bits (stamped by the bus).
    pub secs_bits: u64,
}

impl ProgressSample {
    /// A binary/k-ary frontier sample (margin-row expansions, graph BaB, waves).
    #[must_use]
    pub const fn frontier(cap: CapId, work: u64, created: u64, closes: u64, open: u64) -> Self {
        Self {
            cap,
            unit_index: work,
            work_done: work,
            closes,
            open,
            created,
            level: 0,
            pruned: 0,
            margin_best_bits: 0,
            secs_bits: 0,
        }
    }

    /// A BFS level sample: cumulative evaluated nodes (root included) and
    /// cumulative prunes after the level at `depth`.
    #[must_use]
    pub const fn bfs(cap: CapId, depth: u64, evaluated: u64, pruned: u64) -> Self {
        Self {
            cap,
            unit_index: depth,
            work_done: evaluated,
            closes: pruned,
            open: 0,
            created: 0,
            level: depth,
            pruned,
            margin_best_bits: 0,
            secs_bits: 0,
        }
    }

    /// A scalar-progress sample (best margin, volume fraction, per-iteration
    /// gain) at unit `index`.
    #[must_use]
    pub fn scalar(cap: CapId, index: u64, value: f64) -> Self {
        Self {
            cap,
            unit_index: index,
            work_done: index,
            closes: 0,
            open: 0,
            created: 0,
            level: 0,
            pruned: 0,
            margin_best_bits: value.to_bits(),
            secs_bits: 0,
        }
    }

    /// The same sample with an explicit timestamp (tests, replay).
    #[must_use]
    pub fn at_secs(mut self, secs: f64) -> Self {
        self.secs_bits = secs.to_bits();
        self
    }

    /// Seconds since the bus anchor.
    #[must_use]
    pub fn secs(&self) -> f64 {
        f64::from_bits(self.secs_bits)
    }

    /// The scalar progress value.
    #[must_use]
    pub fn margin_best(&self) -> f64 {
        f64::from_bits(self.margin_best_bits)
    }

    /// The owned, serializable form for flight records and replay.
    #[must_use]
    pub fn record(&self) -> SampleRecord {
        SampleRecord {
            cap: self.cap.as_str().to_owned(),
            unit_index: self.unit_index,
            work_done: self.work_done,
            closes: self.closes,
            open: self.open,
            created: self.created,
            level: self.level,
            pruned: self.pruned,
            margin_best_bits: self.margin_best_bits,
            secs_bits: self.secs_bits,
        }
    }
}

/// The serializable twin of [`ProgressSample`] (the capability as an owned
/// string). Replay maps `cap` back to the binary's own [`CapId`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SampleRecord {
    /// Capability identity.
    pub cap: String,
    /// See [`ProgressSample::unit_index`].
    pub unit_index: u64,
    /// See [`ProgressSample::work_done`].
    pub work_done: u64,
    /// See [`ProgressSample::closes`].
    pub closes: u64,
    /// See [`ProgressSample::open`].
    pub open: u64,
    /// See [`ProgressSample::created`].
    pub created: u64,
    /// See [`ProgressSample::level`].
    pub level: u64,
    /// See [`ProgressSample::pruned`].
    pub pruned: u64,
    /// See [`ProgressSample::margin_best_bits`].
    pub margin_best_bits: u64,
    /// See [`ProgressSample::secs_bits`].
    pub secs_bits: u64,
}

impl SampleRecord {
    /// Back to a sample, with `cap` resolved to a static identity by the
    /// caller (`None` = unknown capability, i.e. a replay divergence).
    #[must_use]
    pub fn to_sample(&self, cap: Option<CapId>) -> Option<ProgressSample> {
        Some(ProgressSample {
            cap: cap?,
            unit_index: self.unit_index,
            work_done: self.work_done,
            closes: self.closes,
            open: self.open,
            created: self.created,
            level: self.level,
            pruned: self.pruned,
            margin_best_bits: self.margin_best_bits,
            secs_bits: self.secs_bits,
        })
    }
}

/// Bus counters for the `[router] done` line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BusStats {
    /// Samples written.
    pub pushed: u64,
    /// Samples dropped because a writer found the ring locked.
    pub dropped: u64,
    /// Oldest samples overwritten because the ring was full.
    pub overwritten: u64,
}

/// Whether taps record anything. One relaxed load: the cost of a dark tap.
#[inline]
#[must_use]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Push one sample. A no-op when the bus is disabled; never blocks, never
/// allocates, never panics. The bus stamps `secs_bits` with its own clock.
#[inline]
pub fn emit(sample: ProgressSample) {
    if !enabled() {
        return;
    }
    push(sample);
}

fn push(mut sample: ProgressSample) {
    let Ok(mut ring) = RING.try_lock() else {
        // WouldBlock or Poisoned: either way, drop it rather than wait.
        DROPPED.fetch_add(1, Ordering::Relaxed);
        return;
    };
    let secs = ring
        .epoch
        .map_or(0.0, |epoch| epoch.elapsed().as_secs_f64());
    sample.secs_bits = secs.to_bits();
    if ring.push(sample) {
        PUSHED.fetch_add(1, Ordering::Relaxed);
    } else {
        DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Enable the bus, anchored at `anchor` (the command's start). Reserves the
/// ring once. Called only by a router session whose mode is not `off`.
pub fn enable(anchor: Instant) {
    {
        let mut ring = RING.lock().unwrap_or_else(PoisonError::into_inner);
        ring.epoch = Some(anchor);
        if ring.buf.capacity() < CAPACITY {
            ring.buf = Vec::with_capacity(CAPACITY);
            ring.head = 0;
        }
    }
    ENABLED.store(true, Ordering::Release);
}

/// Stop recording (a fault, or the session finished).
pub fn disable() {
    ENABLED.store(false, Ordering::Release);
}

/// Take every buffered sample, oldest first, and empty the ring. Called by the
/// router session (never by a lane); it may wait for the lock.
#[must_use]
pub fn drain() -> Vec<ProgressSample> {
    let mut ring = RING.lock().unwrap_or_else(PoisonError::into_inner);
    let head = ring.head.min(ring.buf.len());
    let mut out = Vec::with_capacity(ring.buf.len());
    if let (Some(newer), Some(older)) = (ring.buf.get(head..), ring.buf.get(..head)) {
        out.extend_from_slice(newer);
        out.extend_from_slice(older);
    }
    ring.buf.clear();
    ring.head = 0;
    out
}

/// Counters since the last [`reset`].
#[must_use]
pub fn stats() -> BusStats {
    BusStats {
        pushed: PUSHED.load(Ordering::Relaxed),
        dropped: DROPPED.load(Ordering::Relaxed),
        overwritten: OVERWRITTEN.load(Ordering::Relaxed),
    }
}

/// Disable, empty, and zero the counters (process reuse and tests).
pub fn reset() {
    disable();
    {
        let mut ring = RING.lock().unwrap_or_else(PoisonError::into_inner);
        ring.buf.clear();
        ring.head = 0;
        ring.epoch = None;
    }
    PUSHED.store(0, Ordering::Relaxed);
    DROPPED.store(0, Ordering::Relaxed);
    OVERWRITTEN.store(0, Ordering::Relaxed);
}

/// Run `f` while holding the ring's lock: the contention seam for the
/// writers-never-block test.
#[cfg(test)]
pub(crate) fn with_ring_locked<R>(f: impl FnOnce() -> R) -> R {
    let _guard = RING.lock().unwrap_or_else(PoisonError::into_inner);
    f()
}

/// A per-site emission cadence in work units, so a tap called every step
/// records one sample per `stride` units rather than one per call.
///
/// Stateless with respect to the tree: a work counter that restarts below the
/// last emission (a new class, a new frame) is due immediately.
#[derive(Debug)]
pub struct Stride {
    next: AtomicU64,
}

impl Stride {
    /// A cadence whose first call is due.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next: AtomicU64::new(0),
        }
    }

    /// Whether a sample at `work` is due, advancing the cadence when it is.
    /// Always false while the bus is disabled (a dark tap does no work).
    pub fn due(&self, work: u64, stride: u64) -> bool {
        if !enabled() {
            return false;
        }
        let step = stride.max(1);
        let next = self.next.load(Ordering::Relaxed);
        let restarted = work.saturating_add(step) < next;
        if work >= next || restarted {
            self.next
                .store(work.saturating_add(step), Ordering::Relaxed);
            true
        } else {
            false
        }
    }
}

impl Default for Stride {
    fn default() -> Self {
        Self::new()
    }
}
