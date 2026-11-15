// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #poll-amortise — one shared amortiser for deadline polls in inner loops.
//!
//! # Why this exists
//!
//! A deadline poll costs wall and changes no value. The campaign's own
//! cooperative-deadline work put a poll in the innermost loop of the certified
//! reduction and replaced several bulk copies with per-element `push` + `poll`
//! pairs, which is sound but pays the poll's cost per MAC and blocks the
//! memcpy. This module keeps the poll, and charges it once per bounded run of
//! work instead of once per unit.
//!
//! # The safety contract
//!
//! Amortising may only make a poll notice expiry LATER, never skip it, and the
//! delay must be bounded by a compile-time constant in WORK units — never in
//! iterations, nodes or layers, whose cost is unbounded. [`POLL_CHUNK_UNITS`]
//! is that constant, and it is deliberately the value the tree already uses
//! everywhere else for exactly this purpose
//! (`PatchesMaterializationDeadline::CHECK_STRIDE`, `POLL_ELEMENTS`,
//! `TIGHTEN_POLL_STRIDE`, `GRAPH_IBP_POLL_ELEMENTS`): 4096 scalar float
//! operations, microseconds of work.
//!
//! Every amortised site keeps its unstrided entry checkpoint, so a function
//! entered with the deadline already past still fails fast; only the inner
//! loop is amortised. No amortised poll may move a bound: a refusal still
//! discards the whole private computation, exactly as the per-unit form did.
//!
//! # The observable
//!
//! The counters below are maintained UNCONDITIONALLY — no feature gate, no
//! env read, no preset key. [`poll_amortisation_fragment`] renders them, and
//! `network::crown_memory::amortisation_line` folds them into the one
//! `[poll-amortise]` line the binary prints at every terminal verdict. A null
//! measurement can therefore be told from a dark one by reading the line:
//! `poll_calls=0` is dark, `poll_factor=1.00` is armed-and-ineffective.

use std::sync::atomic::{AtomicU64, Ordering};

use ny_core::Result;

/// Work units between deadline polls in an amortised inner loop.
///
/// One unit is one scalar float operation (one MAC in a certified reduction,
/// one element in a copy). 4096 of them is microseconds, which is far inside
/// the tightest slack any lane has: the margin-row lane's cooperative deadline
/// IS its external cap, so its budget for added overshoot is tens of
/// milliseconds, and the graph BaB lanes hold a 2 s scored-wall stop reserve.
pub(crate) const POLL_CHUNK_UNITS: usize = 4096;

/// Total work units charged through this module.
static POLL_UNITS: AtomicU64 = AtomicU64::new(0);
/// Number of actual poll invocations those units were charged in.
static POLL_CALLS: AtomicU64 = AtomicU64::new(0);

#[inline]
fn record(units: usize) {
    POLL_CALLS.fetch_add(1, Ordering::Relaxed);
    POLL_UNITS.fetch_add(u64::try_from(units).unwrap_or(u64::MAX), Ordering::Relaxed);
}

/// A pollable inner loop's view of its deadline.
///
/// Two implementations only: [`ChunkedPoll`], which charges units and delivers
/// them at the stride, and [`NoPoll`], which compiles to nothing. The second
/// matters: the certified reduction has ONE kernel, reached both with and
/// without a deadline, and routing the no-deadline callers through a counting
/// adapter would put a counter and an atomic in the hot loop of every
/// no-deadline concretization in the tree.
pub(crate) trait ReductionPoll {
    /// Charge one unit of work, polling at the stride boundary.
    fn tick(&mut self) -> Result<()>;
    /// Deliver every charged-but-undelivered unit, before publication.
    fn flush(&mut self) -> Result<()>;
}

/// The poll of a caller that has no deadline. Zero cost by construction.
pub(crate) struct NoPoll;

impl ReductionPoll for NoPoll {
    #[inline]
    fn tick(&mut self) -> Result<()> {
        Ok(())
    }

    #[inline]
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Amortising adapter around a per-unit deadline poll.
///
/// `tick` charges one unit and calls the wrapped poll at most once per
/// [`POLL_CHUNK_UNITS`]; `flush` delivers the remainder. The total number of
/// units delivered to the wrapped poll is identical to the per-unit form, so a
/// downstream strided deadline (`PatchesMaterializationDeadline::work`) sees
/// the same accounting.
pub(crate) struct ChunkedPoll<P> {
    poll: P,
    pending: usize,
}

impl<P> ChunkedPoll<P> {
    pub(crate) const fn new(poll: P) -> Self {
        Self { poll, pending: 0 }
    }
}

impl<P> ReductionPoll for ChunkedPoll<P>
where
    P: FnMut(usize) -> Result<()>,
{
    #[inline]
    fn tick(&mut self) -> Result<()> {
        self.pending = self.pending.saturating_add(1);
        if self.pending >= POLL_CHUNK_UNITS {
            self.flush()
        } else {
            Ok(())
        }
    }

    /// Callers MUST flush before publishing a value, so a refusal still lands
    /// before the result crosses a proof boundary.
    #[inline]
    fn flush(&mut self) -> Result<()> {
        if self.pending == 0 {
            return Ok(());
        }
        let units = self.pending;
        self.pending = 0;
        record(units);
        (self.poll)(units)
    }
}

/// Copy `source` into `destination` in [`POLL_CHUNK_UNITS`]-sized memcpy runs,
/// polling once per run.
///
/// This restores `extend_from_slice` — i.e. a real memcpy — to copies that had
/// been rewritten as per-element `push` + poll pairs. On refusal `destination`
/// holds a partial prefix; every caller discards it, which is the same
/// contract the per-element form had.
pub(crate) fn extend_polled<T, P>(destination: &mut Vec<T>, source: &[T], mut poll: P) -> Result<()>
where
    T: Copy,
    P: FnMut(usize) -> Result<()>,
{
    for chunk in source.chunks(POLL_CHUNK_UNITS) {
        destination.extend_from_slice(chunk);
        record(chunk.len());
        poll(chunk.len())?;
    }
    Ok(())
}

/// Cooperative work for loops whose iterations have different fixed costs.
/// `before` declares a bounded batch BEFORE touching its inputs or outputs.
/// Entry and publication use `checkpoint`, including when no work was done.
/// This is a polling adapter, not a new time, memory or work allowance.
pub(crate) trait CooperativeWork {
    fn checkpoint(&mut self) -> Result<()>;
    fn before(&mut self, units: usize) -> Result<()>;
}

fn check_work_batch(units: usize) -> Result<()> {
    if units > POLL_CHUNK_UNITS {
        return Err(ny_core::NyError::InvalidSpec(
            "cooperative work batch exceeds polling quantum".into(),
        ));
    }
    Ok(())
}

impl CooperativeWork for NoPoll {
    #[inline]
    fn checkpoint(&mut self) -> Result<()> {
        Ok(())
    }

    #[inline]
    fn before(&mut self, units: usize) -> Result<()> {
        check_work_batch(units)
    }
}

/// Poll before the next batch would exceed 4096 declared scalar work units.
/// Unlike an after-iteration counter, ragged batches cannot overshoot the
/// quantum. The callback receives every declared unit exactly once on success.
/// It must observe the caller's existing authority; wrapping another strided
/// poll would multiply the bound. Refusal requires discarding private output.
pub(crate) struct WeightedPoll<P> {
    poll: P,
    pending: usize,
}

impl<P> WeightedPoll<P> {
    pub(crate) const fn new(poll: P) -> Self {
        Self { poll, pending: 0 }
    }
}

impl<P: FnMut(usize) -> Result<()>> CooperativeWork for WeightedPoll<P> {
    fn checkpoint(&mut self) -> Result<()> {
        let units = std::mem::take(&mut self.pending);
        if units != 0 {
            record(units);
        }
        (self.poll)(units)
    }

    #[inline]
    fn before(&mut self, units: usize) -> Result<()> {
        check_work_batch(units)?;
        if self.pending > POLL_CHUNK_UNITS - units {
            self.checkpoint()?;
        }
        self.pending += units;
        Ok(())
    }
}

#[cfg(test)]
#[path = "amortise/cooperative_tests.rs"]
mod cooperative_tests;

/// The poll half of the `[poll-amortise]` observable.
#[must_use]
pub fn poll_amortisation_fragment() -> String {
    let units = POLL_UNITS.load(Ordering::Relaxed);
    let calls = POLL_CALLS.load(Ordering::Relaxed);
    let hundredths = units.saturating_mul(100) / calls.max(1);
    format!(
        "poll_units={units} poll_calls={calls} poll_factor={}.{:02} poll_chunk={POLL_CHUNK_UNITS}",
        hundredths / 100,
        hundredths % 100
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_poll_delivers_every_unit_at_the_declared_stride() {
        let mut calls = 0usize;
        let mut units = 0usize;
        {
            let mut poll: ChunkedPoll<_> = ChunkedPoll::new(|charged: usize| {
                calls += 1;
                units += charged;
                Ok(())
            });
            for _ in 0..(POLL_CHUNK_UNITS * 2 + 7) {
                poll.tick().expect("no refusal");
            }
            poll.flush().expect("no refusal");
        }
        assert_eq!(
            units,
            POLL_CHUNK_UNITS * 2 + 7,
            "every unit must be charged"
        );
        assert_eq!(calls, 3, "two full strides plus one remainder flush");
    }

    #[test]
    fn extend_polled_copies_bitwise_and_polls_once_per_stride() {
        let source: Vec<f32> = (0..(POLL_CHUNK_UNITS + 5))
            .map(|index| f32::from(u8::try_from(index % 251).unwrap_or(0)))
            .collect();
        let mut destination = Vec::new();
        let mut calls = 0usize;
        extend_polled(&mut destination, &source, |_| {
            calls += 1;
            Ok(())
        })
        .expect("no refusal");
        assert_eq!(calls, 2, "one full stride plus the remainder");
        assert_eq!(destination.len(), source.len());
        assert!(
            destination
                .iter()
                .zip(&source)
                .all(|(copied, original)| copied.to_bits() == original.to_bits()),
            "an amortised copy must be bit-identical to the element-wise one"
        );
    }

    #[test]
    fn poll_amortisation_fragment_is_unconditional() {
        let line = poll_amortisation_fragment();
        assert!(line.contains("poll_units="), "{line}");
        assert!(line.contains("poll_calls="), "{line}");
        assert!(line.contains("poll_factor="), "{line}");
        assert!(
            line.contains(&format!("poll_chunk={POLL_CHUNK_UNITS}")),
            "{line}"
        );
    }
}
