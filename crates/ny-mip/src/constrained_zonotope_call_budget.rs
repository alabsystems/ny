// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Call-local execution firewall for constrained-zonotope transforms.
//!
//! A caller supplies one absolute [`Instant`] deadline, a byte count for data
//! already live across the call, and a hard peak-live-byte ceiling.  The
//! tracker never consults process RSS, allocator state, or a global budget.
//! Transform-specific preflight code conservatively adds every logical buffer
//! that can overlap the caller's retained input and the unpublished output.
//! Allocator metadata, capacity rounding, and borrowed storage belong in the
//! caller-selected baseline.
//!
//! Budgeted transforms charge deterministic work items.  A deadline poll occurs
//! after at most [`CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL`] newly charged
//! items, in addition to explicit phase and publication checkpoints.  The
//! tracker is synchronous and starts no work of its own.

use std::mem::size_of;
use std::time::Instant;

/// Hard maximum number of charged transform items between deadline polls.
pub const CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL: usize = 16_384;

/// Caller-owned hard limits for one synchronous constrained-zonotope call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeCallBudget {
    deadline: Instant,
    baseline_live_bytes: usize,
    max_peak_live_bytes: usize,
}

impl ConstrainedZonotopeCallBudget {
    /// Create a call-local budget.
    ///
    /// `baseline_live_bytes` is the caller's conservative accounting for
    /// storage retained throughout the call, including borrowed inputs and
    /// weights when those bytes must fall under the same ceiling.  The
    /// transform adds its own complete logical peak before allocating output
    /// or scratch storage.
    #[must_use]
    pub const fn new(
        deadline: Instant,
        baseline_live_bytes: usize,
        max_peak_live_bytes: usize,
    ) -> Self {
        Self {
            deadline,
            baseline_live_bytes,
            max_peak_live_bytes,
        }
    }

    /// Absolute caller-owned deadline.
    #[must_use]
    pub const fn deadline(self) -> Instant {
        self.deadline
    }

    /// Caller-accounted bytes already live across the call.
    #[must_use]
    pub const fn baseline_live_bytes(self) -> usize {
        self.baseline_live_bytes
    }

    /// Hard aggregate logical peak-live-byte ceiling.
    #[must_use]
    pub const fn max_peak_live_bytes(self) -> usize {
        self.max_peak_live_bytes
    }
}

/// Typed fail-closed reason from the shared execution firewall.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ConstrainedZonotopeCallBudgetError {
    /// The absolute deadline was closed at a named checkpoint.
    #[error("constrained-zonotope deadline expired at {checkpoint}")]
    DeadlineExpired {
        /// Stable transform checkpoint.
        checkpoint: &'static str,
    },

    /// Checked work or memory arithmetic overflowed.
    #[error("constrained-zonotope budget overflow while computing {operation}")]
    ResourceOverflow {
        /// Failed checked calculation.
        operation: &'static str,
    },

    /// The complete preflighted logical peak exceeds the caller's hard cap.
    #[error(
        "constrained-zonotope peak-live-byte limit exceeded: required {required}, limit {limit}"
    )]
    PeakLiveBytesExceeded {
        /// Caller baseline plus transform-owned logical peak.
        required: usize,
        /// Caller-selected hard ceiling.
        limit: usize,
    },
}

/// Accounting receipt for admitted work, including failed attempts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeCallReport {
    peak_live_bytes: usize,
    charged_items: usize,
    deadline_polls: usize,
}

impl ConstrainedZonotopeCallReport {
    /// Preflighted aggregate logical peak, including the caller baseline.
    #[must_use]
    pub const fn peak_live_bytes(self) -> usize {
        self.peak_live_bytes
    }

    /// Deterministic work items charged by the transform.
    #[must_use]
    pub const fn charged_items(self) -> usize {
        self.charged_items
    }

    /// Deadline reads performed at admission, phase, chunk, and publication
    /// checkpoints.
    #[must_use]
    pub const fn deadline_polls(self) -> usize {
        self.deadline_polls
    }
}

/// A completed value paired with its call-local accounting receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeCallOutcome<T> {
    value: T,
    report: ConstrainedZonotopeCallReport,
}

impl<T> ConstrainedZonotopeCallOutcome<T> {
    pub(crate) const fn new(value: T, report: ConstrainedZonotopeCallReport) -> Self {
        Self { value, report }
    }

    /// Completed transform value.
    #[must_use]
    pub const fn value(&self) -> &T {
        &self.value
    }

    /// Consume the wrapper and return the completed transform value.
    #[must_use]
    pub fn into_value(self) -> T {
        self.value
    }

    /// Call-local accounting receipt.
    #[must_use]
    pub const fn report(&self) -> ConstrainedZonotopeCallReport {
        self.report
    }

    /// Consume the wrapper into its value and accounting receipt.
    #[must_use]
    pub fn into_parts(self) -> (T, ConstrainedZonotopeCallReport) {
        (self.value, self.report)
    }
}

/// A transform result paired with the accounting receipt for its full attempt.
///
/// Unlike [`ConstrainedZonotopeCallOutcome`], this wrapper is also returned
/// when admission, validation, deadline, or peak-memory checks fail.  This lets
/// callers compose optional attempts without losing the work and peak receipt
/// needed by an enclosing call budget.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstrainedZonotopeCallAttempt<T, E> {
    result: Result<T, E>,
    report: ConstrainedZonotopeCallReport,
}

impl<T, E> ConstrainedZonotopeCallAttempt<T, E> {
    pub(crate) const fn new(result: Result<T, E>, report: ConstrainedZonotopeCallReport) -> Self {
        Self { result, report }
    }

    /// Borrow the completed value or failure from this attempt.
    pub fn result(&self) -> Result<&T, &E> {
        self.result.as_ref()
    }

    /// Call-local accounting receipt, present on both success and failure.
    #[must_use]
    pub const fn report(&self) -> ConstrainedZonotopeCallReport {
        self.report
    }

    /// Consume the wrapper into its result and accounting receipt.
    pub fn into_parts(self) -> (Result<T, E>, ConstrainedZonotopeCallReport) {
        (self.result, self.report)
    }
}

/// Common guard contract used by the budgeted and genuinely inert legacy
/// paths.  The inert implementation performs no clock reads or allocations.
pub(crate) trait ConstrainedZonotopeCallGate {
    fn is_enforcing(&self) -> bool;

    fn checkpoint(
        &mut self,
        checkpoint: &'static str,
    ) -> Result<(), ConstrainedZonotopeCallBudgetError>;

    fn charge_items(
        &mut self,
        items: usize,
        checkpoint: &'static str,
    ) -> Result<(), ConstrainedZonotopeCallBudgetError>;

    fn preflight_peak_live_bytes(
        &mut self,
        transform_owned_bytes: usize,
    ) -> Result<(), ConstrainedZonotopeCallBudgetError>;

    fn report(&self) -> ConstrainedZonotopeCallReport;
}

/// No-op guard for the pre-existing APIs.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct InertConstrainedZonotopeCallGate;

impl ConstrainedZonotopeCallGate for InertConstrainedZonotopeCallGate {
    #[inline]
    fn is_enforcing(&self) -> bool {
        false
    }

    #[inline]
    fn checkpoint(
        &mut self,
        _checkpoint: &'static str,
    ) -> Result<(), ConstrainedZonotopeCallBudgetError> {
        Ok(())
    }

    #[inline]
    fn charge_items(
        &mut self,
        _items: usize,
        _checkpoint: &'static str,
    ) -> Result<(), ConstrainedZonotopeCallBudgetError> {
        Ok(())
    }

    #[inline]
    fn preflight_peak_live_bytes(
        &mut self,
        _transform_owned_bytes: usize,
    ) -> Result<(), ConstrainedZonotopeCallBudgetError> {
        Ok(())
    }

    #[inline]
    fn report(&self) -> ConstrainedZonotopeCallReport {
        ConstrainedZonotopeCallReport {
            peak_live_bytes: 0,
            charged_items: 0,
            deadline_polls: 0,
        }
    }
}

/// Synchronous call-local tracker.  The clock receives the checkpoint label so
/// unit tests can expire an exact seam without sleeping.
pub(crate) struct ConstrainedZonotopeCallTracker<N> {
    budget: ConstrainedZonotopeCallBudget,
    now: N,
    peak_live_bytes: usize,
    charged_items: usize,
    items_since_poll: usize,
    deadline_polls: usize,
}

impl ConstrainedZonotopeCallTracker<fn(&'static str) -> Instant> {
    pub(crate) fn from_system_clock(
        budget: ConstrainedZonotopeCallBudget,
    ) -> Result<Self, ConstrainedZonotopeCallBudgetError> {
        let (tracker, admission) = Self::from_system_clock_attempt(budget);
        admission.map(|()| tracker)
    }

    pub(crate) fn from_system_clock_attempt(
        budget: ConstrainedZonotopeCallBudget,
    ) -> (Self, Result<(), ConstrainedZonotopeCallBudgetError>) {
        Self::with_clock_attempt(budget, system_now)
    }
}

impl<N> ConstrainedZonotopeCallTracker<N>
where
    N: FnMut(&'static str) -> Instant,
{
    #[cfg(test)]
    pub(crate) fn with_clock(
        budget: ConstrainedZonotopeCallBudget,
        now: N,
    ) -> Result<Self, ConstrainedZonotopeCallBudgetError> {
        let (tracker, admission) = Self::with_clock_attempt(budget, now);
        admission.map(|()| tracker)
    }

    /// Construct a tracker while preserving its receipt if admission fails.
    pub(crate) fn with_clock_attempt(
        budget: ConstrainedZonotopeCallBudget,
        now: N,
    ) -> (Self, Result<(), ConstrainedZonotopeCallBudgetError>) {
        let mut tracker = Self {
            budget,
            now,
            peak_live_bytes: budget.baseline_live_bytes,
            charged_items: 0,
            items_since_poll: 0,
            deadline_polls: 0,
        };
        let admission = tracker
            .checkpoint("admission")
            .and_then(|()| tracker.preflight_peak_live_bytes(0));
        (tracker, admission)
    }

    fn poll(&mut self, checkpoint: &'static str) -> Result<(), ConstrainedZonotopeCallBudgetError> {
        self.deadline_polls = self.deadline_polls.checked_add(1).ok_or(
            ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "deadline poll count",
            },
        )?;
        if (self.now)(checkpoint) >= self.budget.deadline {
            return Err(ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint });
        }
        self.items_since_poll = 0;
        Ok(())
    }
}

impl<N> ConstrainedZonotopeCallGate for ConstrainedZonotopeCallTracker<N>
where
    N: FnMut(&'static str) -> Instant,
{
    fn is_enforcing(&self) -> bool {
        true
    }

    fn checkpoint(
        &mut self,
        checkpoint: &'static str,
    ) -> Result<(), ConstrainedZonotopeCallBudgetError> {
        self.poll(checkpoint)
    }

    fn charge_items(
        &mut self,
        mut items: usize,
        checkpoint: &'static str,
    ) -> Result<(), ConstrainedZonotopeCallBudgetError> {
        self.charged_items = self.charged_items.checked_add(items).ok_or(
            ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "charged work item count",
            },
        )?;
        while items != 0 {
            let until_poll = CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL - self.items_since_poll;
            let consumed = items.min(until_poll);
            self.items_since_poll += consumed;
            items -= consumed;
            if self.items_since_poll == CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL {
                self.poll(checkpoint)?;
            }
        }
        Ok(())
    }

    fn preflight_peak_live_bytes(
        &mut self,
        transform_owned_bytes: usize,
    ) -> Result<(), ConstrainedZonotopeCallBudgetError> {
        let required = self
            .budget
            .baseline_live_bytes
            .checked_add(transform_owned_bytes)
            .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "aggregate peak-live bytes",
            })?;
        if required > self.budget.max_peak_live_bytes {
            return Err(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required,
                limit: self.budget.max_peak_live_bytes,
            });
        }
        self.peak_live_bytes = self.peak_live_bytes.max(required);
        Ok(())
    }

    fn report(&self) -> ConstrainedZonotopeCallReport {
        ConstrainedZonotopeCallReport {
            peak_live_bytes: self.peak_live_bytes,
            charged_items: self.charged_items,
            deadline_polls: self.deadline_polls,
        }
    }
}

fn system_now(_checkpoint: &'static str) -> Instant {
    Instant::now()
}

/// Checked builder for the complete transform-owned logical peak.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ConstrainedZonotopePeakLiveBytes {
    bytes: usize,
}

impl ConstrainedZonotopePeakLiveBytes {
    pub(crate) const fn new() -> Self {
        Self { bytes: 0 }
    }

    pub(crate) fn add_bytes(
        &mut self,
        bytes: usize,
        operation: &'static str,
    ) -> Result<(), ConstrainedZonotopeCallBudgetError> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow { operation })?;
        Ok(())
    }

    pub(crate) fn add_elements<T>(
        &mut self,
        elements: usize,
        operation: &'static str,
    ) -> Result<(), ConstrainedZonotopeCallBudgetError> {
        let bytes = elements
            .checked_mul(size_of::<T>())
            .ok_or(ConstrainedZonotopeCallBudgetError::ResourceOverflow { operation })?;
        self.add_bytes(bytes, operation)
    }

    pub(crate) const fn finish(self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::time::Duration;

    use super::*;

    #[test]
    fn tracker_polls_at_the_hard_chunk_boundary() {
        let start = Instant::now();
        let reads = Cell::new(0_usize);
        let mut tracker = ConstrainedZonotopeCallTracker::with_clock(
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX),
            |_| {
                reads.set(reads.get() + 1);
                start
            },
        )
        .unwrap();
        assert_eq!(reads.get(), 1);
        tracker
            .charge_items(CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL - 1, "test work")
            .unwrap();
        assert_eq!(reads.get(), 1);
        tracker.charge_items(1, "test work").unwrap();
        assert_eq!(reads.get(), 2);
        assert_eq!(
            tracker.report().charged_items(),
            CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL
        );
    }

    #[test]
    fn memory_preflight_checks_boundary_and_overflow() {
        let start = Instant::now();
        let budget = ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 7, 18);
        let mut tracker = ConstrainedZonotopeCallTracker::with_clock(budget, |_| start).unwrap();
        tracker.preflight_peak_live_bytes(11).unwrap();
        assert_eq!(tracker.report().peak_live_bytes(), 18);
        tracker.preflight_peak_live_bytes(3).unwrap();
        assert_eq!(
            tracker.report().peak_live_bytes(),
            18,
            "nested smaller preflights must not erase the true call peak"
        );

        let mut below = ConstrainedZonotopeCallTracker::with_clock(
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 7, 17),
            |_| start,
        )
        .unwrap();
        assert_eq!(
            below.preflight_peak_live_bytes(11),
            Err(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required: 18,
                limit: 17,
            })
        );

        let mut overflow = ConstrainedZonotopeCallTracker::with_clock(
            ConstrainedZonotopeCallBudget::new(
                start + Duration::from_secs(1),
                usize::MAX,
                usize::MAX,
            ),
            |_| start,
        )
        .unwrap();
        assert!(matches!(
            overflow.preflight_peak_live_bytes(1),
            Err(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "aggregate peak-live bytes"
            })
        ));

        let mut bytes = ConstrainedZonotopePeakLiveBytes::new();
        assert!(matches!(
            bytes.add_elements::<u64>(usize::MAX, "test element bytes"),
            Err(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "test element bytes"
            })
        ));
    }

    #[test]
    fn expired_admission_fails_before_any_reportable_work() {
        let now = Instant::now();
        let (tracker, result) = ConstrainedZonotopeCallTracker::with_clock_attempt(
            ConstrainedZonotopeCallBudget::new(now, 0, 0),
            |_| now,
        );
        assert!(matches!(
            result,
            Err(ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                checkpoint: "admission"
            })
        ));
        assert_eq!(tracker.report().peak_live_bytes(), 0);
        assert_eq!(tracker.report().charged_items(), 0);
        assert_eq!(tracker.report().deadline_polls(), 1);
    }

    #[test]
    fn over_cap_baseline_fails_at_admission_before_any_transform_work() {
        let start = Instant::now();
        let reads = Cell::new(0_usize);
        let (tracker, result) = ConstrainedZonotopeCallTracker::with_clock_attempt(
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 8, 7),
            |checkpoint| {
                assert_eq!(checkpoint, "admission");
                reads.set(reads.get() + 1);
                start
            },
        );
        assert!(matches!(
            result,
            Err(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required: 8,
                limit: 7,
            })
        ));
        assert_eq!(reads.get(), 1);
        assert_eq!(tracker.report().peak_live_bytes(), 8);
        assert_eq!(tracker.report().charged_items(), 0);
        assert_eq!(tracker.report().deadline_polls(), 1);
    }

    #[test]
    fn work_and_poll_counters_overflow_fail_closed() {
        let start = Instant::now();
        let budget =
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, usize::MAX);
        let mut work = ConstrainedZonotopeCallTracker::with_clock(budget, |_| start).unwrap();
        work.charged_items = usize::MAX;
        assert_eq!(
            work.charge_items(1, "overflowing work"),
            Err(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "charged work item count",
            })
        );

        let mut polls = ConstrainedZonotopeCallTracker::with_clock(budget, |_| start).unwrap();
        polls.deadline_polls = usize::MAX;
        assert_eq!(
            polls.checkpoint("overflowing poll"),
            Err(ConstrainedZonotopeCallBudgetError::ResourceOverflow {
                operation: "deadline poll count",
            })
        );
    }
}
