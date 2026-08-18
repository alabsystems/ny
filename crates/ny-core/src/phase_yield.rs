// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #phase-yield — invariant I2: a phase's expiry is a phase event, never an
//! instance event.
//!
//! Design: `docs/DESIGN_MARGINAL_VALUE_SCHEDULER_2026-08-08.md` §2.3.
//!
//! ## The defect this exists to make unrepresentable
//!
//! Measured four times in one campaign, in four unrelated subsystems: a phase
//! whose own sub-budget expired raised `NyError::DeadlineExceeded`, and a
//! caller mapped that straight onto a whole-instance `Timeout`. On
//! `cifar100_2024` at a 330 s budget that discarded **209.9 s of live ledger
//! with `domains_explored = 0`** — the phase cap was 40 s and the BaB deadline
//! was 225.9 s.
//!
//! The reason it recurred is that `Result<T>` cannot tell the two apart. A
//! phase returns `Err(DeadlineExceeded)` and the caller has to *remember* to
//! ask "whose deadline?". Four callers did not.
//!
//! ## The fix
//!
//! [`PhaseYield`] has **no variant meaning "abort the instance"**. A phase that
//! runs out of its slice returns [`PhaseYield::Partial`] carrying whatever it
//! certified before stopping; a phase that could not usefully start returns
//! [`PhaseYield::Declined`]. Neither is an error, so neither can be `?`-ed into
//! a caller's error path by accident — which is exactly how all four sites
//! failed.
//!
//! Deciding that the *instance* is over is the scheduler's job, and it needs
//! the global deadline to make it. [`classify_expiry`] is the only place that
//! comparison happens, so it can be got right once.
//!
//! ## Adoption
//!
//! Incremental by design. [`from_result`] lifts an existing
//! `Result<T, NyError>` into a `PhaseYield` given both deadlines, so a call
//! site can be converted without touching the phase it calls.

use std::time::Instant;

/// Why a phase produced no usable output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclineReason {
    /// Predicted cost exceeded what the scheduler was willing to commit
    /// (invariant I1). Carried in whole milliseconds so the type stays `Eq`.
    Unaffordable {
        predicted_ms: u64,
        available_ms: u64,
    },
    /// The phase does not apply to this problem shape.
    Unsupported(&'static str),
    /// The phase ran but produced nothing usable.
    Empty,
}

/// The result of running a phase for a bounded slice.
///
/// Deliberately **not** a `Result`: there is no variant that means "the
/// instance is over". See the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhaseYield<T> {
    /// Finished its work within the slice.
    Complete(T),
    /// Ran out of its slice with usable, already-certified output.
    Partial(T),
    /// Produced nothing usable. Not an error.
    Declined(DeclineReason),
}

impl<T> PhaseYield<T> {
    /// The phase's output, if it produced any.
    pub fn value(self) -> Option<T> {
        match self {
            Self::Complete(v) | Self::Partial(v) => Some(v),
            Self::Declined(_) => None,
        }
    }

    pub fn as_value(&self) -> Option<&T> {
        match self {
            Self::Complete(v) | Self::Partial(v) => Some(v),
            Self::Declined(_) => None,
        }
    }

    /// Whether downstream work can proceed on this output.
    #[must_use]
    pub const fn is_usable(&self) -> bool {
        matches!(self, Self::Complete(_) | Self::Partial(_))
    }

    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(self, Self::Complete(_))
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> PhaseYield<U> {
        match self {
            Self::Complete(v) => PhaseYield::Complete(f(v)),
            Self::Partial(v) => PhaseYield::Partial(f(v)),
            Self::Declined(r) => PhaseYield::Declined(r),
        }
    }
}

/// Whose deadline expired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expiry {
    /// A phase-local budget ended while the instance still has time. The
    /// instance MUST continue.
    PhaseOnly,
    /// The instance's own deadline has passed. Only here may a caller conclude
    /// the run is over.
    Global,
}

/// Classify a deadline expiry against **both** deadlines.
///
/// This is the comparison the four defect sites did not make. Centralising it
/// means it can be got right once instead of remembered four times.
///
/// A missing `global_deadline` means "no instance deadline supplied", which is
/// deliberately treated as [`Expiry::PhaseOnly`]: absent evidence that the
/// instance is over, the safe reading is that it is not. Concluding `Global`
/// from a missing deadline is precisely how budget gets discarded.
#[must_use]
pub fn classify_expiry(
    now: Instant,
    phase_deadline: Option<Instant>,
    global_deadline: Option<Instant>,
) -> Expiry {
    match global_deadline {
        Some(global) if now >= global => Expiry::Global,
        // Either the global deadline is live, or none was supplied. In both
        // cases a phase-local expiry says nothing about the instance.
        _ => {
            let _ = phase_deadline;
            Expiry::PhaseOnly
        }
    }
}

/// Lift an existing `Result` into a [`PhaseYield`], so a call site can adopt
/// invariant I2 without changing the phase it calls.
///
/// - `Ok(v)` → `Complete(v)`.
/// - a deadline error with the **global** deadline spent → the error is
///   returned, because that genuinely is the instance ending.
/// - a deadline error with the global deadline **live** → `Declined(Empty)`,
///   never an instance abort. This is the case worth 209.9 s.
/// - any other error → returned unchanged; I2 is about deadlines only.
///
/// # Errors
/// Propagates non-deadline errors, and deadline errors that are genuinely
/// global.
pub fn from_result<T, E>(
    result: Result<T, E>,
    now: Instant,
    phase_deadline: Option<Instant>,
    global_deadline: Option<Instant>,
    is_deadline: impl FnOnce(&E) -> bool,
) -> Result<PhaseYield<T>, E> {
    match result {
        Ok(v) => Ok(PhaseYield::Complete(v)),
        Err(e) if is_deadline(&e) => match classify_expiry(now, phase_deadline, global_deadline) {
            Expiry::Global => Err(e),
            Expiry::PhaseOnly => Ok(PhaseYield::Declined(DeclineReason::Empty)),
        },
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn t(offset_ms: i64) -> Instant {
        let base = Instant::now();
        if offset_ms >= 0 {
            base + Duration::from_millis(offset_ms as u64)
        } else {
            base.checked_sub(Duration::from_millis((-offset_ms) as u64))
                .expect("test offsets stay inside the process uptime")
        }
    }

    #[test]
    fn phase_expiry_with_a_live_global_deadline_is_phase_only() {
        // THE case worth 209.9s: the 40s phase cap is spent, the 225.9s BaB
        // deadline is not.
        let now = Instant::now();
        assert_eq!(
            classify_expiry(now, Some(t(-1000)), Some(t(180_000))),
            Expiry::PhaseOnly
        );
    }

    #[test]
    fn only_a_spent_global_deadline_is_global() {
        let now = Instant::now();
        assert_eq!(
            classify_expiry(now, Some(t(-1000)), Some(t(-1))),
            Expiry::Global
        );
    }

    #[test]
    fn a_missing_global_deadline_is_never_treated_as_exhaustion() {
        // Absent evidence that the instance is over, the safe reading is that
        // it is not. Concluding Global here is how budget gets discarded.
        let now = Instant::now();
        assert_eq!(
            classify_expiry(now, Some(t(-5000)), None),
            Expiry::PhaseOnly
        );
        assert_eq!(classify_expiry(now, None, None), Expiry::PhaseOnly);
    }

    #[test]
    fn a_live_phase_deadline_does_not_make_it_global() {
        let now = Instant::now();
        assert_eq!(
            classify_expiry(now, Some(t(10_000)), Some(t(20_000))),
            Expiry::PhaseOnly
        );
    }

    #[test]
    fn partial_output_is_usable_and_is_not_complete() {
        let y = PhaseYield::Partial(7u32);
        assert!(
            y.is_usable(),
            "a phase that ran out of slice still has output"
        );
        assert!(!y.is_complete());
        assert_eq!(y.value(), Some(7));
    }

    #[test]
    fn declined_carries_no_value_and_is_not_usable() {
        let y: PhaseYield<u32> = PhaseYield::Declined(DeclineReason::Unaffordable {
            predicted_ms: 82_000,
            available_ms: 40_000,
        });
        assert!(!y.is_usable());
        assert_eq!(y.value(), None);
    }

    #[test]
    fn from_result_turns_a_phase_expiry_into_declined_not_an_error() {
        // The regression this whole type exists for: a phase-local deadline
        // error must NOT become the caller's error.
        let r: Result<u32, &str> = Err("deadline exceeded");
        let y = from_result(r, Instant::now(), Some(t(-1000)), Some(t(180_000)), |e| {
            e.contains("deadline")
        })
        .expect("a phase expiry must not surface as an error");
        assert_eq!(y, PhaseYield::Declined(DeclineReason::Empty));
    }

    #[test]
    fn from_result_still_propagates_a_genuinely_global_expiry() {
        let r: Result<u32, &str> = Err("deadline exceeded");
        let out = from_result(r, Instant::now(), Some(t(-1000)), Some(t(-1)), |e| {
            e.contains("deadline")
        });
        assert!(out.is_err(), "a spent instance deadline IS the run ending");
    }

    #[test]
    fn from_result_propagates_non_deadline_errors_unchanged() {
        // I2 is about deadlines. A real failure must still be a failure.
        let r: Result<u32, &str> = Err("shape mismatch");
        let out = from_result(r, Instant::now(), Some(t(-1000)), Some(t(180_000)), |e| {
            e.contains("deadline")
        });
        assert!(out.is_err(), "a non-deadline error must not be swallowed");
    }

    #[test]
    fn from_result_passes_success_through_as_complete() {
        let r: Result<u32, &str> = Ok(3);
        let y = from_result(r, Instant::now(), None, None, |_| false).unwrap();
        assert_eq!(y, PhaseYield::Complete(3));
    }

    #[test]
    fn map_preserves_the_variant() {
        assert_eq!(
            PhaseYield::Partial(2u8).map(|v| v * 2),
            PhaseYield::Partial(4)
        );
        assert_eq!(
            PhaseYield::Complete(2u8).map(|v| v * 2),
            PhaseYield::Complete(4)
        );
        assert!(matches!(
            PhaseYield::<u8>::Declined(DeclineReason::Empty).map(|v| v * 2),
            PhaseYield::Declined(DeclineReason::Empty)
        ));
    }
}
