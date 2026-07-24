// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Wall-clock budget for certificate construction.
//!
//! Exact-rational deep-CROWN certification is post-verdict work: the verifier
//! has already decided `Verified`, and the certificate is an optional sidecar
//! artifact. On adversarial instances the exact arithmetic can accumulate
//! rationals of enormous magnitude (observed: acasxu max-diff margins driving
//! million-bit numerators), so an unbounded certification pass can stall a CLI
//! run long past every verification deadline. This module gives callers a way
//! to bound that work: install a deadline for the current thread, and the
//! certificate builders poll it at loop boundaries, failing OPEN to
//! "no certificate" (`DeepCrownError::BudgetExceeded`) — never touching the
//! verdict.
//!
//! Thread-local by design, mirroring the rational interning arena
//! ([`crate::rational`]): certificate construction is single-threaded per
//! problem, and a thread-local avoids threading a deadline parameter through
//! the strict-verifier-annotated public `certify*` signatures. All operations
//! here are TOTAL: no panics, no unwraps; an absent deadline simply means
//! "unbounded" (the previous behaviour).
//!
//! LIMITATION (by the same thread-local design): a deadline installed on one
//! thread does NOT propagate to worker threads a caller spawns (e.g. the
//! `certify_onnx` bin's leaf workers) — those run unbounded, exactly as
//! before this module existed. A caller that wants bounded workers must
//! install a guard on each worker thread.

use std::cell::Cell;
use std::time::Instant;

thread_local! {
    static DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// RAII installer for a certificate-construction deadline on the current
/// thread. Restores the previously-installed deadline (usually `None`) on
/// drop, so nested or sequential installations compose and a stale deadline
/// can never leak into unrelated later work on the same thread.
#[must_use = "the deadline is uninstalled when the guard drops"]
pub struct DeadlineGuard {
    prev: Option<Instant>,
}

impl DeadlineGuard {
    /// Install `deadline` as the current thread's certificate budget.
    pub fn install(deadline: Instant) -> Self {
        let prev = DEADLINE.with(|d| d.replace(Some(deadline)));
        DeadlineGuard { prev }
    }
}

impl Drop for DeadlineGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        DEADLINE.with(|d| d.set(prev));
    }
}

/// `true` iff a deadline is installed and has passed. TOTAL: `None` (no
/// budget) is simply `false`, preserving unbounded behaviour for callers that
/// never install a guard.
pub(crate) fn expired() -> bool {
    DEADLINE.with(|d| match d.get() {
        Some(deadline) => Instant::now() >= deadline,
        None => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn no_guard_never_expires() {
        assert!(!expired());
    }

    /// An `Instant` strictly in the past. `checked_sub` (not `-`): the bare
    /// subtraction trips `clippy::unchecked_time_subtraction`, and on a
    /// hypothetical platform where the clock's epoch is closer than the
    /// offset the fallback still yields a passed instant (`now` itself is
    /// `<=` any deadline taken before the assert runs).
    fn past_instant() -> Instant {
        Instant::now()
            .checked_sub(Duration::from_secs(1))
            .unwrap_or_else(Instant::now)
    }

    #[test]
    fn expired_deadline_reports_and_uninstalls_on_drop() {
        {
            let _g = DeadlineGuard::install(past_instant());
            assert!(expired());
        }
        assert!(!expired());
    }

    #[test]
    fn future_deadline_does_not_expire() {
        let _g = DeadlineGuard::install(Instant::now() + Duration::from_hours(1));
        assert!(!expired());
    }

    #[test]
    fn nested_guards_restore_the_outer_deadline() {
        let _outer = DeadlineGuard::install(past_instant());
        assert!(expired());
        {
            let _inner = DeadlineGuard::install(Instant::now() + Duration::from_hours(1));
            assert!(!expired());
        }
        assert!(expired());
    }
}
