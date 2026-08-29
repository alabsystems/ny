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
//! here are TOTAL: no panics, no panicking `unwrap`/`expect`; an absent
//! deadline simply means "unbounded" (the previous behaviour).
//!
//! Totality is why every access goes through [`std::thread::LocalKey::try_with`]
//! and never `with`: `with` PANICS (`panic_access_error`) when the thread-local
//! is touched during its own destruction, a panic boundary the strict verifier
//! cannot discharge (it is deliberately absent from the trusted-total callee
//! set). `try_with` returns `Ok` in exactly the cases `with` would have
//! returned, so the live-TLS path is bit-identical; its `Err` arm — reached only
//! from a thread destructor, and in fact unreachable here, since a const-init
//! `Cell<Option<Instant>>` needs no destructor and so is never torn down — is
//! answered CONSERVATIVELY everywhere: an unreadable budget reads as EXHAUSTED,
//! never as "unlimited". See `expired` for why that direction is the safe one.
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
    /// What [`Drop`] must put back, and whether there is anything to put back
    /// at all.
    ///
    /// * `Some(prev)` — the deadline WAS installed, displacing `prev` (usually
    ///   `None`); drop restores `prev`.
    /// * `None` — the thread-local was already destroyed, so nothing was
    ///   installed and drop must restore NOTHING. A guard that never took
    ///   effect must not be able to CLEAR a bound it did not set; that is the
    ///   only direction in which a wrong restore would be unsafe (it would
    ///   hand back an unbounded budget), so it is modelled explicitly rather
    ///   than left to depend on the fact that a destroyed thread-local can
    ///   never come back to life.
    ///
    /// The nested `Option` is deliberate (clippy's own suggested remedy for the
    /// 3-case shape is a custom enum, which would be heavier than this one
    /// private field warrants).
    #[allow(clippy::option_option)]
    prev: Option<Option<Instant>>,
}

impl DeadlineGuard {
    /// Install `deadline` as the current thread's certificate budget.
    // Explicit match, not `.ok()`: `Result::ok` is a hardened-profile
    // `error_discard` needle, so writing it here would mint a fresh
    // discarded-error verification obligation for a `Result` this fn is
    // deliberately CONSUMING (both arms are handled below), not discarding —
    // the same convention as `alethe_bridge::parse_arg_rat`.
    #[allow(clippy::manual_ok_err)]
    pub fn install(deadline: Instant) -> Self {
        // `try_with`, not `with`: `with` panics (`panic_access_error`) when the
        // thread-local is accessed during its own destruction — see the module
        // docs. `Ok` carries the displaced deadline, exactly what `with` returned.
        //
        // On `Err` the thread-local is gone, so NOTHING was installed. Recording
        // `None` makes this guard inert: it restores nothing on drop. That is the
        // conservative arm — a failed install leaves [`expired`] answering `true`
        // (see its docs), i.e. the request for a bound degrades to the TIGHTEST
        // bound rather than silently to "unbounded".
        let prev = match DEADLINE.try_with(|d| d.replace(Some(deadline))) {
            Ok(prev) => Some(prev),
            Err(_) => None,
        };
        DeadlineGuard { prev }
    }
}

impl Drop for DeadlineGuard {
    fn drop(&mut self) {
        // Restore only what this guard actually displaced (see the field docs);
        // a guard whose install found the thread-local already destroyed has
        // nothing to give back.
        if let Some(prev) = self.prev {
            // `try_with` for the totality reason in the module docs. The `Err`
            // (thread-local already destroyed) arm deliberately does NOTHING, and
            // that cannot loosen the budget: once `DEADLINE` is unwritable it is
            // also unreadable, so `expired` answers `true` from that instant on
            // — the budget LATCHES exhausted. A dropped restore can therefore
            // only ever make the bound tighter, never leak an unbounded one.
            let _ = DEADLINE.try_with(|d| d.set(prev));
        }
    }
}

/// `true` iff a deadline is installed and has passed. TOTAL: `None` (no
/// budget) is simply `false`, preserving unbounded behaviour for callers that
/// never install a guard.
///
/// `try_with` is TOTAL (returns `Result`, never panics), unlike `with`, which
/// panics (`panic_access_error`) if the thread-local is accessed during its own
/// destruction — a panic boundary the strict verifier cannot discharge.
///
/// The `Err` (thread-local already destroyed) arm answers `true`, i.e. BUDGET
/// EXHAUSTED. That is the SEMANTICALLY correct fail-safe answer, not merely the
/// one that verifies. The question this fn answers is "may certificate work
/// continue?", and the two possible errors are not symmetric:
///
/// * Answering `false` (unlimited) would claim there is budget left at the one
///   moment we provably cannot see the budget. Work would then proceed
///   UNBOUNDED on a thread that is being torn down — and torn down alongside
///   the sibling rational arena, so every `Rat` operation would be taking its
///   own `Err`/`None` arm and substituting the `0/1` ZERO value (see
///   [`crate::rational::poisoned`]). That is precisely the unbounded-stall
///   condition this module exists to prevent, compounded by garbage inputs.
/// * Answering `true` (exhausted) merely STOPS optional work: the sole consumer
///   turns it into `DeepCrownError::BudgetExceeded`, which fails OPEN to "no
///   certificate" and never touches the already-decided verdict.
///
/// So the fail-safe answer to "has the budget run out?" when the budget cannot
/// be read is YES. The live-TLS path is unchanged: with no guard installed
/// `expired()` is still `false`.
pub(crate) fn expired() -> bool {
    DEADLINE
        .try_with(|d| match d.get() {
            Some(deadline) => Instant::now() >= deadline,
            None => false,
        })
        .unwrap_or(true)
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
