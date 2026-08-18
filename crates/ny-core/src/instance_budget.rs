// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #instance-budget — the one deadline that means "the run is over".
//!
//! Invariant I1 ([`crate::phase_window`]) says a phase window must derive from
//! predicted cost and a *fraction of what remains*. That requires knowing what
//! remains — and the measured defect is that deep gates **do not**.
//!
//! The forward-linear admission gate is the clearest case. It predicts its own
//! cost correctly (559 GMAC at a calibrated `GMAC/s`), then compares it against
//! `AlphaCrownConfig::deadline` — the 40 s root-α phase cap — because that is
//! the only deadline in scope five call levels down. Measured: the gate saw
//! **37 s and 38 s** when **186 s–226 s** was actually live, refused a build it
//! could afford, and the root degraded to plain IBP.
//!
//! Threading a global deadline through five layers of signature is the
//! "correct" fix and is why it has never been done. This is the cheap one: the
//! instance publishes its deadline once, and any gate that needs to reason
//! about the real remaining budget reads it.
//!
//! ## Why a process global is right here
//!
//! One deadline per process. `ny vnncomp` runs one instance per invocation, so
//! there is exactly one correct answer at any moment, and the alternative —
//! passing it through every intermediate signature that does not otherwise care
//! — is the reason the information is missing in the first place. There is
//! precedent in this tree for exactly this shape (`output_margin_seed`'s
//! published subset, `gpu_memory_ledger`).
//!
//! ## Contract
//!
//! - Publishing is **advisory**: unpublished means [`remaining`] returns `None`
//!   and every caller must fall back to today's behaviour. A gate must never
//!   become *more* permissive because the deadline is missing.
//! - This is scheduling information. It never touches a bound, and a wrong
//!   value can only mis-schedule, never mis-certify.

use std::sync::RwLock;
use std::time::{Duration, Instant};

static INSTANCE_DEADLINE: RwLock<Option<Instant>> = RwLock::new(None);

/// Publish the instance's authoritative deadline. Replaces any previous value.
///
/// Call once, as early as the deadline is known.
pub fn publish(deadline: Instant) {
    if let Ok(mut slot) = INSTANCE_DEADLINE.write() {
        *slot = Some(deadline);
    }
}

/// Clear the published deadline (test support, and process reuse).
pub fn clear() {
    if let Ok(mut slot) = INSTANCE_DEADLINE.write() {
        *slot = None;
    }
}

/// The published deadline, if any.
#[must_use]
pub fn deadline() -> Option<Instant> {
    INSTANCE_DEADLINE.read().ok().and_then(|s| *s)
}

/// Live budget remaining, or `None` when nothing is published.
///
/// Returns `Some(ZERO)` rather than `None` once the deadline has passed: "no
/// time left" and "no information" are different answers, and conflating them
/// is how a gate becomes accidentally permissive.
#[must_use]
pub fn remaining() -> Option<Duration> {
    remaining_at(Instant::now())
}

/// [`remaining`] at an explicit instant. Test seam.
#[must_use]
pub fn remaining_at(now: Instant) -> Option<Duration> {
    deadline().map(|d| d.saturating_duration_since(now))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These share a process global; serialise them.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn unpublished_reports_no_information_not_zero() {
        let _s = serial();
        clear();
        assert_eq!(
            remaining(),
            None,
            "absent must be distinguishable from spent"
        );
        assert_eq!(deadline(), None);
    }

    #[test]
    fn publishes_and_reports_live_budget() {
        let _s = serial();
        clear();
        let now = Instant::now();
        publish(now + Duration::from_secs(200));
        let r = remaining_at(now).expect("published");
        assert!(r >= Duration::from_secs(199) && r <= Duration::from_secs(200));
        clear();
    }

    #[test]
    fn a_spent_deadline_reports_zero_not_none() {
        // "no time left" and "no information" are different answers. A gate
        // that treats them alike becomes permissive at exactly the wrong moment.
        let _s = serial();
        clear();
        let now = Instant::now();
        publish(
            now.checked_sub(Duration::from_secs(5))
                .expect("5s before now is inside the process uptime"),
        );
        assert_eq!(remaining_at(now), Some(Duration::ZERO));
        clear();
    }

    #[test]
    fn republishing_replaces() {
        let _s = serial();
        clear();
        let now = Instant::now();
        publish(now + Duration::from_secs(10));
        publish(now + Duration::from_mins(5));
        assert!(remaining_at(now).expect("published") > Duration::from_secs(200));
        clear();
    }

    #[test]
    fn composes_with_the_i1_admission_rule() {
        // The measured case end to end: the forward-linear build predicts 55s
        // on a quiet host. Against the 40s phase cap it is refused; against the
        // real 226s remaining it is admitted.
        use crate::phase_window::{admit, WindowPolicy};
        let _s = serial();
        clear();
        let now = Instant::now();
        let policy = WindowPolicy::default().with_max_frac(0.6);
        let predicted = Duration::from_secs(55);

        // What the gate sees today: the phase cap.
        assert!(!admit(predicted, Duration::from_secs(38), policy).is_admitted());

        // What it should see.
        publish(now + Duration::from_secs(226));
        let live = remaining_at(now).expect("published");
        assert!(
            admit(predicted, live, policy).is_admitted(),
            "with the real budget the build is affordable"
        );
        clear();
    }
}
