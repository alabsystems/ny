// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #phase-window — invariant I1: no fixed seconds.
//!
//! Design: `docs/DESIGN_MARGINAL_VALUE_SCHEDULER_2026-08-08.md` §2.3.
//!
//! ## The defect
//!
//! Five phase budgets in ny are absolute constants, so their share of the run
//! swings wildly with the instance budget and none of them scales with the
//! host. `root_alpha_cap_secs: 40` is **51% / 17% / 4%** of the BaB slice at
//! 100 s / 330 s / 1200 s. `ROOT_SPEC_ALPHA_GRACE_CAP` is another fixed 40 s,
//! `ROOT_SPEC_GRACE` 3 s, `root_crown_interm_max_secs` 2 s, and the upfront
//! attack lane pins at 4 s above ~53 s of budget.
//!
//! The second half of the defect is worse than the scaling: a phase given a
//! window smaller than its own cost **starts anyway and is cut off**. The
//! forward-linear root costs a measured 55–82 s on this host and was handed a
//! 40 s window; it burned the window and produced nothing, and the run then
//! degraded to plain IBP.
//!
//! ## The rule
//!
//! A window is derived from **predicted cost on this host**, and admitted only
//! if it fits inside a *fraction* of what remains. A phase that does not fit
//! [`Admission::Declined`]s — it never half-runs.
//!
//! `max_frac < 1` is what reserves budget for everything downstream. The
//! measured failure it prevents: a 63 s root inside a 91 s instance left 9 s of
//! branch-and-bound, and the verdict did not move.
//!
//! This generalises `#forward-linear-cost-gate`, which already predicts its own
//! cost from a calibrated `GMAC/s` and applies a ×5/4 admission margin — but for
//! exactly one phase, and against a deadline that was not its own.

use std::time::Duration;

use crate::phase_yield::DeclineReason;

/// Outcome of asking whether a phase can be afforded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Admission {
    /// Run it, for at most this long.
    Admitted(Duration),
    /// Do not start it.
    Declined(DeclineReason),
}

impl Admission {
    #[must_use]
    pub const fn window(&self) -> Option<Duration> {
        match self {
            Self::Admitted(d) => Some(*d),
            Self::Declined(_) => None,
        }
    }

    #[must_use]
    pub const fn is_admitted(&self) -> bool {
        matches!(self, Self::Admitted(_))
    }
}

/// Admission policy for one phase.
#[derive(Debug, Clone, Copy)]
pub struct WindowPolicy {
    /// Largest share of the REMAINING budget this phase may consume. Strictly
    /// below 1.0 so something is always left for what comes after — this is the
    /// term that stops a root pass eating the instance.
    pub max_frac: f64,
    /// Safety multiplier on the prediction, because a cost model that is a
    /// little optimistic must not produce a phase that is cut off. Mirrors the
    /// existing ×5/4 forward-linear admission margin.
    pub margin_num: u32,
    pub margin_den: u32,
    /// Below this, starting is pointless regardless of arithmetic.
    pub floor: Duration,
}

impl Default for WindowPolicy {
    fn default() -> Self {
        Self {
            max_frac: 0.5,
            margin_num: 5,
            margin_den: 4,
            floor: Duration::from_millis(100),
        }
    }
}

impl WindowPolicy {
    #[must_use]
    pub fn with_max_frac(mut self, frac: f64) -> Self {
        self.max_frac = frac;
        self
    }
}

/// Decide whether a phase fits, and for how long.
///
/// `predicted` is the phase's own cost estimate **on this host** — from a
/// measured rate, not a constant. `remaining` is the live global budget.
///
/// Returns [`Admission::Declined`] rather than a truncated window whenever the
/// padded prediction does not fit. That refusal is the point: a phase cut off
/// mid-way has spent the budget and produced nothing, which is strictly worse
/// than never starting.
#[must_use]
pub fn admit(predicted: Duration, remaining: Duration, policy: WindowPolicy) -> Admission {
    let padded = predicted
        .checked_mul(policy.margin_num)
        .map_or(Duration::MAX, |d| d / policy.margin_den.max(1));

    let frac = if policy.max_frac.is_finite() {
        policy.max_frac.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let affordable = remaining.mul_f64(frac);

    if affordable < policy.floor || padded > affordable {
        return Admission::Declined(DeclineReason::Unaffordable {
            predicted_ms: u64::try_from(padded.as_millis()).unwrap_or(u64::MAX),
            available_ms: u64::try_from(affordable.as_millis()).unwrap_or(u64::MAX),
        });
    }
    // Grant the padded prediction, never the whole affordable share: a phase
    // that finishes early must return the remainder to the pool rather than
    // expand into it.
    Admission::Admitted(padded)
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: fn(u64) -> Duration = Duration::from_secs;

    #[test]
    fn a_phase_that_does_not_fit_declines_rather_than_truncating() {
        // THE measured case: the forward-linear root costs ~82s and was handed
        // a 40s window. It started, burned the window, produced nothing, and
        // the run degraded to plain IBP. Declining is strictly better.
        let a = admit(S(82), S(95), WindowPolicy::default());
        assert!(!a.is_admitted(), "must refuse, not truncate to 40s");
        match a {
            Admission::Declined(DeclineReason::Unaffordable { predicted_ms, .. }) => {
                assert_eq!(predicted_ms, 102_500, "82s padded by 5/4");
            }
            other => panic!("expected Unaffordable, got {other:?}"),
        }
    }

    #[test]
    fn the_window_scales_with_the_budget_which_is_the_whole_point() {
        // The same phase, the same host, three instance budgets. A fixed
        // constant would return the same number three times; that is the defect.
        let p = WindowPolicy::default().with_max_frac(0.6);
        assert!(!admit(S(55), S(60), p).is_admitted(), "no room at 60s");
        assert!(admit(S(55), S(200), p).is_admitted(), "fits at 200s");
        assert!(admit(S(55), S(400), p).is_admitted(), "fits at 400s");
    }

    #[test]
    fn max_frac_reserves_budget_for_everything_downstream() {
        // Measured failure this prevents: a 63s root inside a 91s instance left
        // 9s of BaB and the verdict did not move. At max_frac 0.5 a 63s phase
        // is refused at 91s, and the padded 78.75s only fits from ~158s.
        let p = WindowPolicy::default();
        assert!(!admit(S(63), S(91), p).is_admitted());
        assert!(admit(S(63), S(200), p).is_admitted());
    }

    #[test]
    fn a_phase_never_gets_the_whole_affordable_share() {
        // It gets its padded prediction. Finishing early returns the rest to
        // the pool instead of expanding into it.
        match admit(S(10), S(600), WindowPolicy::default()) {
            Admission::Admitted(w) => assert_eq!(w, Duration::from_millis(12_500)),
            other => panic!("expected admission, got {other:?}"),
        }
    }

    #[test]
    fn a_tiny_remaining_budget_declines_on_the_floor() {
        let a = admit(
            Duration::from_millis(1),
            Duration::from_millis(20),
            WindowPolicy::default(),
        );
        assert!(!a.is_admitted(), "below the floor, starting is pointless");
    }

    #[test]
    fn zero_and_nonfinite_fractions_decline_rather_than_panicking() {
        let z = WindowPolicy::default().with_max_frac(0.0);
        assert!(!admit(S(1), S(1000), z).is_admitted());
        let nan = WindowPolicy::default().with_max_frac(f64::NAN);
        assert!(!admit(S(1), S(1000), nan).is_admitted());
        let over = WindowPolicy::default().with_max_frac(9.0);
        // Clamped to 1.0, so it may admit — but never more than the budget.
        if let Admission::Admitted(w) = admit(S(1), S(1000), over) {
            assert!(w <= S(1000));
        }
    }

    #[test]
    fn the_cifar100_forward_linear_case_end_to_end() {
        // Host-measured rate varies with load, and the SAME phase flips
        // admission because of it -- which a fixed 40s window can never express.
        // 559.37 GMAC at 6.80 GMAC/s = 82s (loaded); at 10.10 GMAC/s = 55s.
        let p = WindowPolicy::default().with_max_frac(0.6);
        let tier = S(95);
        assert!(!admit(S(82), tier, p).is_admitted(), "loaded host: refuse");
        assert!(
            !admit(S(55), tier, p).is_admitted(),
            "quiet host: 68.75s > 57s"
        );
        // And at the 330s regime the quiet-host build fits comfortably.
        assert!(admit(S(55), S(226), p).is_admitted());
    }
}
