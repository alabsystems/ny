// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #hull-root / wave-6 theme T5 — **the LP proposes; a closed form or a
//! rigorous replay certifies.**
//!
//! This is the reusable seam shared by the certified single-hidden-layer hull
//! (`single_hidden_hull.rs`, #hull-root), the relusplitter LP-refined
//! intermediates (C7) and any later caller that wants an LP-quality bound
//! without trusting a floating-point LP solver.
//!
//! # The trust boundary
//!
//! AY's float simplex supplies multiplier advice and, only for explicitly
//! prepaid callers, optional raw original-column point advice. The point is
//! never a witness or a bound. The existing multiplier generator
//! ([`ay_milp::LpSession::float_dual_for_expr`], which AY itself documents as
//! "fast, and explicitly UNTRUSTED"). A float dual vector has no authority of
//! its own. It reaches a bound through exactly one of two routes, and nothing
//! else in this module can produce a number the caller may treat as a bound:
//!
//! 1. **Closed form** ([`CertifyRoute::ClosedForm`]): the CALLER's certifier
//!    receives the dual vector and re-derives a bound from it with its own
//!    outward-rounded arithmetic. Weak duality makes every correctly-signed
//!    multiplier vector yield a valid bound, so the certifier's soundness must
//!    never depend on the vector being optimal, feasible or even finite — a
//!    bad vector may only cost tightness. The sign convention of AY's float
//!    duals is not guaranteed, so both `y` and `-y` are offered and the larger
//!    certified value is kept (each is independently valid).
//! 2. **Rigorous replay** ([`CertifyRoute::RigorousReplay`], optional):
//!    [`ay_milp::LpSession::rigorous_bound_expr`], i.e. a Neumaier–Shcherbina
//!    directed-rounding correction of the float dual, falling back to AY's
//!    exact rational rim. This is the authority `LpTightener` already ships
//!    under (`tighten.rs`, `session_optimum`). It is valid for the LP **as
//!    encoded in f64** — so a caller may request it only when every
//!    coefficient of its model is itself a sound encoding of the relaxation it
//!    means (the hull caller does NOT: its knapsack coefficients are plain
//!    float, so it relies on route 1 alone).
//!
//! The rational → f64 conversion of a replayed bound rounds toward `-inf`
//! with an exact comparison ([`rational_lower_f64`]), so the f64 can never
//! claim more than the rational it came from.
//!
//! [`ProposeCertifyOutcome::float_dual_value`] is the plain-float dual
//! objective at the proposer's vector. It is TELEMETRY ONLY (the `lp_lb=`
//! field of an observable) and must never gate a verdict.

use ay_milp::{Col, LpSession, Model, Outcome, Sense, SolveOpts};
use num_rational::BigRational;
use num_traits::ToPrimitive;

use crate::{MipError, Result};

/// Which authority produced a certified bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CertifyRoute {
    /// The caller's closed-form certifier, evaluated on the LP's float duals.
    ClosedForm,
    /// AY's rigorous replay (Neumaier–Shcherbina, else the exact rim).
    RigorousReplay,
}

impl CertifyRoute {
    /// Stable lowercase label for observables.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ClosedForm => "closed_form",
            Self::RigorousReplay => "rigorous_replay",
        }
    }
}

/// Whether the rigorous replay route runs after the closed form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayPolicy {
    /// Closed form only (the model's coefficients are not a sound encoding).
    Skip,
    /// Also run `rigorous_bound_expr` (the model IS a sound encoding).
    Run,
}

/// Everything the seam learned about one minimization.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProposeCertifyOutcome {
    /// Whether AY's float lane returned a dual vector at all.
    pub duals_available: bool,
    /// The float dual vector that produced [`Self::closed_form`] (after the
    /// sign choice), for callers that want to reuse it (e.g. a warm start).
    /// UNTRUSTED.
    pub best_duals: Option<Vec<f64>>,
    /// Plain-float LP dual objective at the proposer's vector, WITHOUT the
    /// caller's constant. TELEMETRY ONLY — not a bound.
    pub float_dual_value: Option<f64>,
    /// Best closed-form certified lower bound over the dual sign candidates.
    pub closed_form: Option<f64>,
    /// Rigorous replay lower bound (when requested and conclusive).
    pub rigorous_replay: Option<f64>,
}

impl ProposeCertifyOutcome {
    /// The best CERTIFIED lower bound and the route that produced it. Only the
    /// two certified fields participate; the telemetry value never does.
    #[must_use]
    pub fn best(&self) -> Option<(f64, CertifyRoute)> {
        let closed = self.closed_form.map(|v| (v, CertifyRoute::ClosedForm));
        let replay = self
            .rigorous_replay
            .map(|v| (v, CertifyRoute::RigorousReplay));
        match (closed, replay) {
            (Some(a), Some(b)) => Some(if b.0 > a.0 { b } else { a }),
            (a, b) => a.or(b),
        }
    }
}

/// Minimize `objective` over `model`, letting AY's float simplex PROPOSE a
/// dual vector and `certify` (the caller's closed form) turn it into a bound.
///
/// `certify` receives a dual vector of length `model.num_rows()` (in the NS
/// convention of `ay_milp`'s `ns.rs`: `d = obj - A^T y`, a `>=` row wants
/// `y_r >= 0`, but the float lane may return either sign — both `y` and `-y`
/// are offered). It must return `None` rather than an unsound number whenever
/// it cannot certify.
///
/// # Errors
/// [`MipError::Solver`] when AY rejects the model or the objective (for
/// example an out-of-range column). A float lane that does not settle is NOT
/// an error: it yields `duals_available = false`.
pub fn propose_then_certify_min(
    model: &Model,
    objective: &[(Col, f64)],
    opts: &SolveOpts,
    replay: ReplayPolicy,
    certify: &mut dyn FnMut(&[f64]) -> Option<f64>,
) -> Result<ProposeCertifyOutcome> {
    propose_then_certify_min_inner(
        model,
        objective,
        opts,
        replay,
        certify,
        FloatAdviceRequest::DualOnly,
    )
    .map(|(outcome, _)| outcome)
}

/// The same one-LP seam with explicitly UNTRUSTED original-column advice.
/// Caller must prepay prefix validation/copy, capture maps, decoding and peak
/// payload BEFORE entering. Missing primal never loses the usual duals.
/// # Errors
/// The same model/objective errors as [`propose_then_certify_min`].
pub(crate) fn propose_then_certify_min_with_advice(
    model: &Model,
    objective: &[(Col, f64)],
    opts: &SolveOpts,
    replay: ReplayPolicy,
    certify: &mut dyn FnMut(&[f64]) -> Option<f64>,
) -> Result<(ProposeCertifyOutcome, Option<Box<[f64]>>)> {
    propose_then_certify_min_inner(
        model,
        objective,
        opts,
        replay,
        certify,
        FloatAdviceRequest::PrimalAndDual,
    )
}

enum FloatAdviceRequest {
    DualOnly,
    PrimalAndDual,
}

fn propose_then_certify_min_inner(
    model: &Model,
    objective: &[(Col, f64)],
    opts: &SolveOpts,
    replay: ReplayPolicy,
    certify: &mut dyn FnMut(&[f64]) -> Option<f64>,
    advice: FloatAdviceRequest,
) -> Result<(ProposeCertifyOutcome, Option<Box<[f64]>>)> {
    let mut outcome = ProposeCertifyOutcome::default();
    let mut session = LpSession::new(model, opts).map_err(|e| MipError::Solver(e.to_string()))?;
    let mut primal = None;
    let duals = match advice {
        FloatAdviceRequest::DualOnly => session.float_dual_for_expr(objective, Sense::Minimize),
        FloatAdviceRequest::PrimalAndDual => session
            .float_advice_for_expr(objective, Sense::Minimize)
            .map(|answer| {
                answer.map(|answer| {
                    primal = answer.primal;
                    answer.duals
                })
            }),
    }
    .map_err(|e| MipError::Solver(e.to_string()))?;
    if let Some(y) = duals {
        outcome.duals_available = true;
        let neg: Vec<f64> = y.iter().map(|v| -v).collect();
        for candidate in [y, neg] {
            if let Some(value) = float_dual_objective(model, objective, &candidate) {
                outcome.float_dual_value = Some(
                    outcome
                        .float_dual_value
                        .map_or(value, |best: f64| best.max(value)),
                );
            }
            if let Some(bound) = certify(&candidate) {
                if bound.is_finite() && outcome.closed_form.is_none_or(|best| bound > best) {
                    outcome.closed_form = Some(bound);
                    outcome.best_duals = Some(candidate);
                }
            }
        }
    }
    if replay == ReplayPolicy::Run {
        let replayed = session
            .rigorous_bound_expr(objective, Sense::Minimize)
            .map_err(|e| MipError::Solver(e.to_string()))?;
        outcome.rigorous_replay = rigorous_lower_f64(&replayed);
    }
    Ok((outcome, primal))
}

/// A rigorous LOWER bound, as f64, from a minimization [`Outcome`].
///
/// Admits only `Bound { rigorous: true }` and `Optimal` (the exact rim's
/// optimum is exact, hence rigorous); every other outcome is `None`. The
/// rational is rounded toward `-inf` by [`rational_lower_f64`].
#[must_use]
pub fn rigorous_lower_f64(outcome: &Outcome) -> Option<f64> {
    match outcome {
        Outcome::Bound {
            dual_bound,
            rigorous: true,
        } => rational_lower_f64(dual_bound),
        Outcome::Optimal { value, .. } => rational_lower_f64(value),
        _ => None,
    }
}

/// The largest f64 found that is `<=` the exact rational `value`, by an exact
/// comparison (never by trusting `to_f64`'s rounding mode). `None` when the
/// value does not fit a finite f64 or the downward walk fails to certify
/// within a few ulps (fail closed).
#[must_use]
pub fn rational_lower_f64(value: &BigRational) -> Option<f64> {
    let mut f = value.to_f64()?;
    if !f.is_finite() {
        return None;
    }
    for _ in 0..4 {
        let back = BigRational::from_float(f)?;
        if back <= *value {
            return Some(f);
        }
        f = f.next_down();
        if !f.is_finite() {
            return None;
        }
    }
    None
}

/// Plain-float Lagrangian dual objective of `min objective·x` over `model` at
/// the multiplier vector `y` (NS convention: `d = obj - A^T y`).
///
/// TELEMETRY ONLY. It is evaluated in round-to-nearest arithmetic and treats a
/// reduced cost within `1e-9` of zero on an infinite bound side as zero, so it
/// is NOT a bound. It exists so an observable can print the LP value the
/// proposer believes in next to the certified value, making a certifier or
/// dual-mapping defect visible as a gap between the two.
#[must_use]
pub fn float_dual_objective(model: &Model, objective: &[(Col, f64)], y: &[f64]) -> Option<f64> {
    let rows = model.num_rows();
    let cols = model.num_cols();
    if y.len() != rows || y.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut reduced = vec![0.0_f64; cols];
    for &(col, w) in objective {
        let index = col.index();
        if index >= cols || !w.is_finite() {
            return None;
        }
        reduced[index] += w;
    }
    let mut value = 0.0_f64;
    for (r, &yr) in y.iter().enumerate() {
        let row = model.row_at(r)?;
        let (coeffs, lb, ub) = model.row(row);
        for &(c, a) in coeffs {
            let index = c as usize;
            if index >= cols {
                return None;
            }
            reduced[index] -= a * yr;
        }
        value += side_term(yr, lb, ub)?;
    }
    for (j, &dj) in reduced.iter().enumerate() {
        let col = model.col_at(j)?;
        let (lb, ub) = model.col_bounds(col);
        value += side_term(dj, lb, ub)?;
    }
    value.is_finite().then_some(value)
}

/// `min_{s in [lb, ub]} coef * s` in plain float, with a `1e-9` dead zone on
/// an infinite side (telemetry only; see [`float_dual_objective`]).
fn side_term(coef: f64, lb: f64, ub: f64) -> Option<f64> {
    const DEAD_ZONE: f64 = 1e-9;
    if coef > 0.0 {
        if lb.is_finite() {
            Some(coef * lb)
        } else if coef <= DEAD_ZONE {
            Some(0.0)
        } else {
            None
        }
    } else if coef < 0.0 {
        if ub.is_finite() {
            Some(coef * ub)
        } else if -coef <= DEAD_ZONE {
            Some(0.0)
        } else {
            None
        }
    } else {
        Some(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn opts() -> SolveOpts {
        SolveOpts::new()
            .with_time_limit(Duration::from_secs(20))
            .with_deadline(Instant::now() + Duration::from_secs(20))
    }

    /// min x + y  s.t.  x + y >= 1,  x in [0, 2], y in [0, 2]. Optimum 1.
    fn tiny_lp() -> (Model, Vec<(Col, f64)>) {
        let mut model = Model::new();
        let x = model.add_col(0.0, 2.0);
        let y = model.add_col(0.0, 2.0);
        let _ = model.add_row(1.0, f64::INFINITY, &[(x, 1.0), (y, 1.0)]);
        (model, vec![(x, 1.0), (y, 1.0)])
    }

    #[test]
    fn rational_lower_f64_never_exceeds_the_rational() {
        let third = BigRational::new(1.into(), 3.into());
        let f = rational_lower_f64(&third).expect("finite");
        assert!(BigRational::from_float(f).expect("finite") <= third);
        let neg_third = -third;
        let g = rational_lower_f64(&neg_third).expect("finite");
        assert!(BigRational::from_float(g).expect("finite") <= neg_third);
        let exact = BigRational::from_float(0.75_f64).expect("finite");
        assert_eq!(rational_lower_f64(&exact), Some(0.75));
    }

    #[test]
    fn seam_offers_both_dual_signs_and_keeps_only_certified_values() {
        let (model, objective) = tiny_lp();
        let mut offered = 0usize;
        // A "certifier" that certifies the dual objective of the ONE row with
        // y >= 0 only (weak duality for this LP: y * 1 + sum_j min_box (1-y) x_j,
        // valid for y in [0, 1]); it declines otherwise.
        let mut certify = |y: &[f64]| -> Option<f64> {
            offered += 1;
            let yr = *y.first()?;
            if !(0.0..=1.0).contains(&yr) {
                return None;
            }
            // (1 - y) >= 0, so the box minimum of each (1-y) x_j is at 0.
            Some(yr.next_down())
        };
        let outcome =
            propose_then_certify_min(&model, &objective, &opts(), ReplayPolicy::Run, &mut certify)
                .expect("tiny LP solves");
        // Fail closed: a disabled float lane would silently turn the
        // closed-form half of this test into a no-op.
        assert!(outcome.duals_available, "AY's float lane returned no duals");
        assert_eq!(offered, 2, "both dual signs must be offered");
        let closed = outcome.closed_form.expect("one sign certifies");
        assert!(closed <= 1.0, "closed form {closed} exceeds the optimum");
        assert!(closed > 0.99, "closed form {closed} is not the LP dual");
        let lp = outcome.float_dual_value.expect("telemetry value");
        assert!((lp - 1.0).abs() < 1e-9, "float dual objective {lp}");
        let replay = outcome.rigorous_replay.expect("rigorous replay concludes");
        assert!(
            replay <= 1.0,
            "rigorous replay {replay} exceeds the optimum"
        );
        assert!(replay > 0.99);
        let (best, _) = outcome.best().expect("some certified bound");
        assert!(best <= 1.0);
    }

    #[test]
    fn best_ignores_telemetry_and_prefers_the_larger_certified_value() {
        let outcome = ProposeCertifyOutcome {
            duals_available: true,
            best_duals: None,
            float_dual_value: Some(100.0),
            closed_form: Some(0.5),
            rigorous_replay: Some(0.25),
        };
        assert_eq!(outcome.best(), Some((0.5, CertifyRoute::ClosedForm)));
        let none = ProposeCertifyOutcome {
            float_dual_value: Some(100.0),
            ..ProposeCertifyOutcome::default()
        };
        assert_eq!(none.best(), None, "telemetry is never a bound");
    }
}
