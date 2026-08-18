// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Predicate-aware LP over a star set `X = { c + G·α : A·α ≤ b, α ∈ [-1,1]^m }`.
//!
//! [`ny_tensor::zonotope::Star`] carries the exact ReLU transformer but cannot solve an LP:
//! the dependency edge is `ny-mip → ny-tensor`, so the solver is unreachable from there
//! (see [`ny_tensor::zonotope::Star::bounds_lp`], which is a stub explaining exactly this).
//! This module is the
//! other half — it lives ABOVE `ny-tensor`, where `ay_milp` is reachable, and consumes the
//! star's `(center, generators, A, b)`.
//!
//! ## What it is for
//!
//! Exact star enumeration splits once per unstable ReLU, so the population is `2^unstable`
//! in the worst case. Two things keep that tractable, and both need this LP:
//!
//! 1. **Feasibility pruning.** A split adds a predicate row; many branches are empty and
//!    every descendant of an empty branch is wasted work. Dropping them early is where the
//!    method gets its leverage.
//! 2. **Tighter stability tests.** [`ny_tensor::zonotope::Star::relu_split`] decides stability from
//!    box-α interval bounds, which ignore `A·α ≤ b`. The predicate-aware range is a subset,
//!    so neurons the interval test calls unstable are often provably stable here — each one
//!    removed halves the remaining population.
//!
//! ## SOUNDNESS
//!
//! The backend is rigorous, not raw float. [`crate::tighten::RelaxationObbt`] documents its
//! `bounds` as "rigorous outward-rounded f64 bounds" and its `infeasible` flag as set only
//! when "a rigorous solve proved the whole relaxation infeasible". So both outputs are
//! usable here without a separate certification pass.
//!
//! Two points specific to this caller:
//!
//! * `RelaxationObbt` warns that `infeasible` is "only the relaxation, not the MILP". For a
//!   star that distinction collapses: the predicate set IS a polytope, so the LP is the
//!   exact feasibility question, and a rigorous infeasibility proof means the branch really
//!   is empty and dropping it loses nothing.
//! * The bounds are outward-rounded, so they can only ever be WEAKER than the truth. A
//!   neuron this test calls stable is therefore genuinely stable, exactly as with the
//!   interval test — the LP just calls it stable more often.
//!
//! [`StarLpReport::sound_bounds`] is still offered as a belt-and-braces intersection for
//! callers who want the result to never be narrower than a bound they proved independently.

use std::time::{Duration, Instant};

use crate::ir::{Col, MilpProblem};
use crate::tighten::obbt_relaxation_bounds;
use crate::{MipError, Result};
use ay_milp::{Col as AyCol, LpSession, Outcome, Sense, SolveOpts};
use num_rational::BigRational;
use num_traits::ToPrimitive;

/// One star's `(center, generators)` for a chosen set of output coordinates, plus its
/// predicate polytope. All rows are in the star's α space of dimension `alpha_dim`.
#[derive(Debug, Clone)]
pub struct StarLpRequest {
    /// α-space dimension `m`.
    pub alpha_dim: usize,
    /// Predicate matrix `A`, row-major, `k` rows of length `alpha_dim`.
    pub a_rows: Vec<Vec<f64>>,
    /// Predicate right-hand side `b`, length `k`.
    pub b: Vec<f64>,
    /// Target coordinates as affine forms `(c_i, g_i)` with `g_i.len() == alpha_dim`.
    pub targets: Vec<(f64, Vec<f64>)>,
}

/// What the rigorous LP proved. See the soundness notes in the module docs.
#[derive(Debug, Clone)]
pub struct StarLpReport {
    /// Rigorous outward-rounded per-target `(lower, upper)`, in `targets` order.
    pub lp_bounds: Vec<(f64, f64)>,
    /// A rigorous solve proved the predicate polytope EMPTY: the branch is unreachable
    /// and may be dropped.
    pub infeasible: bool,
}

impl StarLpReport {
    /// Intersect the LP box with the caller's own SOUND interval bounds, keeping the
    /// WEAKER side of each — so a wrong (too tight) LP answer cannot narrow the result
    /// below what was already proven independently.
    ///
    /// This is the only way to consume [`Self::lp_bounds`] without inheriting the
    /// solver's floating-point trust.
    #[must_use]
    pub fn sound_bounds(&self, interval: &[(f64, f64)]) -> Vec<(f64, f64)> {
        interval
            .iter()
            .enumerate()
            .map(|(i, &(ilo, ihi))| match self.lp_bounds.get(i) {
                Some(&(llo, lhi)) if llo.is_finite() && lhi.is_finite() => {
                    (ilo.min(llo), ihi.max(lhi))
                }
                _ => (ilo, ihi),
            })
            .collect()
    }
}

impl StarLpRequest {
    fn validate(&self) -> Result<()> {
        if self.alpha_dim == 0 {
            return Err(MipError::Encoding("star LP: alpha_dim must be >= 1".into()));
        }
        if self.a_rows.len() != self.b.len() {
            return Err(MipError::Encoding(format!(
                "star LP: {} predicate rows but {} rhs entries",
                self.a_rows.len(),
                self.b.len()
            )));
        }
        for row in &self.a_rows {
            if row.len() != self.alpha_dim {
                return Err(MipError::Encoding(format!(
                    "star LP: predicate row width {} != alpha_dim {}",
                    row.len(),
                    self.alpha_dim
                )));
            }
        }
        for (idx, (c, g)) in self.targets.iter().enumerate() {
            if g.len() != self.alpha_dim {
                return Err(MipError::Encoding(format!(
                    "star LP: target {idx} generator width {} != alpha_dim {}",
                    g.len(),
                    self.alpha_dim
                )));
            }
            if !c.is_finite() || g.iter().any(|v| !v.is_finite()) {
                return Err(MipError::Encoding(format!(
                    "star LP: target {idx} has a non-finite affine form"
                )));
            }
        }
        if self.b.iter().any(|v| !v.is_finite())
            || self.a_rows.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(MipError::Encoding(
                "star LP: non-finite predicate entry".into(),
            ));
        }
        Ok(())
    }

    /// Encode as `α ∈ [-1,1]^m`, `A·α ≤ b`, and one free column per target pinned by an
    /// equality row `x_i − g_i·α = c_i`.
    fn encode(&self) -> Result<(MilpProblem, Vec<Col>)> {
        self.validate()?;
        let mut problem = MilpProblem::new();

        // α columns. Objective coefficients are irrelevant: OBBT drives min/max per column.
        let alpha: Vec<Col> = (0..self.alpha_dim)
            .map(|_| problem.add_col(0.0, -1.0, 1.0))
            .collect();

        // Predicate rows: A·α ≤ b  ⇒  (-inf, b].
        for (row, &rhs) in self.a_rows.iter().zip(&self.b) {
            let coeffs: Vec<(Col, f64)> = row
                .iter()
                .enumerate()
                .filter(|(_, w)| **w != 0.0)
                .map(|(j, &w)| (alpha[j], w))
                .collect();
            problem.add_row(f64::NEG_INFINITY, rhs, coeffs);
        }

        // One column per target, tied to its affine form by an equality:
        //   x_i - g_i·α = c_i
        let mut target_cols = Vec::with_capacity(self.targets.len());
        for (c, g) in &self.targets {
            let x = problem.add_col(0.0, f64::NEG_INFINITY, f64::INFINITY);
            let mut coeffs: Vec<(Col, f64)> = vec![(x, 1.0)];
            coeffs.extend(
                g.iter()
                    .enumerate()
                    .filter(|(_, w)| **w != 0.0)
                    .map(|(j, &w)| (alpha[j], -w)),
            );
            problem.add_row(*c, *c, coeffs);
            target_cols.push(x);
        }
        Ok((problem, target_cols))
    }
}

/// Solve the star's predicate LP for every target coordinate.
///
/// Returns rigorous outward-rounded per-target bounds and a rigorous infeasibility flag
/// (see the module-level soundness notes).
///
/// # Errors
/// [`MipError::Encoding`] on a malformed request; [`MipError::Solver`] on a backend
/// failure.
pub fn star_predicate_bounds(
    request: &StarLpRequest,
    time_limit: Duration,
    deadline: Instant,
) -> Result<StarLpReport> {
    let (problem, targets) = request.encode()?;
    if targets.is_empty() {
        return Ok(StarLpReport {
            lp_bounds: Vec::new(),
            infeasible: false,
        });
    }
    let obbt = obbt_relaxation_bounds(&problem, &targets, 1, time_limit, deadline, targets.len())?;
    Ok(StarLpReport {
        lp_bounds: obbt.bounds,
        infeasible: obbt.infeasible,
    })
}

#[cfg(test)]
#[path = "star_lp_tests.rs"]
mod tests;

/// Outward-round a rational bound to f64: `Minimize` toward `-inf`, `Maximize` toward `+inf`,
/// so the f64 can never claim more than the exact rational it came from.
fn outward_f64(bound: &BigRational, sense: Sense) -> Option<f64> {
    let f = bound.to_f64()?;
    if !f.is_finite() {
        return None;
    }
    let back = BigRational::from_float(f)?;
    let already_safe = match sense {
        Sense::Minimize => back <= *bound,
        Sense::Maximize => back >= *bound,
    };
    if already_safe {
        return Some(f);
    }
    Some(match sense {
        Sense::Minimize => next_down_f64(f),
        Sense::Maximize => next_up_f64(f),
    })
}

fn next_up_f64(x: f64) -> f64 {
    f64::from_bits(if x >= 0.0 {
        x.to_bits() + 1
    } else {
        x.to_bits() - 1
    })
}

fn next_down_f64(x: f64) -> f64 {
    f64::from_bits(if x > 0.0 {
        x.to_bits() - 1
    } else {
        x.to_bits() + 1
    })
}

/// Pull a sound bound out of an `Outcome`, or `None` when it carries none.
fn outcome_bound(outcome: &Outcome, sense: Sense) -> Option<f64> {
    match outcome {
        Outcome::Bound { dual_bound, .. } => outward_f64(dual_bound, sense),
        Outcome::Optimal { value, .. } => outward_f64(value, sense),
        _ => None,
    }
}

/// A star LP session that is built ONCE and then queried per coordinate.
///
/// ## Why this exists
///
/// [`star_predicate_bounds`] goes through `obbt_relaxation_bounds`, which constructs a fresh
/// `ay_milp::LpSession` and runs OBBT rounds on EVERY call. In the star driver that dominated
/// the runtime: ~30 s per node on a problem with five α variables, which is model build and
/// session setup, not solving.
///
/// `ay_milp` is designed for reuse — `LpSession::tighten_col_bounds_rigorous` answers one
/// column against an already-built session. This type exposes that: pay the setup once per
/// node, then query every coordinate cheaply.
///
/// Bounds come from `rigorous_bound`, so they carry the same rigour as the batch path.
///
/// ## NOT YET WIRED INTO THE DRIVER
///
/// A first attempt to use one session per driver node regressed the search: coordinates the
/// batch path resolved came back unresolved here, so the search split to its depth cap
/// instead of deciding. No wrong answer was produced (soundness never depended on it), but
/// the driver got weaker, so it was reverted.
///
/// The cause was a misuse on THIS side, not an AY defect: `tighten_col_bounds_rigorous`
/// RETURNS the two `Outcome`s and does not commit them (there is no `narrow_col_bounds`
/// inside it), so reading `col_bounds` afterwards still reports a free target column's
/// original `(-inf, +inf)`. The bound must be taken from `Outcome::Bound { dual_bound, .. }`,
/// which is what [`Self::bounds`] now does.
pub struct StarLpSession {
    session: LpSession,
    targets: Vec<AyCol>,
}

impl StarLpSession {
    /// Build a session over the ALPHA SPACE ONLY (`m` columns, `k` predicate rows) and bound
    /// each target as a linear EXPRESSION rather than a materialised column.
    ///
    /// This is what `ay_milp::LpSession::rigorous_bound_expr` was added for. The previous
    /// encoding gave every target its own column plus an equality row, so a star with `n`
    /// output coordinates solved against a model of `m + n` columns and `k + n` rows even
    /// when one coordinate was wanted — and query time scales with model size (measured:
    /// 29ms at one target, 138ms at fifty). Here the model is the star's polytope, full stop,
    /// and every coordinate is just a different objective over it.
    ///
    /// # Errors
    /// [`MipError::Encoding`] on a malformed request; [`MipError::Solver`] if AY rejects the
    /// model.
    pub fn new_alpha_only(
        request: &StarLpRequest,
        time_limit: Duration,
        deadline: Instant,
    ) -> Result<Self> {
        request.validate()?;
        let mut problem = MilpProblem::new();
        let alpha: Vec<Col> = (0..request.alpha_dim)
            .map(|_| problem.add_col(0.0, -1.0, 1.0))
            .collect();
        for (row, &rhs) in request.a_rows.iter().zip(&request.b) {
            let coeffs: Vec<(Col, f64)> = row
                .iter()
                .enumerate()
                .filter(|(_, w)| **w != 0.0)
                .map(|(j, &w)| (alpha[j], w))
                .collect();
            problem.add_row(f64::NEG_INFINITY, rhs, coeffs);
        }
        let model = crate::ay_lib::to_ay_model_relaxed(&problem)?;
        let ay_alpha = alpha
            .iter()
            .map(|c| {
                model.col_at(c.0).ok_or_else(|| {
                    MipError::Encoding(format!("star LP: alpha column {} out of range", c.0))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let opts = SolveOpts::new()
            .with_time_limit(time_limit)
            .with_deadline(deadline);
        let session = LpSession::new(&model, &opts).map_err(|e| MipError::Solver(e.to_string()))?;
        Ok(Self {
            session,
            targets: ay_alpha,
        })
    }

    /// Rigorous bounds on `c + g·α` over this session's polytope, with no model growth.
    ///
    /// # Errors
    /// [`MipError::Solver`] on a backend failure.
    pub fn expr_bounds(&mut self, c: f64, g: &[f64]) -> Result<(f64, f64)> {
        if g.len() != self.targets.len() {
            return Err(MipError::Encoding(format!(
                "star LP: generator width {} != alpha dim {}",
                g.len(),
                self.targets.len()
            )));
        }
        let expr: Vec<(AyCol, f64)> = self
            .targets
            .iter()
            .zip(g)
            .filter(|(_, w)| **w != 0.0)
            .map(|(&col, &w)| (col, w))
            .collect();
        let lo = self
            .session
            .rigorous_bound_expr(&expr, Sense::Minimize)
            .map_err(|e| MipError::Solver(e.to_string()))?;
        let hi = self
            .session
            .rigorous_bound_expr(&expr, Sense::Maximize)
            .map_err(|e| MipError::Solver(e.to_string()))?;
        if matches!(lo, Outcome::Infeasible { .. }) || matches!(hi, Outcome::Infeasible { .. }) {
            return Ok((f64::NAN, f64::NAN));
        }
        // The expression bounds `g·α`; the star's constant `c` shifts it.
        Ok((
            outcome_bound(&lo, Sense::Minimize).map_or(f64::NEG_INFINITY, |v| c + v),
            outcome_bound(&hi, Sense::Maximize).map_or(f64::INFINITY, |v| c + v),
        ))
    }

    /// Build the session for one star.
    ///
    /// # Errors
    /// [`MipError::Encoding`] on a malformed request, [`MipError::Solver`] if the backend
    /// rejects the model.
    pub fn new(request: &StarLpRequest, time_limit: Duration, deadline: Instant) -> Result<Self> {
        let (problem, targets) = request.encode()?;
        let model = crate::ay_lib::to_ay_model_relaxed(&problem)?;
        let ay_targets = targets
            .iter()
            .map(|c| {
                model.col_at(c.0).ok_or_else(|| {
                    MipError::Encoding(format!("star LP: target column {} out of range", c.0))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let opts = SolveOpts::new()
            .with_time_limit(time_limit)
            .with_deadline(deadline);
        let session = LpSession::new(&model, &opts).map_err(|e| MipError::Solver(e.to_string()))?;
        Ok(Self {
            session,
            targets: ay_targets,
        })
    }

    /// Rigorous bounds for one target, by index into the request's `targets`.
    ///
    /// Returns `None` when the index is out of range. An infeasible model surfaces as
    /// [`Self::infeasible`] on the same query.
    ///
    /// # Errors
    /// [`MipError::Solver`] on a backend failure.
    pub fn bounds(&mut self, idx: usize) -> Result<Option<(f64, f64)>> {
        let Some(&col) = self.targets.get(idx) else {
            return Ok(None);
        };
        match self.session.tighten_col_bounds_rigorous(col) {
            Ok((lo, hi)) => {
                if matches!(lo, Outcome::Infeasible { .. })
                    || matches!(hi, Outcome::Infeasible { .. })
                {
                    return Ok(Some((f64::NAN, f64::NAN)));
                }
                // Read the OUTCOMES. The call does not narrow the model, so `col_bounds`
                // would still report the target column's original free box.
                Ok(Some((
                    outcome_bound(&lo, Sense::Minimize).unwrap_or(f64::NEG_INFINITY),
                    outcome_bound(&hi, Sense::Maximize).unwrap_or(f64::INFINITY),
                )))
            }
            Err(e) => Err(MipError::Solver(e.to_string())),
        }
    }

    /// Did a rigorous solve prove the whole predicate infeasible?
    ///
    /// # Errors
    /// [`MipError::Solver`] on a backend failure.
    pub fn infeasible(&mut self) -> Result<bool> {
        let Some(&col) = self.targets.first() else {
            return Ok(false);
        };
        match self.session.tighten_col_bounds_rigorous(col) {
            Ok((lo, hi)) => Ok(matches!(lo, Outcome::Infeasible { .. })
                || matches!(hi, Outcome::Infeasible { .. })),
            Err(e) => Err(MipError::Solver(e.to_string())),
        }
    }
}

impl StarLpSession {
    /// Sound bounds on `c + g·α` using AY's FLOAT simplex for the multipliers and this
    /// crate's own Lagrangian evaluation for the bound.
    ///
    /// The untrusted-solver / trusted-verifier split. `rigorous_bound_expr` is rigorous but
    /// falls to the exact rational rim whenever Neumaier–Shcherbina declines, which on the
    /// tight near-degenerate polytopes an exact ReLU split produces is the common case and
    /// costs ~300 ms. Here the simplex only has to supply a good `λ`; weak duality makes
    /// [`crate::star_dual::dual_bound_at`] valid for ANY `λ ≥ 0`, so a wrong `λ` costs
    /// tightness and never soundness, and the rim is never entered.
    ///
    /// Returns `None` when the float lane declines, leaving the caller to fall back.
    ///
    /// # Errors
    /// [`MipError::Solver`] on a backend failure.
    pub fn verified_float_bounds(
        &mut self,
        c: f64,
        g: &[f64],
        a_rows: &[Vec<f64>],
        b: &[f64],
    ) -> Result<Option<(f64, f64)>> {
        if g.len() != self.targets.len() {
            return Err(MipError::Encoding(format!(
                "star LP: generator width {} != alpha dim {}",
                g.len(),
                self.targets.len()
            )));
        }
        let expr: Vec<(AyCol, f64)> = self
            .targets
            .iter()
            .zip(g)
            .filter(|(_, w)| **w != 0.0)
            .map(|(&col, &w)| (col, w))
            .collect();
        if expr.is_empty() {
            return Ok(None);
        }
        let lo_duals = self
            .session
            .float_dual_for_expr(&expr, Sense::Minimize)
            .map_err(|e| MipError::Solver(e.to_string()))?;
        let hi_duals = self
            .session
            .float_dual_for_expr(&expr, Sense::Maximize)
            .map_err(|e| MipError::Solver(e.to_string()))?;
        let (Some(lo_d), Some(hi_d)) = (lo_duals, hi_duals) else {
            return Ok(None);
        };
        // The solver's sign convention is not guaranteed, and weak duality needs lambda >= 0.
        // Try the vector and its negation and keep the better SOUND bound of the two —
        // every candidate is independently valid, so this cannot cost correctness.
        let neg = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| -x).collect() };
        let best = |cands: [&[f64]; 2], neg_obj: bool| -> Option<f64> {
            let (cc, gg) = if neg_obj {
                (-c, g.iter().map(|v| -v).collect::<Vec<_>>())
            } else {
                (c, g.to_vec())
            };
            cands
                .into_iter()
                .filter_map(|lam| crate::star_dual::dual_bound_at(cc, &gg, a_rows, b, lam))
                .fold(None, |acc: Option<f64>, v| {
                    Some(acc.map_or(v, |a| a.max(v)))
                })
        };
        let lo_neg = neg(&lo_d);
        let hi_neg = neg(&hi_d);
        let Some(lo) = best([&lo_d, &lo_neg], false) else {
            return Ok(None);
        };
        let Some(hi_raw) = best([&hi_d, &hi_neg], true) else {
            return Ok(None);
        };
        let hi = -hi_raw;
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Ok(None);
        }
        // Outward pad sized to the evaluation's ACTUAL rounding error (O(k*m) fused
        // multiply-accumulates), not a flat constant. A flat 1e-9 turned a true-zero
        // pre-activation bound into -1.8e-9, which failed the `lo >= 0` stability test and
        // manufactured a split the exact path did not need.
        let terms = a_rows.len().saturating_mul(g.len()).saturating_add(g.len());
        #[allow(clippy::cast_precision_loss)]
        let pad = (terms + 4) as f64 * f64::EPSILON * (1.0 + lo.abs().max(hi.abs()));
        Ok(Some((lo - pad, hi + pad)))
    }
}
