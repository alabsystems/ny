// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #hull-root — a CERTIFIED per-neuron-hull lower bound for one hidden ReLU
//! layer over an input box (wave-6 C4, phase 1; diagnosis
//! `scratchpad/vbs/diag6/safenlp.md` §4.5 and Plan 1).
//!
//! # The problem
//!
//! After canonicalization the network is `Linear -> ReLU -> Linear`, the input
//! set is an axis-aligned box `B = [l, u]`, and the ONE unsafe row is excluded
//! by a strictly positive margin
//!
//! ```text
//! m(x) = d + sum_j c_j * relu(z_j(x)),   z_j(x) = w_j . x + b_j,
//! ```
//!
//! where `c = s^T W2` and `d = s^T b2 + const` for the sparse output objective
//! `s`. The property holds iff `min_{x in B} m(x) > 0`.
//!
//! # The certificate (the ONLY authority in this module)
//!
//! For ANY multipliers — `alpha_j` for `c_j > 0`, and any affine slope
//! `g_j in R^n` for `c_j < 0` — the following holds on the whole box:
//!
//! * `c_j >= 0`, `alpha in [0,1]`: `relu(z) >= alpha * z`, so
//!   `c_j relu(z_j(x)) >= c_j alpha_j z_j(x)`.
//! * `c_j < 0`: `relu(z(x)) - g.x = max(-g.x, (w-g).x + b)`, and each branch
//!   is a SEPARABLE linear function whose box maximum is a per-coordinate
//!   corner choice. So with
//!   `h = max( sum_i max(-g_i l_i, -g_i u_i),  b + sum_i max((w_i-g_i) l_i, (w_i-g_i) u_i) )`
//!   we have `relu(z(x)) <= g.x + h` on `B`, hence
//!   `c_j relu(z_j(x)) >= c_j (g_j.x + h_j)`.
//!
//! Summing gives an AFFINE minorant `K + A.x` of `m` on `B`, whose box minimum
//! is closed-form. [`certify_hull_lower_bound`] evaluates all of it with
//! outward (directed) rounding — every product/sum is an interval whose
//! endpoints are nudged one ulp outward after round-to-nearest — so the value
//! it returns is `<= min_B m` in REAL arithmetic for EVERY multiplier choice.
//! It never consults an LP, a pre-activation bound, or a stability
//! classification; a wrong multiplier costs tightness only. `alpha` is
//! clamped into `[0,1]` (NaN -> 0) and a non-finite `g` or any non-finite
//! intermediate makes it return `None` (fail closed).
//!
//! `c` and `d` are formed from the network's exact f64 parameters rounded
//! DOWNWARD. That is sound because `relu >= 0`: a smaller `c_j` can only
//! lower `c_j relu(z_j)`. So `m(x) >= d + sum_j c_j relu(z_j(x))` holds with the
//! stored (rounded) values, and the certificate is a bound for the true
//! real-arithmetic margin.
//!
//! By LP duality the best multipliers attain the per-neuron IDEAL hull
//! (Anderson et al.'s single-neuron formulation, Balas' extended form): the
//! diagnosis measured that value `> 0` on 30/115 safenlp gap rows with one
//! float LP each. With `g_j = theta_j (.) w_j`, `theta in [0,1]^n`, the best
//! majorant at a point is a fractional knapsack, which is what the proposer
//! LP below encodes compactly.
//!
//! # The proposers (NON-authoritative)
//!
//! 1. **Triangle** (always): the planet/triangle multipliers from plain-float
//!    pre-activation bounds — the baseline the input-split lane already has.
//! 2. **LP** (primary): the compact projected hull LP, solved by AY's float
//!    simplex through the T5 seam [`crate::lp_propose_certify`]. Its row duals
//!    ARE the multipliers (`alpha_j = y_Pj / c_j`, `theta_ji = y_Nji / |c_j|`),
//!    and they are handed straight to the certifier above. The rigorous-replay
//!    route is deliberately NOT used here: the LP's knapsack coefficients are
//!    plain float, so AY's Neumaier–Shcherbina bound would be rigorous for the
//!    encoded f64 LP but not for the network.
//! 3. **Supergradient** (fallback, only when the LP produced no certificate):
//!    a projected, diagonally scaled supergradient ascent on the concave dual
//!    function in `(alpha, theta)`; every improving iterate is re-certified.
//!
//! The report carries the proposer's plain-float LP value (`lp_value`,
//! telemetry) next to the certified value, so a dual-mapping or certifier
//! defect shows up as a gap between them rather than hiding in a null.
//!
//! # Phase 2
//!
//! [`bab`] (#hull-bab) branches on the concave neurons and certifies every
//! node with [`bab::certify_hull_node_lower_bound`], which is this module's
//! closed form plus one weak-duality term per split halfspace.

use std::time::{Duration, Instant};

use ay_milp::{Col, Model, SolveOpts};

use crate::lp_propose_certify::{propose_then_certify_min, ReplayPolicy};

/// Largest input dimension admitted (defensive; safenlp is 30).
pub const HULL_MAX_INPUT_DIM: usize = 1_024;
/// Largest hidden width admitted (defensive; safenlp is 128).
pub const HULL_MAX_HIDDEN_DIM: usize = 4_096;
/// Largest number of knapsack rows in the proposer LP (safenlp: ~1.6k).
pub const HULL_MAX_LP_HULL_ROWS: usize = 262_144;
/// Iteration cap for the supergradient fallback (it also honours the deadline).
pub const HULL_SUBGRADIENT_MAX_ITERS: usize = 2_000;

/// Why a problem could not be admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HullDecline {
    /// Binary64 round-to-nearest with gradual underflow is unavailable.
    UnsupportedEnvironment,
    /// A zero input, hidden or output dimension, or an empty objective.
    EmptyDimension,
    /// Parameter arrays inconsistent with the declared dimensions.
    ShapeMismatch,
    /// A NaN or infinite weight, bias, objective coefficient or box endpoint.
    NonFiniteParameter,
    /// An inverted box (`lower > upper`).
    InvalidBox,
    /// An objective index outside the output dimension.
    ObjectiveIndexOutOfRange,
    /// Beyond the defensive size caps.
    SizeCap,
    /// A caller-supplied cooperative resource checkpoint stopped this work.
    ResourceStopped,
}

impl HullDecline {
    /// Stable lowercase label for observables.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::UnsupportedEnvironment => "unsupported_environment",
            Self::EmptyDimension => "empty_dimension",
            Self::ShapeMismatch => "shape_mismatch",
            Self::NonFiniteParameter => "non_finite_parameter",
            Self::InvalidBox => "invalid_box",
            Self::ObjectiveIndexOutOfRange => "objective_index_out_of_range",
            Self::SizeCap => "size_cap",
            Self::ResourceStopped => "resource_stopped",
        }
    }
}

/// Every authoritative entry checks the current calling thread. AY and node
/// workers are separate boundaries; a guard at construction cannot qualify
/// arithmetic performed later under a different floating-point environment.
fn require_hull_environment() -> Result<(), HullDecline> {
    #[cfg(test)]
    if environment_tests::environment_refused_for_test() {
        return Err(HullDecline::UnsupportedEnvironment);
    }
    ny_core::require_f64_interval_proof_environment()
        .map_err(|_| HullDecline::UnsupportedEnvironment)
}

/// Proposer results remain untrusted. Recheck after its execution before any
/// returned candidate, telemetry arithmetic or prior positive bound is used.
fn guarded_hull_proposal<T>(propose: impl FnOnce() -> T) -> Result<T, HullDecline> {
    require_hull_environment()?;
    let result = propose();
    require_hull_environment()?;
    Ok(result)
}

/// One single-hidden-layer margin-minimization problem, in exact f64 data.
#[derive(Debug, Clone)]
pub struct HullMarginProblem {
    input_dim: usize,
    hidden_dim: usize,
    /// `hidden_dim x input_dim`, row-major, exact.
    w1: Vec<f64>,
    /// Exact first-layer bias.
    b1: Vec<f64>,
    /// Margin coefficient per hidden neuron, rounded DOWN (valid: relu >= 0).
    c: Vec<f64>,
    /// Margin constant, rounded DOWN.
    d: f64,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl HullMarginProblem {
    /// Build the problem `min_{x in [lower, upper]} objective . (W2 relu(W1 x + b1) + b2) + objective_constant`.
    ///
    /// `w1` is `hidden x input` row-major, `w2` is `output x hidden`
    /// row-major; `objective` is a sparse `(output index, coefficient)` list.
    /// Every value is taken as an EXACT real number; the caller must pass the
    /// network's exact parameters (f32 weights widen exactly; folded biases
    /// must be the exact f64 sidecar, never a rounded structural copy).
    ///
    /// # Errors
    /// A [`HullDecline`] naming the first violated admission condition.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        w1: Vec<f64>,
        b1: Vec<f64>,
        input_dim: usize,
        w2: &[f64],
        b2: &[f64],
        output_dim: usize,
        objective: &[(usize, f64)],
        objective_constant: f64,
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> Result<Self, HullDecline> {
        Self::new_polled(
            w1,
            b1,
            input_dim,
            w2,
            b2,
            output_dim,
            objective,
            objective_constant,
            lower,
            upper,
            &mut || true,
        )
    }

    /// Same original construction and rounding, with caller-owned stop polls.
    #[allow(clippy::too_many_arguments)]
    fn new_polled(
        w1: Vec<f64>,
        b1: Vec<f64>,
        input_dim: usize,
        w2: &[f64],
        b2: &[f64],
        output_dim: usize,
        objective: &[(usize, f64)],
        objective_constant: f64,
        lower: Vec<f64>,
        upper: Vec<f64>,
        poll: &mut dyn FnMut() -> bool,
    ) -> Result<Self, HullDecline> {
        if !poll() {
            return Err(HullDecline::ResourceStopped);
        }
        require_hull_environment()?;
        let hidden_dim = b1.len();
        if input_dim == 0 || hidden_dim == 0 || output_dim == 0 || objective.is_empty() {
            return Err(HullDecline::EmptyDimension);
        }
        if input_dim > HULL_MAX_INPUT_DIM || hidden_dim > HULL_MAX_HIDDEN_DIM {
            return Err(HullDecline::SizeCap);
        }
        let first_len = hidden_dim
            .checked_mul(input_dim)
            .ok_or(HullDecline::SizeCap)?;
        let second_len = output_dim
            .checked_mul(hidden_dim)
            .ok_or(HullDecline::SizeCap)?;
        if w1.len() != first_len
            || w2.len() != second_len
            || b2.len() != output_dim
            || lower.len() != input_dim
            || upper.len() != input_dim
        {
            return Err(HullDecline::ShapeMismatch);
        }
        let all_finite = w1
            .iter()
            .chain(&b1)
            .chain(w2)
            .chain(b2)
            .chain(&lower)
            .chain(&upper)
            .enumerate()
            .all(|(i, v)| (i % 1024 != 0 || poll()) && v.is_finite())
            && objective_constant.is_finite()
            && objective
                .iter()
                .enumerate()
                .all(|(i, &(_, s))| (i % 1024 != 0 || poll()) && s.is_finite());
        if !all_finite {
            if !poll() {
                return Err(HullDecline::ResourceStopped);
            }
            return Err(HullDecline::NonFiniteParameter);
        }
        if lower.iter().zip(&upper).any(|(l, u)| l > u) {
            return Err(HullDecline::InvalidBox);
        }
        for (i, &(k, _)) in objective.iter().enumerate() {
            if i % 1024 == 0 && !poll() {
                return Err(HullDecline::ResourceStopped);
            }
            if k >= output_dim {
                return Err(HullDecline::ObjectiveIndexOutOfRange);
            }
        }
        let (c, d) = form_margin_down(hidden_dim, w2, b2, objective, objective_constant, poll)?;
        if !poll() {
            return Err(HullDecline::ResourceStopped);
        }
        Ok(Self {
            input_dim,
            hidden_dim,
            w1,
            b1,
            c,
            d,
            lower,
            upper,
        })
    }

    /// Input dimension `n`.
    #[must_use]
    pub const fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Hidden width `H`.
    #[must_use]
    pub const fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// The (downward-rounded) margin coefficients `c`.
    #[must_use]
    pub fn margin_coefficients(&self) -> &[f64] {
        &self.c
    }

    /// The (downward-rounded) margin constant `d`.
    #[must_use]
    pub const fn margin_constant(&self) -> f64 {
        self.d
    }

    fn w_row(&self, j: usize) -> &[f64] {
        &self.w1[j * self.input_dim..(j + 1) * self.input_dim]
    }
}

/// Shared original-head reduction for construction and independent replay.
fn form_margin_down(
    hidden: usize,
    w2: &[f64],
    b2: &[f64],
    objective: &[(usize, f64)],
    constant: f64,
    poll: &mut dyn FnMut() -> bool,
) -> Result<(Vec<f64>, f64), HullDecline> {
    if !poll() {
        return Err(HullDecline::ResourceStopped);
    }
    let mut c = vec![0.0_f64; hidden];
    for (j, cj) in c.iter_mut().enumerate() {
        if !poll() {
            return Err(HullDecline::ResourceStopped);
        }
        let mut acc = 0.0_f64;
        for (i, &(k, s)) in objective.iter().enumerate() {
            if i % 1024 == 0 && !poll() {
                return Err(HullDecline::ResourceStopped);
            }
            acc = add_down(acc, product_down(s, w2[k * hidden + j]));
        }
        *cj = acc;
    }
    let mut d = constant;
    for (i, &(k, s)) in objective.iter().enumerate() {
        if i % 1024 == 0 && !poll() {
            return Err(HullDecline::ResourceStopped);
        }
        d = add_down(d, product_down(s, b2[k]));
    }
    if !d.is_finite() || c.iter().any(|v| !v.is_finite()) {
        return Err(HullDecline::NonFiniteParameter);
    }
    if !poll() {
        return Err(HullDecline::ResourceStopped);
    }
    Ok((c, d))
}

/// Multipliers for [`certify_hull_lower_bound`]. ANY values are admissible;
/// they only decide how tight the certified bound is.
#[derive(Debug, Clone, PartialEq)]
pub struct HullMultipliers {
    /// Per hidden neuron: the lower slope `alpha_j` used when `c_j > 0`
    /// (clamped into `[0,1]`, NaN treated as `0`).
    pub alpha: Vec<f64>,
    /// `hidden x input`, row-major: the majorant slope `g_j` used when
    /// `c_j < 0` (must be finite).
    pub g: Vec<f64>,
}

impl HullMultipliers {
    /// All-zero multipliers (valid; loose).
    #[must_use]
    pub fn zeros(problem: &HullMarginProblem) -> Self {
        Self {
            alpha: vec![0.0; problem.hidden_dim],
            g: vec![0.0; problem.hidden_dim * problem.input_dim],
        }
    }
}

// ---------------------------------------------------------------------------
// Directed rounding. Round-to-nearest is within half an ulp of the exact
// result, so one `next_down`/`next_up` step brackets it (also across a binade
// boundary, and on overflow: `next_down(+inf) = MAX <= exact`,
// `next_up(-inf) = -MAX >= exact`). Inputs are always finite here.
// ---------------------------------------------------------------------------

#[inline]
fn add_down(a: f64, b: f64) -> f64 {
    (a + b).next_down()
}

#[inline]
fn add_up(a: f64, b: f64) -> f64 {
    (a + b).next_up()
}

#[inline]
fn sub_down(a: f64, b: f64) -> f64 {
    (a - b).next_down()
}

#[inline]
fn sub_up(a: f64, b: f64) -> f64 {
    (a - b).next_up()
}

#[inline]
fn mul_down(a: f64, b: f64) -> f64 {
    (a * b).next_down()
}

#[inline]
fn mul_up(a: f64, b: f64) -> f64 {
    (a * b).next_up()
}

/// `s * w` rounded down, exact (no nudge) for the `+-1` objective weights a
/// relational row uses.
#[inline]
fn product_down(s: f64, w: f64) -> f64 {
    if s == 1.0 {
        w
    } else if s == -1.0 {
        -w
    } else {
        mul_down(s, w)
    }
}

/// `[lo, hi] * s` for an exact scalar `s`, rounded outward.
#[inline]
fn scale_interval(lo: f64, hi: f64, s: f64) -> (f64, f64) {
    if s >= 0.0 {
        (mul_down(lo, s), mul_up(hi, s))
    } else {
        (mul_down(hi, s), mul_up(lo, s))
    }
}

/// Upper bound (rounded up) on `h = max_{x in B} (relu(w.x + b) - g.x)`.
///
/// `relu(w.x + b) - g.x = max(-g.x, (w - g).x + b)`; each branch's box maximum
/// is a per-coordinate corner choice. `w - g` is carried as an outward interval
/// and its box maximum taken over all four interval/box corners (bilinear).
/// `None` on any non-finite intermediate (fail closed).
fn majorant_offset_upper(
    w: &[f64],
    b: f64,
    g: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> Option<f64> {
    let mut branch_zero = 0.0_f64;
    let mut branch_active = b;
    for i in 0..w.len() {
        let (l, u) = (lower[i], upper[i]);
        let neg_g = -g[i];
        branch_zero = add_up(branch_zero, mul_up(neg_g, l).max(mul_up(neg_g, u)));
        let delta_lo = sub_down(w[i], g[i]);
        let delta_hi = sub_up(w[i], g[i]);
        if !delta_lo.is_finite() || !delta_hi.is_finite() {
            return None;
        }
        let corner = mul_up(delta_lo, l)
            .max(mul_up(delta_lo, u))
            .max(mul_up(delta_hi, l))
            .max(mul_up(delta_hi, u));
        branch_active = add_up(branch_active, corner);
    }
    let h = branch_zero.max(branch_active);
    h.is_finite().then_some(h)
}

#[inline]
fn sanitize_alpha(alpha: f64) -> f64 {
    if alpha.is_nan() {
        0.0
    } else {
        alpha.clamp(0.0, 1.0)
    }
}

/// THE AUTHORITY: a certified lower bound on `min_{x in B} m(x)` in real
/// arithmetic, valid for ANY `multipliers` (see the module doc).
///
/// Returns `None` on a shape mismatch, a non-finite `g`, or any non-finite
/// intermediate — never an unsound number.
#[must_use]
pub fn certify_hull_lower_bound(
    problem: &HullMarginProblem,
    multipliers: &HullMultipliers,
) -> Option<f64> {
    certify_hull_lower_bound_polled(problem, multipliers, &mut || true)
}

/// Original checker; polls do not participate in any arithmetic or proof.
fn certify_hull_lower_bound_polled(
    problem: &HullMarginProblem,
    multipliers: &HullMultipliers,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    if !poll() {
        return None;
    }
    require_hull_environment().ok()?;
    let n = problem.input_dim;
    let h = problem.hidden_dim;
    if multipliers.alpha.len() != h || multipliers.g.len() != h.checked_mul(n)? {
        return None;
    }
    let mut coef_lo = vec![0.0_f64; n];
    let mut coef_hi = vec![0.0_f64; n];
    let mut constant = problem.d;
    for j in 0..h {
        if !poll() {
            return None;
        }
        let cj = problem.c[j];
        let w = problem.w_row(j);
        let bj = problem.b1[j];
        if cj > 0.0 {
            let alpha = sanitize_alpha(multipliers.alpha[j]);
            if alpha == 0.0 {
                continue;
            }
            // kappa = c_j * alpha_j as an outward interval (both >= 0).
            let (k_lo, k_hi) = (mul_down(cj, alpha), mul_up(cj, alpha));
            for i in 0..n {
                let (lo, hi) = scale_interval(k_lo, k_hi, w[i]);
                coef_lo[i] = add_down(coef_lo[i], lo);
                coef_hi[i] = add_up(coef_hi[i], hi);
            }
            constant = add_down(constant, scale_interval(k_lo, k_hi, bj).0);
        } else if cj < 0.0 {
            let g = &multipliers.g[j * n..(j + 1) * n];
            if g.iter().any(|v| !v.is_finite()) {
                return None;
            }
            let hj = majorant_offset_upper(w, bj, g, &problem.lower, &problem.upper)?;
            for i in 0..n {
                coef_lo[i] = add_down(coef_lo[i], mul_down(cj, g[i]));
                coef_hi[i] = add_up(coef_hi[i], mul_up(cj, g[i]));
            }
            // c_j < 0 and hj is an UPPER bound, so c_j * hj rounded down is a
            // lower bound on c_j * h_true.
            constant = add_down(constant, mul_down(cj, hj));
        }
    }
    if !constant.is_finite() || coef_lo.iter().chain(&coef_hi).any(|v| !v.is_finite()) {
        return None;
    }
    let mut total = constant;
    for i in 0..n {
        let (l, u) = (problem.lower[i], problem.upper[i]);
        let term = mul_down(coef_lo[i], l)
            .min(mul_down(coef_lo[i], u))
            .min(mul_down(coef_hi[i], l))
            .min(mul_down(coef_hi[i], u));
        total = add_down(total, term);
    }
    (poll() && total.is_finite()).then_some(total)
}

// ---------------------------------------------------------------------------
// Proposers (non-authoritative; plain float)
// ---------------------------------------------------------------------------

/// Plain-float role of a hidden neuron, used ONLY to shape proposals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    /// `c_j == 0`: contributes nothing.
    Inert,
    /// `z_j <= 0` on the box (float estimate).
    Dead,
    /// `z_j >= 0` on the box (float estimate): exact linear term.
    Live,
    /// Unstable, `c_j > 0`: convex term, lower slope `alpha_j`.
    PosUnstable,
    /// Unstable, `c_j < 0`: concave term, hull majorant.
    NegUnstable,
}

/// Plain-float `(min, max)` of each pre-activation over the box.
fn preactivation_bounds(problem: &HullMarginProblem) -> Vec<(f64, f64)> {
    preactivation_bounds_polled(problem, &mut || true).expect("unrestricted proposal scan")
}

fn preactivation_bounds_polled(
    problem: &HullMarginProblem,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Vec<(f64, f64)>> {
    if !poll() {
        return None;
    }
    let mut bounds = Vec::with_capacity(problem.hidden_dim);
    for j in 0..problem.hidden_dim {
        if !poll() {
            return None;
        }
        let w = problem.w_row(j);
        let mut lo = problem.b1[j];
        let mut hi = problem.b1[j];
        for (i, &wi) in w.iter().enumerate() {
            let a = wi * problem.lower[i];
            let b = wi * problem.upper[i];
            lo += a.min(b);
            hi += a.max(b);
        }
        bounds.push((lo, hi));
    }
    Some(bounds)
}

fn classify(problem: &HullMarginProblem, bounds: &[(f64, f64)]) -> Vec<Role> {
    problem
        .c
        .iter()
        .zip(bounds)
        .map(|(&cj, &(lo, hi))| {
            if cj == 0.0 {
                Role::Inert
            } else if hi <= 0.0 {
                Role::Dead
            } else if lo >= 0.0 {
                Role::Live
            } else if cj > 0.0 {
                Role::PosUnstable
            } else {
                Role::NegUnstable
            }
        })
        .collect()
}

/// Multipliers for the roles that need no search: Live -> exact
/// (`alpha = 1`, `g = w`); Dead/Inert -> zero. Unstable neurons start at zero.
fn fixed_role_multipliers(problem: &HullMarginProblem, roles: &[Role]) -> HullMultipliers {
    fixed_role_multipliers_polled(problem, roles, &mut || true)
        .expect("unrestricted multiplier proposal")
}

fn fixed_role_multipliers_polled(
    problem: &HullMarginProblem,
    roles: &[Role],
    poll: &mut dyn FnMut() -> bool,
) -> Option<HullMultipliers> {
    if !poll() {
        return None;
    }
    let n = problem.input_dim;
    let mut zero = |len| {
        let mut values = Vec::with_capacity(len);
        for i in 0..len {
            if i % 1024 == 0 && !poll() {
                return None;
            }
            values.push(0.0);
        }
        Some(values)
    };
    let mut m = HullMultipliers {
        alpha: zero(problem.hidden_dim)?,
        g: zero(problem.hidden_dim * n)?,
    };
    for (j, role) in roles.iter().enumerate() {
        if !poll() {
            return None;
        }
        if *role == Role::Live {
            m.alpha[j] = 1.0;
            m.g[j * n..(j + 1) * n].copy_from_slice(problem.w_row(j));
        }
    }
    Some(m)
}

/// A proposal policy for positive unstable ReLUs. Neither choice grants proof
/// authority: the original directed checker accepts arbitrary valid slopes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositiveLowerSlopePolicy {
    /// The existing triangle slope `u / (u - l)`.
    Proportional,
    /// The standard area rule: one when `u >= -l`, otherwise zero.
    Area,
}

/// The planet/triangle multipliers: slope `u / (u - l)` on both sides.
fn triangle_multipliers(
    problem: &HullMarginProblem,
    roles: &[Role],
    bounds: &[(f64, f64)],
) -> HullMultipliers {
    triangle_multipliers_polled(problem, roles, bounds, &mut || true)
        .expect("unrestricted triangle proposal")
}

fn triangle_multipliers_polled(
    problem: &HullMarginProblem,
    roles: &[Role],
    bounds: &[(f64, f64)],
    poll: &mut dyn FnMut() -> bool,
) -> Option<HullMultipliers> {
    triangle_multipliers_with_policy_polled(
        problem,
        roles,
        bounds,
        PositiveLowerSlopePolicy::Proportional,
        poll,
    )
}

fn triangle_multipliers_with_policy_polled(
    problem: &HullMarginProblem,
    roles: &[Role],
    bounds: &[(f64, f64)],
    positive_policy: PositiveLowerSlopePolicy,
    poll: &mut dyn FnMut() -> bool,
) -> Option<HullMultipliers> {
    let n = problem.input_dim;
    let mut m = fixed_role_multipliers_polled(problem, roles, poll)?;
    for (j, role) in roles.iter().enumerate() {
        if !poll() {
            return None;
        }
        let (lo, hi) = bounds[j];
        let slope = hi / (hi - lo);
        if !slope.is_finite() {
            continue;
        }
        match role {
            Role::PosUnstable => {
                m.alpha[j] = match positive_policy {
                    PositiveLowerSlopePolicy::Proportional => slope.clamp(0.0, 1.0),
                    PositiveLowerSlopePolicy::Area => {
                        if hi >= -lo {
                            1.0
                        } else {
                            0.0
                        }
                    }
                }
            }
            Role::NegUnstable => {
                let w = problem.w_row(j);
                for i in 0..n {
                    m.g[j * n + i] = slope * w[i];
                }
            }
            Role::Inert | Role::Dead | Role::Live => {}
        }
    }
    Some(m)
}

/// The compact projected hull LP (see the module doc) plus the row maps that
/// turn its duals back into multipliers.
struct HullLp {
    model: Model,
    objective: Vec<(Col, f64)>,
    /// Plain-float constant so that `LP value = constant + objective optimum`.
    constant: f64,
    /// Per neuron: the row index of `s_j - w_j.x >= b_j` (PosUnstable only).
    pos_rows: Vec<Option<usize>>,
    /// Per neuron: `(input index, row index)` of each knapsack row
    /// `t_ji + a_ji mu_j + w_ji x_i >= w_ji U_ji` (NegUnstable only).
    neg_rows: Vec<Vec<(usize, usize)>>,
}

/// Build the proposer LP:
///
/// ```text
/// min  sum_Live c_j w_j.x + sum_Pos c_j s_j + sum_Neg |c_j| (l1_j mu_j + sum_i t_ji)
/// s.t. s_j - w_j.x >= b_j                          (Pos)
///      t_ji + a_ji mu_j + w_ji x_i >= w_ji U_ji    (Neg, a_ji > 0)
///      x in B,  s, mu, t >= 0
/// ```
///
/// with `a_ji = |w_ji| (u_i - l_i)` and `U_ji` the maximizing corner of
/// `w_ji x_i`. At fixed `x` the Neg block equals `|c_j| (l1_j - conc_j(x))`,
/// the ideal single-neuron concave envelope as a fractional knapsack. `None`
/// when a coefficient is non-finite or the size cap is exceeded (AY's
/// `add_row` asserts finite coefficients, so this is checked first).
fn build_hull_lp(
    problem: &HullMarginProblem,
    roles: &[Role],
    bounds: &[(f64, f64)],
) -> Option<HullLp> {
    build_hull_lp_with_capture(problem, roles, bounds, None)
}

fn build_hull_lp_with_capture(
    problem: &HullMarginProblem,
    roles: &[Role],
    bounds: &[(f64, f64)],
    capture: Option<&mut primal_advice::Capture<'_>>,
) -> Option<HullLp> {
    let mut capture = capture.filter(|c| c.matches_source(problem));
    let n = problem.input_dim;
    let h = problem.hidden_dim;
    let mut model = Model::new();
    let x: Vec<Col> = (0..n)
        .map(|i| model.add_col(problem.lower[i], problem.upper[i]))
        .collect();
    if let Some(c) = capture.as_deref_mut() {
        for (i, &col) in x.iter().enumerate() {
            let _ = c.input(i, col);
        }
    }
    let mut linear_x = vec![0.0_f64; n];
    let mut objective: Vec<(Col, f64)> = Vec::new();
    let mut constant = problem.d;
    let mut pos_rows = vec![None; h];
    let mut neg_rows: Vec<Vec<(usize, usize)>> = vec![Vec::new(); h];
    let mut hull_rows = 0usize;
    for (j, role) in roles.iter().enumerate() {
        let cj = problem.c[j];
        let w = problem.w_row(j);
        let bj = problem.b1[j];
        match role {
            Role::Inert | Role::Dead => {
                if let Some(c) = capture.as_deref_mut() {
                    let _ = c.zero(j);
                }
            }
            Role::Live => {
                if let Some(c) = capture.as_deref_mut() {
                    let _ = c.zero(j);
                }
                for (acc, &wi) in linear_x.iter_mut().zip(w) {
                    *acc += cj * wi;
                }
                constant += cj * bj;
            }
            Role::PosUnstable => {
                let s = model.add_col(0.0, f64::INFINITY);
                if let Some(c) = capture.as_deref_mut() {
                    let _ = c.positive(j, s);
                }
                objective.push((s, cj));
                let mut coeffs = Vec::with_capacity(n + 1);
                coeffs.push((s, 1.0));
                for (i, &wi) in w.iter().enumerate() {
                    if wi != 0.0 {
                        coeffs.push((x[i], -wi));
                    }
                }
                let row = model.add_row(bj, f64::INFINITY, &coeffs);
                pos_rows[j] = Some(row.index());
            }
            Role::NegUnstable => {
                let abs_c = -cj;
                let l1 = bounds[j].1;
                let mu_weight = abs_c * l1;
                if !mu_weight.is_finite() {
                    return None;
                }
                let mu = model.add_col(0.0, f64::INFINITY);
                if let Some(c) = capture.as_deref_mut() {
                    let _ = c.negative(j, mu, l1);
                }
                objective.push((mu, mu_weight));
                constant -= mu_weight;
                for (i, &wi) in w.iter().enumerate() {
                    let (l, u) = (problem.lower[i], problem.upper[i]);
                    let a = wi.abs() * (u - l);
                    if a.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || !a.is_finite() {
                        continue;
                    }
                    let upper_corner = if wi > 0.0 { u } else { l };
                    let rhs = wi * upper_corner;
                    if !rhs.is_finite() {
                        return None;
                    }
                    hull_rows += 1;
                    if hull_rows > HULL_MAX_LP_HULL_ROWS {
                        return None;
                    }
                    let t = model.add_col(0.0, f64::INFINITY);
                    if let Some(c) = capture.as_deref_mut() {
                        let _ = c.term(j, t);
                    }
                    objective.push((t, abs_c));
                    let row = model.add_row(rhs, f64::INFINITY, &[(t, 1.0), (mu, a), (x[i], wi)]);
                    neg_rows[j].push((i, row.index()));
                }
            }
        }
    }
    for (i, &qi) in linear_x.iter().enumerate() {
        if !qi.is_finite() {
            return None;
        }
        if qi != 0.0 {
            objective.push((x[i], qi));
        }
    }
    if !constant.is_finite() || objective.iter().any(|&(_, w)| !w.is_finite()) {
        return None;
    }
    Some(HullLp {
        model,
        objective,
        constant,
        pos_rows,
        neg_rows,
    })
}

/// `num / den` clamped into `[0,1]`; `0` for any non-finite or non-positive
/// denominator case (a multiplier outside the dual cone only loosens).
#[inline]
fn unit_ratio(num: f64, den: f64) -> f64 {
    let r = num / den;
    if den > 0.0 && r.is_finite() {
        r.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Map the LP's row duals to certifier multipliers:
/// `alpha_j = y_Pj / c_j`, `g_ji = (y_Nji / |c_j|) w_ji`.
fn multipliers_from_duals(
    problem: &HullMarginProblem,
    roles: &[Role],
    lp: &HullLp,
    y: &[f64],
) -> Option<HullMultipliers> {
    if y.len() != lp.model.num_rows() {
        return None;
    }
    let n = problem.input_dim;
    let mut m = fixed_role_multipliers(problem, roles);
    for (j, role) in roles.iter().enumerate() {
        let cj = problem.c[j];
        match role {
            Role::PosUnstable => {
                let row = lp.pos_rows[j]?;
                m.alpha[j] = unit_ratio(y[row], cj);
            }
            Role::NegUnstable => {
                let w = problem.w_row(j);
                for &(i, row) in &lp.neg_rows[j] {
                    m.g[j * n + i] = unit_ratio(y[row], -cj) * w[i];
                }
            }
            Role::Inert | Role::Dead | Role::Live => {}
        }
    }
    Some(m)
}

/// Which proposer produced the certified bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HullBoundRoute {
    /// AY float-LP duals, certified by the closed form.
    Lp,
    /// Supergradient fallback iterate, certified by the closed form.
    Subgradient,
    /// Planet/triangle multipliers, certified by the closed form.
    Triangle,
}

impl HullBoundRoute {
    /// Stable lowercase label for observables.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Lp => "lp",
            Self::Subgradient => "subgradient",
            Self::Triangle => "triangle",
        }
    }
}

/// What happened to the LP proposer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HullLpStatus {
    /// Current-thread arithmetic assumptions were refused; no bound returned.
    UnsupportedEnvironment,
    /// Duals returned and certified.
    Solved,
    /// AY's float lane returned no duals (did not settle, or deadline).
    NoDuals,
    /// Duals returned, but no sign candidate certified.
    NoCertificate,
    /// AY rejected the model or objective.
    Error,
    /// Nothing unstable: the triangle multipliers are already exact.
    Empty,
    /// A non-finite coefficient or the row cap.
    NotBuilt,
    /// The deadline had passed before the LP could start.
    Skipped,
    /// A fresh triangle certificate already met the strict-positive goal.
    SkippedByGoal,
    /// Proposals ran, but the strict-positive publication deadline expired.
    DeadlineExpired,
}

impl HullLpStatus {
    /// Stable lowercase label for observables.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::UnsupportedEnvironment => "unsupported_environment",
            Self::Solved => "ok",
            Self::NoDuals => "no_duals",
            Self::NoCertificate => "no_certificate",
            Self::Error => "error",
            Self::Empty => "empty",
            Self::NotBuilt => "not_built",
            Self::Skipped => "skipped",
            Self::SkippedByGoal => "skipped_by_goal",
            Self::DeadlineExpired => "deadline_expired",
        }
    }
}

/// The caller's actual root objective. FullBound retains the historical
/// proposer sequence; StrictPositive may stop after a fresh positive triangle
/// certificate without constructing an LP that cannot change the decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HullBoundGoal {
    FullBound,
    StrictPositive,
}

/// The outcome of one hull root bound.
#[derive(Debug, Clone, PartialEq)]
pub struct HullRootReport {
    /// Unstable neurons with a nonzero margin coefficient (float estimate).
    pub unstable: usize,
    /// Of those, the concave (`c_j < 0`) ones — the only combinatorial ones.
    pub neg_unstable: usize,
    /// Of those, the convex (`c_j > 0`) ones.
    pub pos_unstable: usize,
    /// Plain-float LP value including the constant. TELEMETRY ONLY.
    pub lp_value: Option<f64>,
    /// The best CERTIFIED lower bound on the margin (authoritative).
    pub certified: Option<f64>,
    /// Which proposer's multipliers produced [`Self::certified`].
    pub route: Option<HullBoundRoute>,
    /// What happened to the LP proposer.
    pub lp_status: HullLpStatus,
    /// Supergradient fallback iterations run (0 when the LP certified).
    pub subgradient_iterations: usize,
}

impl HullRootReport {
    fn unsupported_environment() -> Self {
        Self {
            unstable: 0,
            neg_unstable: 0,
            pos_unstable: 0,
            lp_value: None,
            certified: None,
            route: None,
            lp_status: HullLpStatus::UnsupportedEnvironment,
            subgradient_iterations: 0,
        }
    }

    /// True iff the certified margin lower bound is strictly positive — the
    /// unsafe row is then empty on the box.
    #[must_use]
    pub fn verified(&self) -> bool {
        self.certified.is_some_and(|v| v > 0.0)
    }
}

/// Run the triangle, LP and (fallback) supergradient proposers and return the
/// best CERTIFIED margin lower bound. Never panics on admitted input; every
/// failure degrades to a looser certified value or `certified = None`.
#[must_use]
pub fn hull_root_bound(problem: &HullMarginProblem, deadline: Instant) -> HullRootReport {
    hull_root_bound_for_goal(problem, deadline, HullBoundGoal::FullBound)
}

/// Run the original root engine for an explicit goal on the supplied deadline.
/// This does not allocate a new allowance, change the full-bound API, or grant
/// a plain report independent original-model/publication authority.
#[must_use]
pub fn hull_root_bound_for_goal(
    problem: &HullMarginProblem,
    deadline: Instant,
    goal: HullBoundGoal,
) -> HullRootReport {
    if require_hull_environment().is_err() {
        return HullRootReport::unsupported_environment();
    }
    let expired = || {
        Instant::now() >= deadline
            || ny_core::instance_budget::deadline().is_some_and(|d| Instant::now() >= d)
    };
    let deadline_report = |negative, positive| HullRootReport {
        unstable: negative + positive,
        neg_unstable: negative,
        pos_unstable: positive,
        lp_value: None,
        certified: None,
        route: None,
        lp_status: HullLpStatus::Skipped,
        subgradient_iterations: 0,
    };
    if goal == HullBoundGoal::StrictPositive && expired() {
        return deadline_report(0, 0);
    }
    let bounds = preactivation_bounds(problem);
    let roles = classify(problem, &bounds);
    let neg_unstable = roles.iter().filter(|r| **r == Role::NegUnstable).count();
    let pos_unstable = roles.iter().filter(|r| **r == Role::PosUnstable).count();

    let triangle = triangle_multipliers(problem, &roles, &bounds);
    let mut best: Option<(f64, HullBoundRoute)> =
        certify_hull_lower_bound(problem, &triangle).map(|v| (v, HullBoundRoute::Triangle));
    if goal == HullBoundGoal::StrictPositive {
        if expired() {
            return deadline_report(neg_unstable, pos_unstable);
        }
        if let Some((value, route)) = best.filter(|(value, _)| value.is_finite() && *value > 0.0) {
            if require_hull_environment().is_err() {
                return HullRootReport::unsupported_environment();
            }
            if expired() {
                return deadline_report(neg_unstable, pos_unstable);
            }
            return HullRootReport {
                unstable: neg_unstable + pos_unstable,
                neg_unstable,
                pos_unstable,
                lp_value: None,
                certified: Some(value),
                route: Some(route),
                lp_status: HullLpStatus::SkippedByGoal,
                subgradient_iterations: 0,
            };
        }
    }
    let mut best_multipliers = triangle;

    let mut lp_value = None;
    let lp_status = if neg_unstable + pos_unstable == 0 {
        HullLpStatus::Empty
    } else if Instant::now() >= deadline {
        HullLpStatus::Skipped
    } else {
        match build_hull_lp(problem, &roles, &bounds) {
            None => HullLpStatus::NotBuilt,
            Some(lp) => {
                let remaining = deadline.saturating_duration_since(Instant::now());
                let opts = SolveOpts::new()
                    .with_time_limit(remaining.max(Duration::from_millis(1)))
                    .with_deadline(deadline);
                let mut certify = |y: &[f64]| -> Option<f64> {
                    require_hull_environment().ok()?;
                    let m = multipliers_from_duals(problem, &roles, &lp, y)?;
                    certify_hull_lower_bound(problem, &m)
                };
                match guarded_hull_proposal(|| {
                    propose_then_certify_min(
                        &lp.model,
                        &lp.objective,
                        &opts,
                        ReplayPolicy::Skip,
                        &mut certify,
                    )
                }) {
                    Err(_) => return HullRootReport::unsupported_environment(),
                    Ok(Err(_)) => HullLpStatus::Error,
                    Ok(Ok(outcome)) => {
                        lp_value = outcome.float_dual_value.map(|v| v + lp.constant);
                        if !outcome.duals_available {
                            HullLpStatus::NoDuals
                        } else if let (Some(value), Some(y)) =
                            (outcome.closed_form, outcome.best_duals.as_deref())
                        {
                            if let Some(m) = multipliers_from_duals(problem, &roles, &lp, y) {
                                if best.is_none_or(|(b, _)| value > b) {
                                    best = Some((value, HullBoundRoute::Lp));
                                    best_multipliers = m;
                                }
                            }
                            HullLpStatus::Solved
                        } else {
                            HullLpStatus::NoCertificate
                        }
                    }
                }
            }
        }
    };

    let mut subgradient_iterations = 0usize;
    let run_fallback = !matches!(lp_status, HullLpStatus::Solved | HullLpStatus::Empty)
        && !best.is_some_and(|(v, _)| v > 0.0);
    if run_fallback {
        let (improved, iterations) = subgradient_ascent(
            problem,
            &roles,
            &bounds,
            &best_multipliers,
            best.map(|(v, _)| v),
            deadline,
            HULL_SUBGRADIENT_MAX_ITERS,
        );
        subgradient_iterations = iterations;
        if let Some(value) = improved {
            best = Some((value, HullBoundRoute::Subgradient));
        }
    }

    if require_hull_environment().is_err() {
        return HullRootReport::unsupported_environment();
    }
    let late = goal == HullBoundGoal::StrictPositive && expired();
    HullRootReport {
        unstable: neg_unstable + pos_unstable,
        neg_unstable,
        pos_unstable,
        lp_value,
        certified: if late { None } else { best.map(|(v, _)| v) },
        route: if late { None } else { best.map(|(_, r)| r) },
        lp_status: if late {
            HullLpStatus::DeadlineExpired
        } else {
            lp_status
        },
        subgradient_iterations,
    }
}

/// Projected, diagonally scaled supergradient ascent on the concave dual
/// function `D(alpha, theta)` (`g_j = theta_j (.) w_j`), starting from
/// `start`. Every iterate whose plain-float value beats the best seen is
/// RE-CERTIFIED by [`certify_hull_lower_bound`]; only certified values leave
/// this function. Returns the best certified value if it improved on
/// `baseline`, and the iteration count.
fn subgradient_ascent(
    problem: &HullMarginProblem,
    roles: &[Role],
    bounds: &[(f64, f64)],
    start: &HullMultipliers,
    baseline: Option<f64>,
    deadline: Instant,
    max_iters: usize,
) -> (Option<f64>, usize) {
    let n = problem.input_dim;
    let h = problem.hidden_dim;
    let mut alpha = start.alpha.clone();
    // theta_ji recovered from g_ji = theta_ji w_ji (0 where w_ji == 0).
    let mut theta = vec![0.0_f64; h * n];
    for (j, role) in roles.iter().enumerate() {
        if *role == Role::NegUnstable {
            let w = problem.w_row(j);
            for i in 0..n {
                if w[i] != 0.0 {
                    let ratio = start.g[j * n + i] / w[i];
                    theta[j * n + i] = if ratio.is_finite() {
                        ratio.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                }
            }
        }
    }
    let mut candidate = start.clone();
    let mut best_certified = baseline;
    let mut best_float = f64::NEG_INFINITY;
    let mut improved = None;
    let mut coef = vec![0.0_f64; n];
    let mut x_hat = vec![0.0_f64; n];
    let mut iterations = 0usize;
    for k in 0..max_iters {
        if Instant::now() >= deadline {
            break;
        }
        iterations = k + 1;
        // Plain-float dual value and its box minimizer.
        coef.iter_mut().for_each(|v| *v = 0.0);
        let mut constant = problem.d;
        for (j, role) in roles.iter().enumerate() {
            let cj = problem.c[j];
            let w = problem.w_row(j);
            let bj = problem.b1[j];
            match role {
                Role::Live => {
                    for i in 0..n {
                        coef[i] += cj * w[i];
                    }
                    constant += cj * bj;
                }
                Role::PosUnstable => {
                    let a = alpha[j];
                    for i in 0..n {
                        coef[i] += cj * a * w[i];
                    }
                    constant += cj * a * bj;
                }
                Role::NegUnstable => {
                    // h(theta) = -sum_i theta_i w_i L_i + max(0, l1 - sum_i theta_i a_i)
                    let l1 = bounds[j].1;
                    let mut offset = 0.0_f64;
                    let mut used = 0.0_f64;
                    for i in 0..n {
                        let t = theta[j * n + i];
                        let (l, u) = (problem.lower[i], problem.upper[i]);
                        let lower_corner = if w[i] > 0.0 { l } else { u };
                        offset -= t * w[i] * lower_corner;
                        used += t * w[i].abs() * (u - l);
                        coef[i] += cj * t * w[i];
                    }
                    constant += cj * (offset + (l1 - used).max(0.0));
                }
                Role::Inert | Role::Dead => {}
            }
        }
        let mut value = constant;
        for i in 0..n {
            x_hat[i] = if coef[i] >= 0.0 {
                problem.lower[i]
            } else {
                problem.upper[i]
            };
            value += coef[i] * x_hat[i];
        }
        if value.is_finite() && value > best_float {
            best_float = value;
            for (j, role) in roles.iter().enumerate() {
                match role {
                    Role::PosUnstable => candidate.alpha[j] = alpha[j],
                    Role::NegUnstable => {
                        let w = problem.w_row(j);
                        for i in 0..n {
                            candidate.g[j * n + i] = theta[j * n + i] * w[i];
                        }
                    }
                    Role::Inert | Role::Dead | Role::Live => {}
                }
            }
            if let Some(certified) = certify_hull_lower_bound(problem, &candidate) {
                if best_certified.is_none_or(|b| certified > b) {
                    best_certified = Some(certified);
                    improved = Some(certified);
                }
            }
        }
        // Scaled supergradient step; every scaled component lies in [-1, 1].
        #[allow(clippy::cast_precision_loss)]
        let step = 0.5 / ((k + 1) as f64).sqrt();
        for (j, role) in roles.iter().enumerate() {
            let w = problem.w_row(j);
            let (l0, l1) = bounds[j];
            match role {
                Role::PosUnstable => {
                    let span = l1 - l0;
                    if span > 0.0 {
                        let mut z = problem.b1[j];
                        for i in 0..n {
                            z += w[i] * x_hat[i];
                        }
                        alpha[j] = (alpha[j] + step * z / span).clamp(0.0, 1.0);
                    }
                }
                Role::NegUnstable => {
                    let mut used = 0.0_f64;
                    for i in 0..n {
                        used +=
                            theta[j * n + i] * w[i].abs() * (problem.upper[i] - problem.lower[i]);
                    }
                    let knapsack_slack = if l1 - used > 0.0 { 1.0 } else { 0.0 };
                    for i in 0..n {
                        let (l, u) = (problem.lower[i], problem.upper[i]);
                        if w[i] == 0.0 || u <= l {
                            continue;
                        }
                        // u_ji(x_hat) in [0, 1]: normalized position toward
                        // the maximizing corner of w_ji x_i.
                        let position = if w[i] > 0.0 {
                            (x_hat[i] - l) / (u - l)
                        } else {
                            (u - x_hat[i]) / (u - l)
                        };
                        let t = &mut theta[j * n + i];
                        *t = (*t + step * (knapsack_slack - position)).clamp(0.0, 1.0);
                    }
                }
                Role::Inert | Role::Dead | Role::Live => {}
            }
        }
    }
    (improved, iterations)
}

// #hull-bab (wave-6 C4 phase 2): the ReLU-split BaB over the concave neurons,
// certified per node by an extension of `certify_hull_lower_bound`. A child
// module so it reuses this module's private proposer/rounding helpers without
// widening their visibility. Dark unless `bab.single_hidden_hull: bab`.
#[path = "single_hidden_hull_bab.rs"]
pub mod bab;

#[cfg(test)]
#[path = "single_hidden_hull_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "single_hidden_hull_environment_tests.rs"]
mod environment_tests;

#[cfg(test)]
#[path = "single_hidden_hull_goal_tests.rs"]
mod goal_tests;

/// Bounded caller-resource ingress. LP/BaB have not earned this profile.
#[path = "single_hidden_hull_resource.rs"]
pub mod resource;

/// Checked ideal-root majorants for the existing signed complete MIP.
#[path = "single_hidden_hull_mip_cuts.rs"]
pub mod mip_cuts;

#[path = "single_hidden_hull_primal_advice.rs"]
mod primal_advice;
