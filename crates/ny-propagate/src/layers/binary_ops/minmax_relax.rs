// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound linear relaxations for element-wise `max(x, y)` and `min(x, y)`.
//!
//! Both ops are piecewise-linear and convex (max) / concave (min), so they
//! admit exact convex-hull linear envelopes over an axis-aligned input box
//! `[lx, ux] × [ly, uy]`. These envelopes feed the CROWN backward pass exactly
//! like the McCormick envelope for `Mul`.
//!
//! # Derivation (max)
//!
//! `max(x, y) = y + relu(x - y)`. Let `t = x - y`, ranging over
//! `[t_l, t_u] = [lx - uy, ux - ly]`. Then `max(x, y) = y + relu(t)`.
//!
//! * If `t_u <= 0` then `x <= y` on the whole box, so `max = y` exactly.
//! * If `t_l >= 0` then `x >= y` on the whole box, so `max = x` exactly.
//! * Otherwise (`t_l < 0 < t_u`) we use the standard ReLU convex hull:
//!     - Upper: `relu(t) <= s * (t - t_l)` with slope `s = t_u / (t_u - t_l)`,
//!       giving `max(x, y) <= s*x + (1-s)*y - s*t_l`. The slope `s` and
//!       `1-s` are rounded to f32, perturbing the ideal coefficient pair, so the
//!       constant is NOT the ideal `-s*t_l`: instead it is recomputed (and
//!       rounded up) from the worst box corner using the ACTUAL f32 coefficients
//!       (`max_corner_residual`). Because `max(x,y) - (a*x + b*y)` is convex its
//!       maximum over the box is at a corner, so this keeps the plane provably
//!       `>= max` everywhere regardless of the f32 slope perturbation.
//!     - Lower: `relu(t) >= 0` and `relu(t) >= t`, giving the two sound
//!       lower planes `max(x, y) >= y` and `max(x, y) >= x`. Any convex
//!       combination of these two planes is also a sound lower bound because
//!       `max` is convex.
//!
//! # Derivation (min)
//!
//! `min(x, y) = -max(-x, -y)`. We reuse the max envelope on the negated box
//! and negate the resulting planes. Concretely the roles of lower/upper swap.
//!
//! Soundness is verified by dense sampling in the unit tests of `min.rs` /
//! `max.rs` and the proptest soundness suite.

use ndarray::{Array1, Array2};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::mul::BoundDir;
use crate::shape::{broadcast_flat_index_map, broadcast_shapes};
use crate::{contiguous_flat_slice, LinearBounds};

/// An affine plane `coeff_x * x + coeff_y * y + c`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Plane {
    pub coeff_x: f32,
    pub coeff_y: f32,
    pub c: f32,
}

impl Plane {
    /// Evaluate the plane at `(x, y)` in f64 for tie-break comparisons.
    #[inline]
    fn eval(&self, x: f32, y: f32) -> f64 {
        self.coeff_x as f64 * x as f64 + self.coeff_y as f64 * y as f64 + self.c as f64
    }
}

/// Lower and upper affine envelopes for an element-wise binary op.
#[derive(Clone, Copy, Debug)]
pub(super) struct Envelope {
    /// `z >= lower(x, y)` for all `(x, y)` in the box.
    pub lower: Plane,
    /// `z <= upper(x, y)` for all `(x, y)` in the box.
    pub upper: Plane,
}

#[cfg(test)]
impl Envelope {
    /// Evaluate the lower plane at `(x, y)` (f32). Used by sibling-module
    /// relaxation soundness tests (e.g. `atan2_relax`).
    #[inline]
    pub(super) fn lower_eval(&self, x: f32, y: f32) -> f32 {
        self.lower.eval(x, y) as f32
    }

    /// Evaluate the upper plane at `(x, y)` (f32).
    #[inline]
    pub(super) fn upper_eval(&self, x: f32, y: f32) -> f32 {
        self.upper.eval(x, y) as f32
    }
}

/// Largest value of `f(x, y) - (coeff_x*x + coeff_y*y)` over the four corners
/// of the box `[lx, ux] × [ly, uy]`, evaluated in f64.
///
/// For a convex `f` (e.g. `max`) the linear-corrected residual `f - linear` is
/// convex, so its maximum over the box is attained at a corner. The caller adds
/// (then `next_up_f32`-rounds) this as the upper-plane constant `c`, guaranteeing
/// `coeff_x*x + coeff_y*y + c >= f(x, y)` everywhere on the box regardless of the
/// f32 rounding applied to `coeff_x`/`coeff_y`. Each `coeff as f64 * v as f64`
/// product is exact (both operands are f32-exact-in-f64), so the only rounding is
/// the final f64 subtraction, which the caller's `next_up_f32` absorbs.
#[inline]
fn max_corner_residual(
    coeff_x: f32,
    coeff_y: f32,
    lx: f32,
    ux: f32,
    ly: f32,
    uy: f32,
    f: impl Fn(f32, f32) -> f64,
) -> f64 {
    let cx = coeff_x as f64;
    let cy = coeff_y as f64;
    let mut best = f64::NEG_INFINITY;
    for &(x, y) in &[(lx, ly), (lx, uy), (ux, ly), (ux, uy)] {
        let resid = f(x, y) - (cx * x as f64 + cy * y as f64);
        if resid > best {
            best = resid;
        }
    }
    best
}

/// Sound linear envelope for `z = max(x, y)` over `[lx, ux] × [ly, uy]`.
///
/// Returns `None` if any bound is non-finite (caller keeps the IBP fallback).
#[inline]
pub(super) fn max_envelope(lx: f32, ux: f32, ly: f32, uy: f32) -> Option<Envelope> {
    if !lx.is_finite() || !ux.is_finite() || !ly.is_finite() || !uy.is_finite() {
        return None;
    }

    // t = x - y ranges over [t_l, t_u].
    let t_l = lx - uy;
    let t_u = ux - ly;

    // Degenerate: x <= y everywhere => max = y.
    if t_u <= 0.0 {
        let p = Plane {
            coeff_x: 0.0,
            coeff_y: 1.0,
            c: 0.0,
        };
        return Some(Envelope { lower: p, upper: p });
    }
    // Degenerate: x >= y everywhere => max = x.
    if t_l >= 0.0 {
        let p = Plane {
            coeff_x: 1.0,
            coeff_y: 0.0,
            c: 0.0,
        };
        return Some(Envelope { lower: p, upper: p });
    }

    // Mixed: t_l < 0 < t_u. Upper = convex-hull (ReLU secant) envelope.
    let denom = (t_u - t_l) as f64;
    // denom = t_u - t_l > 0 because t_u > 0 > t_l here.
    let s = (t_u as f64 / denom) as f32; // slope in (0, 1)
    let coeff_x = s;
    let coeff_y = 1.0 - s; // computed in f32; perturbs the ideal (s, 1-s) pair

    // The mathematical upper plane is `s*x + (1-s)*y - s*t_l`, exact at the two
    // box corners where the ReLU secant touches `relu(t)`. But `coeff_x`/`coeff_y`
    // are the *rounded* f32 coefficients, so the ideal constant `-s*t_l` no longer
    // guarantees the plane sits above `max` at those corners (the audit's sub-1e-4
    // violation). Recompute the constant directly against the ACTUAL f32 coeffs so
    // the bias absorbs any f32 slope perturbation.
    //
    // `g(x,y) = max(x,y) - (coeff_x*x + coeff_y*y)` is convex (max convex, linear
    // part affine), so its maximum over the box is attained at a vertex. Setting
    // `c >= max_corners g` makes `coeff_x*x + coeff_y*y + c >= max(x,y)` on the
    // whole box for ANY coefficient choice. The corner residuals are computed in
    // f64 (each f32->f64 product is exact); `next_up_f32` then guarantees the
    // stored f32 `c` is >= the true f64 residual even after the cast-to-f32 round.
    let c_upper = max_corner_residual(coeff_x, coeff_y, lx, ux, ly, uy, |x, y| x.max(y) as f64);
    let upper = Plane {
        coeff_x,
        coeff_y,
        c: next_up_f32(c_upper as f32),
    };

    // Lower: the better of the two sound planes z >= x and z >= y is chosen at
    // CROWN time via weight-sign selection. Here we expose `z >= x` as the
    // canonical lower plane; the caller may also use `z >= y`. We pick the
    // plane with the larger value at the box midpoint to keep a single tight
    // lower plane (still sound, since both planes are sound everywhere).
    // Bit-identical plane-selection anchors: f32::midpoint rounds differently at overflow/subnormal edges.
    #[allow(clippy::manual_midpoint)]
    let xm = 0.5 * (lx + ux);
    #[allow(clippy::manual_midpoint)]
    let ym = 0.5 * (ly + uy);
    let plane_x = Plane {
        coeff_x: 1.0,
        coeff_y: 0.0,
        c: 0.0,
    };
    let plane_y = Plane {
        coeff_x: 0.0,
        coeff_y: 1.0,
        c: 0.0,
    };
    let lower = if plane_x.eval(xm, ym) >= plane_y.eval(xm, ym) {
        plane_x
    } else {
        plane_y
    };

    Some(Envelope { lower, upper })
}

/// Sound linear envelope for `z = min(x, y)` over `[lx, ux] × [ly, uy]`.
///
/// Uses `min(x, y) = -max(-x, -y)`: build the max envelope on the negated box
/// and negate the planes, swapping lower/upper.
///
/// Returns `None` if any bound is non-finite (caller keeps the IBP fallback).
#[inline]
pub(super) fn min_envelope(lx: f32, ux: f32, ly: f32, uy: f32) -> Option<Envelope> {
    // Negated box: -x in [-ux, -lx], -y in [-uy, -ly].
    let env = max_envelope(-ux, -lx, -uy, -ly)?;
    // min(x,y) = -max(-x,-y). If max(-x,-y) <= U(-x,-y) = a*(-x)+b*(-y)+c
    // then min(x,y) = -max >= -(a*(-x)+b*(-y)+c) = a*x + b*y - c. So the
    // max upper plane becomes the min lower plane (negate constant only;
    // coeffs flip sign twice). Likewise max lower -> min upper.
    let lower = Plane {
        coeff_x: env.upper.coeff_x,
        coeff_y: env.upper.coeff_y,
        c: next_down_f32(-(env.upper.c as f64) as f32),
    };
    let upper = Plane {
        coeff_x: env.lower.coeff_x,
        coeff_y: env.lower.coeff_y,
        c: next_up_f32(-(env.lower.c as f64) as f32),
    };
    Some(Envelope { lower, upper })
}

/// Widen one accumulated ENVELOPE CONSTANT outward by a certified bound on its
/// own floating-point error (#minmax-envelope-const).
///
/// Twin of `widen_mccormick_constant` (`binary_ops/mul/mod.rs`,
/// `binary_ops/matmul/crown_dense.rs`, `binary_ops/bilinear/linear.rs`), which
/// folds the identical shape — an incoming f32 bias plus one `w * plane_const`
/// product per inner-loop step — for the McCormick envelope. Same formula, same
/// constants, so the four certificates cannot drift apart; duplicated rather than
/// shared because each copy is private to its own module.
///
/// # The defect this closes
///
/// [`propagate_minmax_linear_binary`] gained the COEFFICIENT certificate
/// (`γ^{f32}_d · S`, published through
/// [`LinearBounds::new_or_conservative_with_err`]) but kept folding the envelope
/// CONSTANT as `const += w as f64 * pl.c as f64` in plain round-to-nearest f64,
/// published through the directed f32 cast alone. A relaxation has a coefficient
/// term AND a constant term and BOTH need the certificate: certifying one half
/// leaves the published relation exactly as unsound as before, and worse than the
/// pre-fix state in that it now *claims* a certificate.
///
/// Recursive summation of `adds + 1` summands satisfies
/// `|fl(S) − S| ≤ γ^{f64}_{adds} · Σ|summand|` (Higham, *Accuracy and Stability*,
/// Thm 3.1). The charge scales with the MAGNITUDE sum, not with `|S|`, so it does
/// NOT vanish under cancellation — which is precisely when a result-scaled 1-ULP
/// directed cast under-counts it. Worked case: terms `+2^60`, 4096 × `−1`,
/// `−2^60` fold to exactly `0.0` in f64 where the truth is `−4096`; the
/// magnitude-sum charge covers it (`γ^{f64}_{4098} · 2.3e18 ≈ 1.0e6 ≫ 4096`),
/// a ULP of the `0.0` result does not.
///
/// # The certificate
///
/// ```text
/// S_safe        = magnitude_sum      / (1 - gamma^f64_{adds+1})
/// S_terms_safe  = term_magnitude_sum / (1 - gamma^f64_{adds+1})
/// err           = gamma^f64_{adds+1} * S_safe  +  gamma^f32_3 * S_terms_safe
/// out           = down(accumulated - err)   /   up(accumulated + err)
/// ```
///
/// The `1/(1 - gamma)` inflation mirrors `conv2d::bound`: `magnitude_sum` is
/// itself an f64 round-to-nearest sum of non-negative values and can fall SHORT of
/// the true magnitude sum by its own accumulation deficit, so charging against the
/// raw figure could under-bound the very error it is meant to cover. The final
/// directed f64 step absorbs the rounding of the widening subtract/add.
///
/// The `γ^{f32}_3` term against the PLANE terms is SLACK here rather than repair:
/// every `envelope_fn` in the tree already rounds its plane constant outward
/// ([`max_envelope`] emits `0.0` on its exact branches and `next_up_f32(…)` on the
/// secant upper plane; [`min_envelope`] negates through
/// `next_down_f32`/`next_up_f32`; `atan2_relax` likewise), and `w` is read
/// straight out of the incoming relation with no arithmetic, so `f64::from(w) *
/// f64::from(pl.c)` is EXACT. It is applied anyway, at the same figure the
/// McCormick twins use, because `envelope_fn` is a caller-supplied closure: a
/// future envelope whose constant is *not* pre-rounded would otherwise reach this
/// fold uncharged and silently. Erring outward costs a verdict; erring inward
/// costs a false `unsat`.
///
/// # Direction
///
/// `err >= 0` always (`gamma >= 0`, both magnitude sums are sums of `abs` values,
/// `inflate >= 1`), so a `Lower` constant only ever moves DOWN and an `Upper`
/// constant only ever moves UP. `adds == 0` returns the input bit-identically: no
/// f64 addition ran and no plane was consumed, so there is genuinely nothing to
/// charge. A NaN anywhere fails CLOSED to the infinite constant.
fn widen_envelope_constant(
    accumulated: f64,
    magnitude_sum: f64,
    term_magnitude_sum: f64,
    adds: usize,
    dir: BoundDir,
) -> f64 {
    let trivial = match dir {
        BoundDir::Lower => f64::NEG_INFINITY,
        BoundDir::Upper => f64::INFINITY,
    };
    if adds == 0 {
        // `accumulated` is the exact f64 image of the incoming f32 bias.
        return accumulated;
    }
    let gamma_acc = crate::layers::linear::crown_single_gamma_n_f64(adds.saturating_add(1));
    let inflate = if gamma_acc < 1.0 {
        1.0 / (1.0 - gamma_acc)
    } else {
        f64::INFINITY
    };
    // Up to 2 f32 roundings per term inside a caller-supplied `envelope_fn`, plus
    // one step of slack — the same figure the McCormick twins charge.
    let gamma_plane = crate::layers::linear::crown_single_gamma_n_f32(3);
    let err = gamma_acc * (magnitude_sum * inflate) + gamma_plane * (term_magnitude_sum * inflate);
    if err.is_nan() {
        return trivial;
    }
    if err <= 0.0 {
        return accumulated;
    }
    let widened = match dir {
        BoundDir::Lower => accumulated - err,
        BoundDir::Upper => accumulated + err,
    };
    if widened.is_nan() {
        return trivial;
    }
    match dir {
        BoundDir::Lower => widened.next_down(),
        BoundDir::Upper => widened.next_up(),
    }
}

/// Running envelope CONSTANT fold that carries its own Higham certificate
/// (#minmax-envelope-const).
///
/// Replaces the bare `const_lower += w as f64 * pl.c as f64` chain in
/// [`propagate_minmax_linear_binary`], the CONSTANT half of the same relaxation
/// whose COEFFICIENT half that function already certifies.
///
/// `magnitude_sum` carries the incoming bias (the f64 accumulation charge is
/// against EVERY summand); `term_magnitude_sum` excludes it (the per-term f32
/// plane-rounding charge applies only to the plane terms). `adds` counts the f64
/// additions actually performed, so a term-free fold charges exactly zero and is
/// published bit-identically to the historical directed-cast result.
struct EnvelopeConstFold {
    accumulated: f64,
    magnitude_sum: f64,
    term_magnitude_sum: f64,
    adds: usize,
    dir: BoundDir,
}

impl EnvelopeConstFold {
    /// Seed the fold with the incoming relation's f32 bias (exact in f64).
    #[inline]
    fn new(bias: f32, dir: BoundDir) -> Self {
        let accumulated = f64::from(bias);
        Self {
            accumulated,
            magnitude_sum: accumulated.abs(),
            term_magnitude_sum: 0.0,
            adds: 0,
            dir,
        }
    }

    /// Fold in one `w * plane_const` term and record its magnitude.
    ///
    /// The product is EXACT in f64 (an f32 × f32 product needs ≤ 48 < 53
    /// significand bits), so the ONLY rounding introduced here is the f64 `+=`
    /// — which is precisely what [`widen_envelope_constant`] charges for.
    #[inline]
    fn add_term(&mut self, w: f32, plane_const: f32) {
        let term = f64::from(w) * f64::from(plane_const);
        self.accumulated += term;
        let magnitude = term.abs();
        self.magnitude_sum += magnitude;
        self.term_magnitude_sum += magnitude;
        self.adds += 1;
    }

    /// Publish the certified-outward f64 constant.
    #[inline]
    fn finish(&self) -> f64 {
        widen_envelope_constant(
            self.accumulated,
            self.magnitude_sum,
            self.term_magnitude_sum,
            self.adds,
            self.dir,
        )
    }
}

/// Shared CROWN backward driver for element-wise `max`/`min`.
///
/// `envelope_fn` returns the sound lower/upper affine planes for a single
/// `(x, y)` element box; when it returns `None` for any required element the
/// whole op falls back to IBP (signalled via `UnsupportedOp`).
///
/// Mirrors `MulBinaryLayer::propagate_linear_binary`: broadcast-aware, splits
/// the McCormick-style constant into a separate bias channel carried entirely
/// on `bounds_a` (with `bounds_b` bias zeroed) so the DAG counts it once.
///
/// # Certified coefficient error (#vnncomp-aw-soundness)
///
/// Each published coefficient cell is built by an f32 `+=` chain
/// (`w * plane_coeff` accumulated over the broadcast axis), so the STORED f32
/// coefficient is not the exact real-arithmetic one. The sound-envelope proof
/// in `max_envelope` / `max_corner_residual` covers only the per-element PLANE
/// (a plane is valid for whatever f32 `coeff_x`/`coeff_y` it stores); it says
/// nothing about the f32 accumulation that COMPOSES those planes with the
/// incoming weights. Under cancellation that accumulation error is unbounded
/// relative to the result: `w = [-2^24, -1, 2^24]` with `coeff_x = 1` stores
/// `0` where the exact coefficient is `-1`.
///
/// This driver therefore accumulates `S = Σ |w| · |plane_coeff|` (the MAGNITUDE
/// sum, exactly, in f64 — an f32×f32 product needs ≤ 48 < 53 bits) alongside
/// each coefficient and publishes the Higham certificate
/// `|stored − exact| ≤ γ^{f32}_d · S` through
/// [`LinearBounds::new_or_conservative_with_err`], exactly as
/// `MulBinaryLayer::propagate_linear_binary` (`binary_ops/mul/mod.rs`) and
/// `MatMulLayer::propagate_linear_binary` (`binary_ops/matmul/crown_dense.rs`)
/// do. `concretize` then discharges it OUTWARD as
/// `Σ_j err[i,j] · max(|in_l[j]|, |in_u[j]|)`. Publishing through
/// `new_or_conservative` instead (error `None`) claims the coefficients are
/// EXACT and is the defect this replaces.
///
/// # Certified envelope CONSTANT (#minmax-envelope-const)
///
/// The coefficient certificate above is only HALF the relaxation. The envelope
/// CONSTANT is the other half, and it was folded as
/// `const += w as f64 * pl.c as f64` in plain round-to-nearest f64 and published
/// through the directed f32 cast alone — an allowance scaled by the RESULT, which
/// collapses toward a subnormal under exactly the cancellation that makes the
/// accumulation error largest. Both constants now fold through
/// [`EnvelopeConstFold`], which publishes the Higham magnitude-sum certificate
/// `γ^{f64}_{adds+1} · Σ|summand|` its McCormick twins do. Certifying only the
/// coefficients would leave the relation as unsound as before while *claiming* a
/// certificate.
///
/// Weight-sign selection: for the output lower bound, a non-negative incoming
/// weight uses the lower plane of `z`, a negative weight uses the upper plane
/// (and vice versa for the output upper bound). This is the same rule the
/// `Mul` "Middle" relaxation uses and is exactly the requirement that the
/// resulting affine form remains a valid CROWN lower/upper bound.
pub(super) fn propagate_minmax_linear_binary<F>(
    bounds: &LinearBounds,
    input_a_bounds: &BoundedTensor,
    input_b_bounds: &BoundedTensor,
    op_name: &str,
    envelope_fn: F,
) -> Result<(LinearBounds, LinearBounds)>
where
    F: Fn(f32, f32, f32, f32) -> Option<Envelope>,
{
    let n = bounds.num_inputs();
    let num_outputs = bounds.num_outputs();
    let n_a = input_a_bounds.len();
    let n_b = input_b_bounds.len();

    let output_shape = broadcast_shapes(input_a_bounds.shape(), input_b_bounds.shape())
        .ok_or_else(|| NyError::ShapeMismatch {
            expected: input_a_bounds.shape().to_vec(),
            got: input_b_bounds.shape().to_vec(),
        })?;
    let broadcast_n: usize = checked_shape_product(&output_shape).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "{op_name} broadcast shape product overflows: {output_shape:?}"
        ))
    })?;
    if broadcast_n != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![broadcast_n],
        });
    }

    let a_idx_map = broadcast_flat_index_map(&output_shape, input_a_bounds.shape());
    let b_idx_map = broadcast_flat_index_map(&output_shape, input_b_bounds.shape());

    let a_lower_flat = contiguous_flat_slice(input_a_bounds.lower());
    let a_upper_flat = contiguous_flat_slice(input_a_bounds.upper());
    let b_lower_flat = contiguous_flat_slice(input_b_bounds.lower());
    let b_upper_flat = contiguous_flat_slice(input_b_bounds.upper());

    let mut lower_a_a = Array2::<f32>::zeros((num_outputs, n_a));
    let mut lower_a_b = Array2::<f32>::zeros((num_outputs, n_b));
    let mut upper_a_a = Array2::<f32>::zeros((num_outputs, n_a));
    let mut upper_a_b = Array2::<f32>::zeros((num_outputs, n_b));
    // Certified-error accumulators (#vnncomp-aw-soundness), mirroring
    // `MulBinaryLayer::propagate_linear_binary` and `crown_dense.rs`:
    // `S = Σ |w| · |plane_coeff|` per coefficient cell. Accumulated in f64 with
    // EXACT products (f32×f32 needs ≤ 48 < 53 bits), so the only rounding is the
    // f64 additions — a relative deficit of at most γ^{f64}_d ≈ d·2^-53, which is
    // more than covered by the one-f32-ULP `next_up_f32` applied at publication
    // (relative ≥ 2^-24) below.
    let mut lower_s_a = Array2::<f64>::zeros((num_outputs, n_a));
    let mut lower_s_b = Array2::<f64>::zeros((num_outputs, n_b));
    let mut upper_s_a = Array2::<f64>::zeros((num_outputs, n_a));
    let mut upper_s_b = Array2::<f64>::zeros((num_outputs, n_b));
    // f64 bias accumulation to avoid catastrophic cancellation.
    let mut lower_b_total = Array1::<f64>::zeros(num_outputs);
    let mut upper_b_total = Array1::<f64>::zeros(num_outputs);

    // Per-cell f32 accumulation DEPTH. The inner `for j in 0..n` loop below does
    // exactly ONE unconditional `+=` into `lower_a_a[[out_idx, a_idx_map[j]]]`
    // (and likewise for the b side) per iteration, and `out_idx` is fixed by the
    // outer loop, so the number of f32 additions landing in a given cell is
    // exactly `#{ j : a_idx_map[j] == a_idx }`. Higham (Accuracy and Stability,
    // Thm 3.1) then gives `|fl(Σ_{t≤d} w_t·c_t) − Σ w_t·c_t| ≤ γ^{f32}_d · Σ|w_t·c_t|`
    // for a depth-`d` inner product. `n` (the loop extent, the constant the Mul
    // sibling uses) is also sound because `d ≤ n`; the exact depth is used here
    // because it is the same certificate with the true `d`, and `n` over-charges
    // by up to a factor `n` on the common NON-broadcast case where `d == 1`.
    let mut a_depth = vec![0usize; n_a];
    let mut b_depth = vec![0usize; n_b];
    for j in 0..n {
        a_depth[a_idx_map[j]] += 1;
        b_depth[b_idx_map[j]] += 1;
    }

    // Precompute per-element envelopes once (independent of output row). If any
    // element has a non-finite box, refuse and let the caller use IBP.
    let mut envelopes: Vec<Envelope> = Vec::with_capacity(n);
    for j in 0..n {
        let a_idx = a_idx_map[j];
        let b_idx = b_idx_map[j];
        let lx = a_lower_flat[a_idx];
        let ux = a_upper_flat[a_idx];
        let ly = b_lower_flat[b_idx];
        let uy = b_upper_flat[b_idx];
        let env = envelope_fn(lx, ux, ly, uy).ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "{op_name} CROWN backward requires finite input bounds"
            ))
        })?;
        envelopes.push(env);
    }

    for out_idx in 0..num_outputs {
        // Certified ENVELOPE-CONSTANT folds (#minmax-envelope-const) — the CONSTANT
        // half of the same relaxation whose COEFFICIENT half is certified below.
        // The bare `+= w as f64 * pl.c as f64` chain these replace published a
        // round-to-nearest f64 sum of `n` terms with NO error term, relying on the
        // directed f32 cast at publication; that allowance is scaled by the RESULT
        // and so collapses exactly when cancellation makes the accumulation error
        // largest. See `widen_envelope_constant`.
        let mut const_lower = EnvelopeConstFold::new(bounds.lower_b()[out_idx], BoundDir::Lower);
        let mut const_upper = EnvelopeConstFold::new(bounds.upper_b()[out_idx], BoundDir::Upper);

        for j in 0..n {
            let w_lower = bounds.lower_a()[[out_idx, j]];
            let w_upper = bounds.upper_a()[[out_idx, j]];
            let a_idx = a_idx_map[j];
            let b_idx = b_idx_map[j];
            let env = envelopes[j];

            // Output lower bound: w >= 0 uses z's lower plane, w < 0 uses upper.
            let pl = if w_lower >= 0.0 { env.lower } else { env.upper };
            lower_a_a[[out_idx, a_idx]] += w_lower * pl.coeff_x;
            lower_a_b[[out_idx, b_idx]] += w_lower * pl.coeff_y;
            lower_s_a[[out_idx, a_idx]] += (w_lower as f64).abs() * (pl.coeff_x as f64).abs();
            lower_s_b[[out_idx, b_idx]] += (w_lower as f64).abs() * (pl.coeff_y as f64).abs();
            const_lower.add_term(w_lower, pl.c);

            // Output upper bound: w >= 0 uses z's upper plane, w < 0 uses lower.
            let pu = if w_upper >= 0.0 { env.upper } else { env.lower };
            upper_a_a[[out_idx, a_idx]] += w_upper * pu.coeff_x;
            upper_a_b[[out_idx, b_idx]] += w_upper * pu.coeff_y;
            upper_s_a[[out_idx, a_idx]] += (w_upper as f64).abs() * (pu.coeff_x as f64).abs();
            upper_s_b[[out_idx, b_idx]] += (w_upper as f64).abs() * (pu.coeff_y as f64).abs();
            const_upper.add_term(w_upper, pu.c);
        }

        lower_b_total[out_idx] = const_lower.finish();
        upper_b_total[out_idx] = const_upper.finish();
    }

    // The f64 totals reaching this point have ALREADY been widened outward for
    // their own accumulation and plane-rounding error by `widen_envelope_constant`
    // (via `EnvelopeConstFold::finish`); the directed cast here covers ONLY the
    // f64 -> f32 narrowing, which is NOT sufficient on its own
    // (#minmax-envelope-const, and see the identical note in
    // `matmul/crown_dense.rs`).
    let lower_b_f32 = lower_b_total.mapv(|v| next_down_f32(v as f32));
    let upper_b_f32 = upper_b_total.mapv(|v| next_up_f32(v as f32));

    // Certified coefficient error: `|stored − exact| ≤ γ^{f32}_d · S`, rounded UP
    // to a sound f32 and carried so `concretize` penalizes OUTWARD. `next_up_f32`
    // adds a full f32 ULP (relative ≥ 2^-24), which dominates both the ≤ 0.5-ULP
    // f64→f32 cast-down and the ≤ (d+4)·2^-53 relative deficit of the f64 `S` sum
    // and `γ` evaluation, so the published f32 is ≥ the exact `γ_d · S_true`.
    // That inequality is COMPLETE, not asymptotic: `crown_single_gamma_n_f32`
    // returns +∞ once `d·2^-24 ≥ 1`, so whenever `γ` is finite `d < 2^24`, and the
    // half-ULP left over after the cast (≥ 2^-25 relative) exceeds
    // `(d+4)·2^-53 < 2^-25`. A non-finite `γ` yields a non-finite err, which
    // `new_or_conservative_with_err` sanitizes to `+∞` and concretize degrades the
    // row on — fail-closed.
    let gamma_a: Vec<f64> = a_depth
        .iter()
        .map(|&d| crate::layers::linear::crown_single_gamma_n_f32(d))
        .collect();
    let gamma_b: Vec<f64> = b_depth
        .iter()
        .map(|&d| crate::layers::linear::crown_single_gamma_n_f32(d))
        .collect();
    let mut lower_a_a_err = Array2::<f32>::zeros((num_outputs, n_a));
    let mut upper_a_a_err = Array2::<f32>::zeros((num_outputs, n_a));
    for i in 0..num_outputs {
        for (j, &g) in gamma_a.iter().enumerate() {
            lower_a_a_err[[i, j]] = next_up_f32((g * lower_s_a[[i, j]]) as f32);
            upper_a_a_err[[i, j]] = next_up_f32((g * upper_s_a[[i, j]]) as f32);
        }
    }
    let mut lower_a_b_err = Array2::<f32>::zeros((num_outputs, n_b));
    let mut upper_a_b_err = Array2::<f32>::zeros((num_outputs, n_b));
    for i in 0..num_outputs {
        for (j, &g) in gamma_b.iter().enumerate() {
            lower_a_b_err[[i, j]] = next_up_f32((g * lower_s_b[[i, j]]) as f32);
            upper_a_b_err[[i, j]] = next_up_f32((g * upper_s_b[[i, j]]) as f32);
        }
    }

    let bounds_a = LinearBounds::new_or_conservative_with_err(
        lower_a_a,
        lower_b_f32,
        upper_a_a,
        upper_b_f32,
        lower_a_a_err,
        upper_a_a_err,
    )?;
    let bounds_b = LinearBounds::new_or_conservative_with_err(
        lower_a_b,
        Array1::zeros(num_outputs),
        upper_a_b,
        Array1::zeros(num_outputs),
        lower_a_b_err,
        upper_a_b_err,
    )?;

    Ok((bounds_a, bounds_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    /// One ULP of `z` (the spacing to the next representable f32 above |z|).
    /// Used as the soundness tolerance instead of the old loose `1e-4`.
    fn ulp_of(z: f32) -> f64 {
        let a = z.abs();
        (next_up_f32(a) as f64 - a as f64).max(f64::MIN_POSITIVE)
    }

    /// Dense-sample the box and assert lower(x,y) <= f(x,y) <= upper(x,y) to
    /// within a TIGHT ~1-ULP tolerance (not the old 1e-4 that masked the
    /// sub-1e-4 envelope violation). The envelope planes are evaluated in f64
    /// exactly as the CROWN driver accumulates them, so a 1-ULP slack only
    /// covers the unavoidable final f32->f64 representation of the inputs.
    fn assert_envelope_sound(
        env: Envelope,
        lx: f32,
        ux: f32,
        ly: f32,
        uy: f32,
        f: impl Fn(f32, f32) -> f32,
    ) {
        let steps = 40;
        for i in 0..=steps {
            let x = lx + (ux - lx) * (i as f32 / steps as f32);
            for j in 0..=steps {
                let y = ly + (uy - ly) * (j as f32 / steps as f32);
                let z = f(x, y) as f64;
                let lo = env.lower.eval(x, y);
                let hi = env.upper.eval(x, y);
                let tol = ulp_of(z as f32);
                assert!(
                    lo <= z + tol,
                    "lower {lo} > f {z} (slack {}) at ({x},{y}) box [{lx},{ux}]x[{ly},{uy}]",
                    lo - z
                );
                assert!(
                    hi >= z - tol,
                    "upper {hi} < f {z} (slack {}) at ({x},{y}) box [{lx},{ux}]x[{ly},{uy}]",
                    z - hi
                );
            }
        }
    }

    fn boxes() -> Vec<(f32, f32, f32, f32)> {
        vec![
            (-3.0, 2.0, -1.0, 4.0),   // mixed / mixed
            (0.0, 5.0, -2.0, 3.0),    // mixed
            (-5.0, -1.0, -4.0, -2.0), // both negative
            (1.0, 6.0, 2.0, 8.0),     // both positive, x mostly below y
            (5.0, 10.0, 0.0, 3.0),    // x dominates (x>=y everywhere)
            (0.0, 1.0, 5.0, 6.0),     // y dominates (x<=y everywhere)
            (-2.0, 2.0, -2.0, 2.0),   // symmetric
            (2.0, 2.0, -1.0, 3.0),    // x is a point
            (0.0, 0.0, 0.0, 0.0),     // degenerate point
        ]
    }

    #[test]
    fn max_envelope_is_sound_dense() {
        for (lx, ux, ly, uy) in boxes() {
            let env = max_envelope(lx, ux, ly, uy).unwrap();
            assert_envelope_sound(env, lx, ux, ly, uy, |x, y| x.max(y));
        }
    }

    #[test]
    fn min_envelope_is_sound_dense() {
        for (lx, ux, ly, uy) in boxes() {
            let env = min_envelope(lx, ux, ly, uy).unwrap();
            assert_envelope_sound(env, lx, ux, ly, uy, |x, y| x.min(y));
        }
    }

    /// Deterministic xorshift RNG returning f64 in [0, 1).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Strict per-corner check: the upper plane must be `>= max` and the lower
    /// plane must be `<= max` at every box corner with NO tolerance at all
    /// (the corners are where the relaxation is tight and the f32 slope
    /// perturbation is worst — this is exactly the envelope-violation the audit
    /// flagged). Evaluated in f64 the way the CROWN driver accumulates planes.
    fn assert_corners_strict(env: Envelope, lx: f32, ux: f32, ly: f32, uy: f32, is_max: bool) {
        for &(x, y) in &[(lx, ly), (lx, uy), (ux, ly), (ux, uy)] {
            let z = if is_max { x.max(y) } else { x.min(y) } as f64;
            let lo = env.lower.eval(x, y);
            let hi = env.upper.eval(x, y);
            assert!(
                hi >= z,
                "upper {hi} < {z} at corner ({x},{y}) box [{lx},{ux}]x[{ly},{uy}] (violation {})",
                z - hi
            );
            assert!(
                lo <= z,
                "lower {lo} > {z} at corner ({x},{y}) box [{lx},{ux}]x[{ly},{uy}] (violation {})",
                lo - z
            );
        }
    }

    /// Many random boxes biased toward the regime that triggers the f32 slope
    /// perturbation: wide magnitude range, mixed sign (so `t_l < 0 < t_u`), and
    /// large coordinate magnitudes that amplify the `(s, 1-s)` rounding error.
    /// Asserts both the strict-corner property and a dense ~1-ULP interior check
    /// for max AND min.
    #[test]
    fn minmax_envelope_random_boxes_tight_ulp() {
        let mut rng = Rng(0x1234_5678_9abc_def0);
        let mut mixed_seen = 0u64;
        for _ in 0..200_000 {
            // Scale spans 0.01 .. 1e6 to stress coefficient-rounding error.
            let scale = 10f64.powf(rng.next() * 8.0 - 2.0);
            let lx = ((rng.next() * 2.0 - 1.0) * scale) as f32;
            let ux = lx + (rng.next() * scale) as f32;
            let ly = ((rng.next() * 2.0 - 1.0) * scale) as f32;
            let uy = ly + (rng.next() * scale) as f32;
            if !lx.is_finite() || !ux.is_finite() || !ly.is_finite() || !uy.is_finite() {
                continue;
            }
            // Count how many boxes land in the mixed (relaxed) regime.
            if (lx - uy) < 0.0 && (ux - ly) > 0.0 {
                mixed_seen += 1;
            }

            let max_env = max_envelope(lx, ux, ly, uy).unwrap();
            let min_env = min_envelope(lx, ux, ly, uy).unwrap();

            // Strict (zero-tolerance) corner soundness: this is the property the
            // old 1e-4 tolerance masked.
            assert_corners_strict(max_env, lx, ux, ly, uy, true);
            assert_corners_strict(min_env, lx, ux, ly, uy, false);

            // Dense interior soundness at ~1 ULP.
            assert_envelope_sound(max_env, lx, ux, ly, uy, |x, y| x.max(y));
            assert_envelope_sound(min_env, lx, ux, ly, uy, |x, y| x.min(y));
        }
        // Sanity: the random generator actually exercises the relaxed branch.
        assert!(
            mixed_seen > 1_000,
            "expected many mixed-regime boxes, saw only {mixed_seen}"
        );
    }

    /// Hand-picked box reproducing the audit's worst observed violation regime:
    /// a tiny upper-x endpoint next to a large negative y endpoint, where the
    /// f32 `(s, 1-s)` pair previously pushed the upper plane below the true max
    /// by ~1.5e-5 at the corner. Asserts the fixed plane is sound there.
    #[test]
    fn upper_plane_sound_at_audit_worst_corner() {
        let (lx, ux, ly, uy) = (
            -286.696_08f32,
            -0.003_692_627f32,
            -374.136_05f32,
            -134.931_73f32,
        );
        let env = max_envelope(lx, ux, ly, uy).unwrap();
        assert_corners_strict(env, lx, ux, ly, uy, true);
        // The specific corner the probe flagged.
        let (x, y) = (ux, ly);
        let z = x.max(y) as f64;
        let hi = env.upper.eval(x, y);
        assert!(
            hi >= z,
            "regression: upper {hi} < max {z} at audit corner ({x},{y}) (violation {})",
            z - hi
        );
    }

    #[test]
    fn non_finite_box_returns_none() {
        assert!(max_envelope(f32::NEG_INFINITY, 1.0, 0.0, 1.0).is_none());
        assert!(min_envelope(0.0, 1.0, 0.0, f32::INFINITY).is_none());
        assert!(max_envelope(0.0, f32::NAN, 0.0, 1.0).is_none());
    }

    fn bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        let n = lower.len();
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper.to_vec()).unwrap(),
        )
        .unwrap()
    }

    /// Drive the full CROWN backward through an identity spec and confirm the
    /// concretized scalar bounds enclose f at every sampled box corner+interior.
    fn assert_driver_sound(a: &BoundedTensor, b: &BoundedTensor, is_max: bool) {
        let n = a.len();
        // Identity incoming spec: each output picks one element directly.
        let ident = LinearBounds::identity(n);
        let (lb_a, lb_b) = if is_max {
            propagate_minmax_linear_binary(&ident, a, b, "MaxBinary", max_envelope).unwrap()
        } else {
            propagate_minmax_linear_binary(&ident, a, b, "MinBinary", min_envelope).unwrap()
        };

        let al = a.lower();
        let au = a.upper();
        let bl = b.lower();
        let bu = b.upper();
        let steps = 25;
        for k in 0..n {
            // bias lives entirely on lb_a; lb_b bias is zero.
            for i in 0..=steps {
                let x = al[k] + (au[k] - al[k]) * (i as f32 / steps as f32);
                for j in 0..=steps {
                    let y = bl[k] + (bu[k] - bl[k]) * (j as f32 / steps as f32);
                    let z = if is_max { x.max(y) } else { x.min(y) };
                    // Reconstruct the affine lower/upper form for output k.
                    let lo =
                        lb_a.lower_a()[[k, k]] * x + lb_b.lower_a()[[k, k]] * y + lb_a.lower_b()[k];
                    let hi =
                        lb_a.upper_a()[[k, k]] * x + lb_b.upper_a()[[k, k]] * y + lb_a.upper_b()[k];
                    let tol = 1e-3 * (1.0 + z.abs());
                    assert!(lo <= z + tol, "driver lower {lo} > {z} at k={k} ({x},{y})");
                    assert!(hi >= z - tol, "driver upper {hi} < {z} at k={k} ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn driver_max_sound() {
        let a = bt(&[-3.0, 0.0, 5.0, 1.0], &[2.0, 5.0, 10.0, 6.0]);
        let b = bt(&[-1.0, -2.0, 0.0, 2.0], &[4.0, 3.0, 3.0, 8.0]);
        assert_driver_sound(&a, &b, true);
    }

    #[test]
    fn driver_min_sound() {
        let a = bt(&[-3.0, 0.0, 5.0, 1.0], &[2.0, 5.0, 10.0, 6.0]);
        let b = bt(&[-1.0, -2.0, 0.0, 2.0], &[4.0, 3.0, 3.0, 8.0]);
        assert_driver_sound(&a, &b, false);
    }

    /// Exercise the w < 0 plane-selection branch by using a negated incoming
    /// spec (-I). For output k the lower form must bound -f from below and the
    /// upper form must bound -f from above.
    fn assert_driver_sound_negated(a: &BoundedTensor, b: &BoundedTensor, is_max: bool) {
        let n = a.len();
        let neg_ident = {
            let mut la = Array2::<f32>::zeros((n, n));
            let mut ua = Array2::<f32>::zeros((n, n));
            for k in 0..n {
                la[[k, k]] = -1.0;
                ua[[k, k]] = -1.0;
            }
            LinearBounds::new(la, Array1::zeros(n), ua, Array1::zeros(n)).unwrap()
        };
        let (lb_a, lb_b) = if is_max {
            propagate_minmax_linear_binary(&neg_ident, a, b, "MaxBinary", max_envelope).unwrap()
        } else {
            propagate_minmax_linear_binary(&neg_ident, a, b, "MinBinary", min_envelope).unwrap()
        };
        let (al, au, bl, bu) = (a.lower(), a.upper(), b.lower(), b.upper());
        let steps = 25;
        for k in 0..n {
            for i in 0..=steps {
                let x = al[k] + (au[k] - al[k]) * (i as f32 / steps as f32);
                for j in 0..=steps {
                    let y = bl[k] + (bu[k] - bl[k]) * (j as f32 / steps as f32);
                    let f = if is_max { x.max(y) } else { x.min(y) };
                    let z = -f;
                    let lo =
                        lb_a.lower_a()[[k, k]] * x + lb_b.lower_a()[[k, k]] * y + lb_a.lower_b()[k];
                    let hi =
                        lb_a.upper_a()[[k, k]] * x + lb_b.upper_a()[[k, k]] * y + lb_a.upper_b()[k];
                    let tol = 1e-3 * (1.0 + z.abs());
                    assert!(lo <= z + tol, "neg lower {lo} > {z} at k={k} ({x},{y})");
                    assert!(hi >= z - tol, "neg upper {hi} < {z} at k={k} ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn driver_max_negated_spec_sound() {
        let a = bt(&[-3.0, 0.0, 5.0, 1.0], &[2.0, 5.0, 10.0, 6.0]);
        let b = bt(&[-1.0, -2.0, 0.0, 2.0], &[4.0, 3.0, 3.0, 8.0]);
        assert_driver_sound_negated(&a, &b, true);
    }

    #[test]
    fn driver_min_negated_spec_sound() {
        let a = bt(&[-3.0, 0.0, 5.0, 1.0], &[2.0, 5.0, 10.0, 6.0]);
        let b = bt(&[-1.0, -2.0, 0.0, 2.0], &[4.0, 3.0, 3.0, 8.0]);
        assert_driver_sound_negated(&a, &b, false);
    }
}

#[cfg(test)]
mod cancellation_audit_tests {
    use super::*;

    /// Audit for the failure mode that broke the McCormick planes
    /// (`docs/MULBINARY_MCCORMICK_F32_CANCELLATION_2026-07-28.md`): an interval
    /// whose endpoints differ by more than 2^24 in magnitude, so that an f32
    /// `(l - u) + u` round-trip loses `l` entirely.
    ///
    /// `minmax_envelope_random_boxes_tight_ulp` draws BOTH endpoints from one
    /// scale, so it does not target within-interval asymmetry. This does: `|l|`
    /// is driven far below `ulp(u)` on one or both operands.
    ///
    /// Expected to PASS — `max_corner_residual` derives the constant in f64 from
    /// the ALREADY-f32-rounded coefficients and rounds outward, which is exactly
    /// the certification the McCormick constructor lacked. This test records that
    /// as measured rather than assumed.
    #[test]
    fn minmax_envelopes_enclose_under_extreme_within_interval_asymmetry() {
        let cases: &[(f32, f32, f32, f32)] = &[
            (-1.0, 16_777_216.0, 1.0, 100.0),
            (-1.0, 1.6e9, -1.0, 1.6e9),
            (-1e-6, 1e6, -1e-6, 1e6),
            (-0.026_068_168, 1_153_335.9, 7_220.358, 148_963.05),
            (-1e-30, 1e30, -1e-30, 1e30),
            (1e-20, 3.4e38, -3.4e38, -1e-20),
            (-3.4e38, 3.4e38, -1.0, 1.0),
        ];
        for &(lx, ux, ly, uy) in cases {
            let Some(env) = max_envelope(lx, ux, ly, uy) else {
                continue; // degenerate/decided box: no envelope needed
            };
            for &(x, y) in &[(lx, ly), (lx, uy), (ux, ly), (ux, uy)] {
                let truth = f64::from(x.max(y));
                let lo = env.lower.eval(x, y);
                let hi = env.upper.eval(x, y);
                assert!(
                    lo <= truth,
                    "max LOWER plane above truth: x=[{lx:e},{ux:e}] y=[{ly:e},{uy:e}] \
                     at ({x:e},{y:e}): {lo:e} > {truth:e}"
                );
                assert!(
                    hi >= truth,
                    "max UPPER plane below truth: x=[{lx:e},{ux:e}] y=[{ly:e},{uy:e}] \
                     at ({x:e},{y:e}): {hi:e} < {truth:e}"
                );
            }
            let Some(menv) = min_envelope(lx, ux, ly, uy) else {
                continue;
            };
            for &(x, y) in &[(lx, ly), (lx, uy), (ux, ly), (ux, uy)] {
                let truth = f64::from(x.min(y));
                let lo = menv.lower.eval(x, y);
                let hi = menv.upper.eval(x, y);
                assert!(
                    lo <= truth,
                    "min LOWER plane above truth: x=[{lx:e},{ux:e}] y=[{ly:e},{uy:e}] \
                     at ({x:e},{y:e}): {lo:e} > {truth:e}"
                );
                assert!(
                    hi >= truth,
                    "min UPPER plane below truth: x=[{lx:e},{ux:e}] y=[{ly:e},{uy:e}] \
                     at ({x:e},{y:e}): {hi:e} < {truth:e}"
                );
            }
        }
    }
}

#[cfg(test)]
mod coefficient_certificate_tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    fn bt(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        let n = lower.len();
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower.to_vec()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper.to_vec()).unwrap(),
        )
        .unwrap()
    }

    /// COUNTEREXAMPLE for the uncertified MinBinary/MaxBinary/Atan2 CROWN
    /// backward: two large opposite-sign incoming weights cancel inside the f32
    /// `+=` chain, so the STORED coefficient is not the exact one, and the
    /// published relation (which claimed exactness by carrying no error matrix)
    /// concretizes to a lower bound ABOVE the true minimum.
    ///
    /// Construction (all arithmetic exact and checkable by hand):
    /// * `A = x` is a single element BROADCAST against `B`'s three elements, so
    ///   all three incoming weights accumulate into the SAME coefficient cell.
    /// * `x ∈ [10, 20]`, `y_j ∈ [0, 1]` ⇒ `t_l = lx − uy = 9 ≥ 0`, so
    ///   `max_envelope` takes its exact `x ≥ y` branch and every plane is
    ///   `Plane { coeff_x: 1, coeff_y: 0, c: 0 }`.
    /// * incoming weights `w = [−2^24, −1, +2^24]`.
    ///
    /// Exact coefficient on `x`:  `−2^24 − 1 + 2^24 = −1`.
    /// f32 accumulation:          `fl(fl(0 − 2^24) − 1) + 2^24`
    ///                          `= fl(−16777217) + 2^24`  (ulp(2^24) = 2, ties-to-even)
    ///                          `= −2^24 + 2^24 = 0`.
    /// The op's true value is `Σ_j w_j · max(x, y_j) = −x ∈ [−20, −10]`, minimum
    /// `−20`. With the coefficient stored as `0` and no error carried, the
    /// concretized lower bound is `0` — it EXCLUDES the true value by 20, i.e. a
    /// false `unsat` is available from this relation. This assertion fails on the
    /// pre-fix code and passes once the `γ^{f32}_d · S` certificate is published
    /// (`S = 2^24 + 1 + 2^24`, `d = 3` ⇒ `err ≈ 6.0`, penalty `6.0 · 20 = 120`).
    #[test]
    fn max_binary_broadcast_cancellation_stays_enclosed() {
        const P24: f32 = 16_777_216.0; // 2^24

        let a = bt(&[10.0], &[20.0]);
        let b = bt(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);

        let mut la = Array2::<f32>::zeros((1, 3));
        la[[0, 0]] = -P24;
        la[[0, 1]] = -1.0;
        la[[0, 2]] = P24;
        let ua = la.clone();
        let spec = LinearBounds::new(la, Array1::zeros(1), ua, Array1::zeros(1)).unwrap();

        let (bounds_a, bounds_b) =
            propagate_minmax_linear_binary(&spec, &a, &b, "MaxBinary", max_envelope).unwrap();

        // Precondition of the counterexample: the f32 chain really did cancel.
        assert_eq!(
            bounds_a.lower_a()[[0, 0]],
            0.0,
            "counterexample precondition: the f32 += chain must cancel to 0"
        );

        // The certificate must exist and must cover the 1.0 coefficient deviation.
        let err_lo = bounds_a
            .lower_a_err()
            .expect("Min/Max CROWN backward must publish a coefficient certificate");
        assert!(
            f64::from(err_lo[[0, 0]]) >= 1.0,
            "certificate {} does not cover |stored 0 − exact −1| = 1",
            err_lo[[0, 0]]
        );

        // End-to-end: the published relation must lower-bound the true minimum −20.
        let lo =
            bounds_a.concretize_sound(&a).lower()[0] + bounds_b.concretize_sound(&b).lower()[0];
        assert!(
            lo <= -20.0,
            "published CROWN lower bound {lo} is ABOVE the true minimum -20 of \
             sum_j w_j*max(x,y_j) = -x over x in [10,20]; the f32 coefficient \
             accumulation error is uncertified"
        );
    }

    /// The B-side counterexample, driven through `min_envelope`, and through the
    /// `bounds_b` publication path (which carries a ZERO bias, so nothing else can
    /// absorb the coefficient error).
    ///
    /// Here `B = y` is the single element BROADCAST against `A`'s three, so all
    /// three incoming weights land in the same B coefficient cell. With
    /// `x_j ∈ [10, 20]` and `y ∈ [0, 1]` we have `u_y = 1 ≤ l_x = 10`, so
    /// `min_envelope` takes its exact `y ≤ x` branch and every plane is
    /// `{ coeff_x: 0, coeff_y: 1, c: ~0 }`.
    ///
    /// With `w = [-2^24, -1, +2^24]` the B cell accumulates
    /// `fl(fl(0 - 2^24) - 1) + 2^24 = 0` where the exact coefficient is `-1`. The
    /// op's true value is `Σ_j w_j · min(x_j, y) = -y ∈ [-1, 0]`, minimum `-1`,
    /// but the relation published without an error matrix concretizes to `0`.
    #[test]
    fn min_binary_broadcast_cancellation_on_b_side_stays_enclosed() {
        const P24: f32 = 16_777_216.0;

        let a = bt(&[10.0, 10.0, 10.0], &[20.0, 20.0, 20.0]);
        let b = bt(&[0.0], &[1.0]);

        let mut la = Array2::<f32>::zeros((1, 3));
        la[[0, 0]] = -P24;
        la[[0, 1]] = -1.0;
        la[[0, 2]] = P24;
        let ua = la.clone();
        let spec = LinearBounds::new(la, Array1::zeros(1), ua, Array1::zeros(1)).unwrap();

        let (bounds_a, bounds_b) =
            propagate_minmax_linear_binary(&spec, &a, &b, "MinBinary", min_envelope).unwrap();

        // Precondition: the exact-branch planes put everything on the y coefficient
        // and the f32 chain there really did cancel.
        assert_eq!(
            bounds_b.lower_a()[[0, 0]],
            0.0,
            "counterexample precondition: the B-side f32 += chain must cancel to 0"
        );

        // Both published relations must carry a certificate; `None` is the defect.
        assert!(
            bounds_a.lower_a_err().is_some() && bounds_a.upper_a_err().is_some(),
            "MinBinary bounds_a published without a coefficient certificate"
        );
        let err_b = bounds_b
            .lower_a_err()
            .expect("MinBinary bounds_b published without a coefficient certificate");
        assert!(
            f64::from(err_b[[0, 0]]) >= 1.0,
            "certificate {} does not cover |stored 0 - exact -1| = 1",
            err_b[[0, 0]]
        );

        // End-to-end: the published relation must lower-bound the true minimum -1.
        let lo =
            bounds_a.concretize_sound(&a).lower()[0] + bounds_b.concretize_sound(&b).lower()[0];
        assert!(
            lo <= -1.0,
            "published CROWN lower bound {lo} is ABOVE the true minimum -1 of \
             sum_j w_j*min(x_j,y) = -y over y in [0,1]; the f32 coefficient \
             accumulation error is uncertified"
        );
    }

    /// COUNTEREXAMPLE for the CONSTANT half of the same relaxation
    /// (#minmax-envelope-const): the session that added the coefficient
    /// certificate above left `const += w as f64 * pl.c as f64` folding in plain
    /// round-to-nearest f64, published through the directed f32 cast alone.
    ///
    /// # Why the driver is exercised with an explicit envelope
    ///
    /// This asserts the DRIVER's wiring, not a kernel. `envelope_fn` is a
    /// caller-supplied closure — `max.rs`, `min.rs` and `atan2.rs` each pass a
    /// different one — so a sound envelope is exactly the driver's contract, and
    /// this test supplies one. It has to: `max_envelope`'s LOWER plane always
    /// carries `c == 0` (its two exact branches emit `{0,1,0}`/`{1,0,0}`, and on
    /// the mixed branch the lower plane is one of those two), so the real max/min
    /// envelopes cannot produce a sign-mixed constant fold at all. The closure
    /// below returns `{coeff_x: 0, coeff_y: 0, c: lx}` on a DEGENERATE box
    /// `lx == ux == ly == uy`, where `max(x, y) == lx` identically — so the plane
    /// is EXACT, not merely sound, and carries no relaxation slack that could
    /// mask or manufacture the defect.
    ///
    /// # The construction (the classic Higham cancellation, exact in f64)
    ///
    /// Column 0 and the last column sit at the point `2^60`; the 4096 columns
    /// between them sit at the point `-1`. With `w_lower = [+1, +1×4096, -1]` the
    /// constant fold sees the terms `+2^60`, `4096 × (−1)`, `−2^60`:
    ///
    /// * `ulp(2^60) = 256`, so every `−1` is swallowed whole by the running sum;
    /// * the final `−2^60` then cancels it to EXACTLY `0.0`.
    ///
    /// The exact sum is `−4096`, and because every plane is exact and every
    /// coefficient is `0`, `−4096` is also the op's exact value
    /// `Σ_j w_j · max(x_j, y_j) = 2^60 − 4096 − 2^60`. Publishing the `0.0` with
    /// only `next_down_f32` applied yields a lower bound of `−1.4e−45`, which is
    /// ABOVE the true value by 4096: a false `unsat` is available from it. The
    /// `γ^{f64}_{adds+1} · Σ|summand|` charge covers it — the magnitude sum is
    /// `2^61 + 4096`, so the charge is ~1e6, four orders of magnitude larger than
    /// the deficit it must cover — precisely because it does NOT scale with the
    /// cancelled result.
    ///
    /// The UPPER half is asserted in the same call, mirrored
    /// (`w_upper = [+1, −1×4096, −1]` ⇒ terms `+2^60`, `4096 × (+1)`, `−2^60`,
    /// exact sum `+4096`), because certifying one direction and not the other is
    /// the exact defect class this test exists to close.
    #[test]
    fn minmax_constant_fold_cancellation_is_certified_outward() {
        const BIG: f32 = 1_152_921_504_606_846_976.0; // 2^60
        const N: usize = 4096;
        let n = N + 2;

        // Exact envelope on a degenerate box: max(x, y) == lx everywhere on it.
        let exact_point_envelope = |lx: f32, ux: f32, ly: f32, uy: f32| -> Option<Envelope> {
            assert!(lx == ux && ly == uy && lx == ly, "degenerate box expected");
            let p = Plane {
                coeff_x: 0.0,
                coeff_y: 0.0,
                c: lx,
            };
            Some(Envelope { lower: p, upper: p })
        };

        let mut pts = vec![-1.0f32; n];
        pts[0] = BIG;
        pts[n - 1] = BIG;
        let a = bt(&pts, &pts);
        let b = bt(&pts, &pts);

        let mut la = Array2::<f32>::from_elem((1, n), 1.0);
        la[[0, n - 1]] = -1.0;
        let mut ua = Array2::<f32>::from_elem((1, n), -1.0);
        ua[[0, 0]] = 1.0;

        // Preconditions, checked against the same f64 semantics the driver uses.
        let mut naive_lo = 0.0f64;
        let mut naive_hi = 0.0f64;
        for j in 0..n {
            naive_lo += f64::from(la[[0, j]]) * f64::from(pts[j]);
            naive_hi += f64::from(ua[[0, j]]) * f64::from(pts[j]);
        }
        assert_eq!(
            naive_lo, 0.0,
            "precondition: the sequential f64 lower fold must cancel to exactly 0"
        );
        assert_eq!(
            naive_hi, 0.0,
            "precondition: the sequential f64 upper fold must cancel to exactly 0"
        );
        let exact_lo = -(N as f64); // 2^60 - 4096 - 2^60, grouped exactly
        let exact_hi = N as f64;

        let spec = LinearBounds::new(la, Array1::zeros(1), ua, Array1::zeros(1)).unwrap();
        let (bounds_a, bounds_b) = propagate_minmax_linear_binary(
            &spec,
            &a,
            &b,
            "MaxBinaryConstFold",
            exact_point_envelope,
        )
        .unwrap();

        // Precondition: every plane coefficient is 0, so the COEFFICIENT
        // certificate is identically zero here and cannot be what saves the bound.
        for j in 0..n {
            assert_eq!(bounds_a.lower_a()[[0, j]], 0.0);
            assert_eq!(bounds_a.upper_a()[[0, j]], 0.0);
            assert_eq!(bounds_b.lower_a()[[0, j]], 0.0);
            assert_eq!(bounds_b.upper_a()[[0, j]], 0.0);
        }

        // The published CONSTANT must bracket the exact fold OUTWARD. `bounds_b`
        // carries a zero bias, so the whole constant is on `bounds_a`.
        let published_lo = f64::from(bounds_a.lower_b()[0]) + f64::from(bounds_b.lower_b()[0]);
        let published_hi = f64::from(bounds_a.upper_b()[0]) + f64::from(bounds_b.upper_b()[0]);
        assert!(
            published_lo <= exact_lo,
            "published lower constant {published_lo} is ABOVE the exact fold \
             {exact_lo}; the f64 accumulation error of the envelope-constant fold \
             is uncertified (naive f64 sum was {naive_lo})"
        );
        assert!(
            published_hi >= exact_hi,
            "published upper constant {published_hi} is BELOW the exact fold \
             {exact_hi}; certifying only the LOWER constant is the same \
             half-a-relaxation defect (naive f64 sum was {naive_hi})"
        );

        // End-to-end: with exact planes and zero coefficients the op's value is
        // pinned to -4096 / +4096, so the published relation must enclose it.
        let lo =
            bounds_a.concretize_sound(&a).lower()[0] + bounds_b.concretize_sound(&b).lower()[0];
        let hi =
            bounds_a.concretize_sound(&a).upper()[0] + bounds_b.concretize_sound(&b).upper()[0];
        assert!(
            f64::from(lo) <= exact_lo,
            "published CROWN lower bound {lo} is ABOVE the true value -4096 of \
             sum_j w_j*max(x_j,y_j); the envelope-CONSTANT fold error is uncertified"
        );
        assert!(
            f64::from(hi) >= exact_hi,
            "published CROWN upper bound {hi} is BELOW the true value +4096 of \
             sum_j w_j*max(x_j,y_j); the envelope-CONSTANT fold error is uncertified"
        );
    }
}
