// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! A SOUND interval-valued carrier for both-constant folds that binary32
//! cannot represent exactly.
//!
//! # Why this module exists
//!
//! `try_fold_matmul_exact` folds a both-constant `MatMul` only when every
//! product AND every partial sum is exactly representable in binary32. A
//! length-6 binary32 dot product essentially never is, so the node does not
//! fold, `convert_matmul` fails it closed to an unbounded `OpaqueSkip`, and the
//! taint spreads to every consumer of the constant.
//!
//! Rounding the dot product to a binary32 POINT would remove the taint, but it
//! would also be a lie: it asserts the network has weight `round(W)` and then
//! proves a property of THAT network. That is the same class of error as the
//! relaxed clip retracted in 807bf511 (nineteen false `unsat` verdicts), one
//! layer down — a step that is TIGHTER than the truth.
//!
//! This module takes the third option. It computes an outward-rounded
//! ENCLOSURE, so the frozen constant becomes a set that provably contains the
//! truth rather than a point that provably does not equal it.
//!
//! # What the enclosure claims
//!
//! For a folded tensor with carrier `[lower, upper]`, elementwise:
//!
//! 1. **Exact-real containment.** The exact real-arithmetic value of the
//!    folded expression, evaluated over the stored binary32 leaves as exact
//!    dyadic rationals, lies in `[lower, upper]`. This is the truth model the
//!    rest of the tree already uses — see the `exact_f32_*` certificates in
//!    `common.rs` and the `LinearBounds::lower_a_err` invariant ("`true_coeff`
//!    is the exact real-arithmetic coefficient").
//! 2. **Binary32-implementation containment.** The value that ANY binary32
//!    implementation of the same expression produces — any summation order,
//!    fused multiply-add included — also lies in `[lower, upper]`.
//!
//! (2) is not required by (1)'s truth model, but it is cheap and it is what
//! makes the carrier safe against the counterexample oracle, which evaluates
//! the ORIGINAL graph in binary32 through ONNX Runtime. Every rule below
//! preserves BOTH properties, so they hold for arbitrarily long fold chains by
//! induction: a stored binary32 initializer satisfies both trivially
//! (`[v, v]`), and each op below maps operands satisfying both to a result
//! satisfying both.
//!
//! # Why every step can only widen
//!
//! * A product of two binary32 values is EXACT in binary64 (24 + 24 = 48 <= 53
//!   significand bits). No rounding, so nothing to bound.
//! * Every accumulation is performed in binary64 with an explicit DIRECTED
//!   step: the running lower endpoint is pushed down by `next_down_f64` after
//!   each addition and the upper endpoint up by `next_up_f64`. Round-to-nearest
//!   errs by strictly less than one ulp, and `next_*` moves a full ulp, so the
//!   running endpoints are valid bounds on the exact real partial sums
//!   independently of the order the terms are visited.
//! * Every cast back to binary32 is pushed outward by the tree's own directed
//!   casts (`cast_f64_to_f32_down` / `cast_f64_to_f32_up`), which step ONLY
//!   when the cast is inexact.
//! * The binary32-implementation term is added on top, outward, as a Higham
//!   `gamma_n * S` bound (contractions) or a unit-roundoff bound (elementwise).
//!   Both are summation-order independent.
//!
//! Any operand or result that is not finite makes the fold DECLINE. Refusing is
//! always available and always sound: the node simply stays unfolded and keeps
//! today's fail-closed behaviour.

use ndarray::{ArrayD, Ix1, Ix2, IxDyn};
use ny_core::dd::{next_down_f64, next_up_f64, two_sum};
use ny_tensor::{add_down_f32, add_up_f32, cast_f64_to_f32_down, cast_f64_to_f32_up, sub_down_f32};

/// Binary32 unit roundoff `u = 2^-24`: the relative bound on one
/// round-to-nearest binary32 operation.
const U_F32: f64 = 5.960_464_477_539_063e-8; // 2^-24

/// A frozen constant known only up to a rigorous elementwise enclosure.
///
/// The invariant is documented on this module: `lower <= truth <= upper`, for
/// both the exact-real truth and any binary32 implementation of the same
/// expression.
#[derive(Clone, Debug)]
pub(crate) struct IntervalTensor {
    pub(crate) lower: ArrayD<f32>,
    pub(crate) upper: ArrayD<f32>,
}

impl IntervalTensor {
    /// A degenerate (zero-width) carrier around an exactly known point tensor.
    pub(crate) fn point(value: &ArrayD<f32>) -> Self {
        Self {
            lower: value.clone(),
            upper: value.clone(),
        }
    }

    pub(crate) fn shape(&self) -> &[usize] {
        self.lower.shape()
    }

    /// Whether the carrier is bitwise zero-width everywhere, i.e. the value is
    /// exactly known after all.
    pub(crate) fn is_point(&self) -> bool {
        self.lower
            .iter()
            .zip(self.upper.iter())
            .all(|(l, u)| l.to_bits() == u.to_bits())
    }

    /// Reject a carrier that is not a usable enclosure. Returning `None` makes
    /// the caller decline the fold.
    fn checked(lower: ArrayD<f32>, upper: ArrayD<f32>) -> Option<Self> {
        if lower.shape() != upper.shape() {
            return None;
        }
        for (&l, &u) in lower.iter().zip(upper.iter()) {
            if !l.is_finite() || !u.is_finite() || l > u {
                return None;
            }
        }
        Some(Self { lower, upper })
    }
}

/// Higham's growth factor `gamma_n = n*u / (1 - n*u)` with the BINARY32 unit
/// roundoff. Multiplied by the absolute-product sum `S`, it bounds the error of
/// a length-`n` dot product accumulated in binary32, for EVERY summation order
/// (Higham, *Accuracy and Stability of Numerical Algorithms*, Thm 3.1). Mirrors
/// `ny_propagate`'s `crown_single::gamma_n_f32`, which certifies binary32 `A*W`
/// coefficients the same way.
///
/// Order independence is the point: the carrier must contain what an unknown
/// reference implementation computes, not what this loop computes.
fn gamma_n_f32(n: usize) -> f64 {
    let d = (n as f64) * U_F32;
    if d >= 1.0 {
        f64::INFINITY
    } else {
        d / (1.0 - d)
    }
}

/// `a + b` in binary64, rounded toward -inf.
///
/// `two_sum` is an error-free transformation, so the residual reports EXACTLY
/// what the addition lost. Stepping only when the residual is negative keeps an
/// exact addition exact — which is what lets a fold whose every step happens to
/// be representable (a zero operand, a power of two) collapse back to a point
/// and be canonicalized out of the interval map entirely.
fn add_down_f64(a: f64, b: f64) -> f64 {
    let (hi, lo) = two_sum(a, b);
    if !hi.is_finite() {
        return hi;
    }
    if lo < 0.0 {
        next_down_f64(hi)
    } else {
        hi
    }
}

/// `a + b` in binary64, rounded toward +inf. Mirror of [`add_down_f64`].
fn add_up_f64(a: f64, b: f64) -> f64 {
    let (hi, lo) = two_sum(a, b);
    if !hi.is_finite() {
        return hi;
    }
    if lo > 0.0 {
        next_up_f64(hi)
    } else {
        hi
    }
}

/// Widen a binary64 enclosure outward and land it in binary32.
///
/// `extra` is a non-negative binary64 slack added on both sides before the
/// cast. Every step rounds away from the interval, and every step is a NO-OP
/// when it is exact.
fn close_f64_interval(lo: f64, hi: f64, extra: f64) -> Option<(f32, f32)> {
    if !lo.is_finite() || !hi.is_finite() || !extra.is_finite() || extra < 0.0 {
        return None;
    }
    let lo = if extra == 0.0 {
        lo
    } else {
        add_down_f64(lo, -extra)
    };
    let hi = if extra == 0.0 {
        hi
    } else {
        add_up_f64(hi, extra)
    };
    // The narrowing cast is the only remaining rounding, and the tree's own
    // directed casts perform it in the correct direction and ONLY when the cast
    // is inexact.
    let lo32 = cast_f64_to_f32_down(lo);
    let hi32 = cast_f64_to_f32_up(hi);
    if !lo32.is_finite() || !hi32.is_finite() || lo32 > hi32 {
        return None;
    }
    Some((lo32, hi32))
}

/// Enclosure of one interval dot product.
///
/// `a` and `b` yield `(lower, upper)` endpoint pairs of equal length.
///
/// * Each term's four corner products are EXACT in binary64.
/// * The running endpoints are accumulated with an explicit directed step, so
///   they bound the exact real sum whatever the order.
/// * `S` accumulates `max |corner|` with round-up; `gamma_n_f32(n) * S` then
///   covers any binary32 evaluation of the same contraction.
fn interval_dot(
    a: impl Iterator<Item = (f32, f32)>,
    b: impl Iterator<Item = (f32, f32)>,
) -> Option<(f32, f32)> {
    let mut lo = 0.0_f64;
    let mut hi = 0.0_f64;
    let mut abs_sum = 0.0_f64;
    let mut terms = 0_usize;
    for ((al, ah), (bl, bh)) in a.zip(b) {
        if !al.is_finite() || !ah.is_finite() || !bl.is_finite() || !bh.is_finite() {
            return None;
        }
        let (al, ah, bl, bh) = (f64::from(al), f64::from(ah), f64::from(bl), f64::from(bh));
        // Exact in binary64: 24 + 24 = 48 <= 53 significand bits.
        let corners = [al * bl, al * bh, ah * bl, ah * bh];
        let mut term_lo = corners[0];
        let mut term_hi = corners[0];
        for &c in &corners[1..] {
            if c < term_lo {
                term_lo = c;
            }
            if c > term_hi {
                term_hi = c;
            }
        }
        lo = add_down_f64(lo, term_lo);
        hi = add_up_f64(hi, term_hi);
        abs_sum = add_up_f64(abs_sum, term_lo.abs().max(term_hi.abs()));
        terms += 1;
    }
    // `gamma_n * S` with `S == 0` is exactly zero: a contraction whose every
    // term is zero is computed exactly by any implementation, and inventing a
    // subnormal slack there would stop an exactly-zero fold from collapsing to a
    // point.
    let implementation_slack = if abs_sum == 0.0 {
        0.0
    } else {
        next_up_f64(gamma_n_f32(terms) * abs_sum)
    };
    close_f64_interval(lo, hi, implementation_slack)
}

/// Interval `MatMul`, mirroring the rank cases `try_fold_matmul_exact` accepts.
///
/// Returns `None` — decline, keep today's behaviour — for any rank combination
/// or shape the exact fold would not have handled either.
pub(crate) fn matmul(lhs: &IntervalTensor, rhs: &IntervalTensor) -> Option<IntervalTensor> {
    let (ll, lu) = (&lhs.lower, &lhs.upper);
    let (rl, ru) = (&rhs.lower, &rhs.upper);
    if ll.ndim() == 1 && rl.ndim() == 1 {
        let ll = ll.clone().into_dimensionality::<Ix1>().ok()?;
        let lu = lu.clone().into_dimensionality::<Ix1>().ok()?;
        let rl = rl.clone().into_dimensionality::<Ix1>().ok()?;
        let ru = ru.clone().into_dimensionality::<Ix1>().ok()?;
        if ll.len() != rl.len() {
            return None;
        }
        let (lo, hi) = interval_dot(
            ll.iter().zip(lu.iter()).map(|(&a, &b)| (a, b)),
            rl.iter().zip(ru.iter()).map(|(&a, &b)| (a, b)),
        )?;
        return IntervalTensor::checked(ndarray::arr0(lo).into_dyn(), ndarray::arr0(hi).into_dyn());
    }

    // Promote every remaining accepted case to 2-D x 2-D and remember how to
    // shape the result back, exactly as the exact fold does.
    let (a_lo, a_hi, b_lo, b_hi, out_shape) = if ll.ndim() == 2 && rl.ndim() == 2 {
        let a_lo = ll.clone().into_dimensionality::<Ix2>().ok()?;
        let a_hi = lu.clone().into_dimensionality::<Ix2>().ok()?;
        let b_lo = rl.clone().into_dimensionality::<Ix2>().ok()?;
        let b_hi = ru.clone().into_dimensionality::<Ix2>().ok()?;
        let shape = vec![a_lo.nrows(), b_lo.ncols()];
        (a_lo, a_hi, b_lo, b_hi, shape)
    } else if ll.ndim() == 1 && rl.ndim() == 2 {
        let a_lo = ll.clone().into_shape_with_order((1, ll.len())).ok()?;
        let a_hi = lu.clone().into_shape_with_order((1, lu.len())).ok()?;
        let b_lo = rl.clone().into_dimensionality::<Ix2>().ok()?;
        let b_hi = ru.clone().into_dimensionality::<Ix2>().ok()?;
        let shape = vec![b_lo.ncols()];
        (a_lo, a_hi, b_lo, b_hi, shape)
    } else if ll.ndim() == 2 && rl.ndim() == 1 {
        let a_lo = ll.clone().into_dimensionality::<Ix2>().ok()?;
        let a_hi = lu.clone().into_dimensionality::<Ix2>().ok()?;
        let b_lo = rl.clone().into_shape_with_order((rl.len(), 1)).ok()?;
        let b_hi = ru.clone().into_shape_with_order((ru.len(), 1)).ok()?;
        let shape = vec![a_lo.nrows()];
        (a_lo, a_hi, b_lo, b_hi, shape)
    } else {
        return None;
    };

    if a_lo.ncols() != b_lo.nrows() {
        return None;
    }
    let rows = a_lo.nrows();
    let cols = b_lo.ncols();
    let count = rows.checked_mul(cols)?;
    let mut lower = Vec::with_capacity(count);
    let mut upper = Vec::with_capacity(count);
    for i in 0..rows {
        for j in 0..cols {
            let (lo, hi) = interval_dot(
                a_lo.row(i).iter().copied().zip(a_hi.row(i).iter().copied()),
                b_lo.column(j)
                    .iter()
                    .copied()
                    .zip(b_hi.column(j).iter().copied()),
            )?;
            lower.push(lo);
            upper.push(hi);
        }
    }
    IntervalTensor::checked(
        ArrayD::from_shape_vec(IxDyn(&out_shape), lower).ok()?,
        ArrayD::from_shape_vec(IxDyn(&out_shape), upper).ok()?,
    )
}

/// Broadcast two dynamic-rank shapes ONNX-style, or `None` if incompatible.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut out = vec![1usize; ndim];
    for (i, dim) in out.iter_mut().enumerate() {
        let ad = if i + a.len() >= ndim {
            a[i + a.len() - ndim]
        } else {
            1
        };
        let bd = if i + b.len() >= ndim {
            b[i + b.len() - ndim]
        } else {
            1
        };
        *dim = if ad == bd {
            ad
        } else if ad == 1 {
            bd
        } else if bd == 1 {
            ad
        } else {
            return None;
        };
    }
    Some(out)
}

/// Elementwise slack covering one binary32 round-to-nearest of a result whose
/// magnitude is bounded by `magnitude`.
///
/// Zero magnitude needs no slack: a binary32 sum whose exact value is zero is
/// zero, and returning a subnormal here would widen every exact entry and
/// destroy the "an exact fold is indistinguishable from today" property.
fn elementwise_rounding_slack(magnitude: f32) -> f32 {
    if magnitude == 0.0 {
        return 0.0;
    }
    cast_f64_to_f32_up(f64::from(magnitude) * U_F32)
}

/// Push one endpoint pair outward by the binary32 implementation slack.
fn widen_for_binary32(lower: f32, upper: f32) -> (f32, f32) {
    let slack = elementwise_rounding_slack(lower.abs().max(upper.abs()));
    if slack == 0.0 {
        return (lower, upper);
    }
    (sub_down_f32(lower, slack), add_up_f32(upper, slack))
}

/// Interval `Add` with ONNX broadcasting.
///
/// `[l1 + l2, h1 + h2]` with directed binary32 addition (exact whenever the
/// addition is exact), then one unit-roundoff step outward so the carrier also
/// contains a binary32 implementation's rounded sum.
pub(crate) fn add(lhs: &IntervalTensor, rhs: &IntervalTensor) -> Option<IntervalTensor> {
    const MAX_BROADCAST_ELEMENTS: usize = 10_000_000;
    let shape = broadcast_shape(lhs.shape(), rhs.shape())?;
    let total = ny_core::checked_shape_product(&shape)?;
    if total > MAX_BROADCAST_ELEMENTS {
        return None;
    }
    let dyn_shape = IxDyn(&shape);
    let a_lo = lhs.lower.broadcast(dyn_shape.clone())?;
    let a_hi = lhs.upper.broadcast(dyn_shape.clone())?;
    let b_lo = rhs.lower.broadcast(dyn_shape.clone())?;
    let b_hi = rhs.upper.broadcast(dyn_shape)?;

    let raw_lower = ndarray::Zip::from(&a_lo)
        .and(&b_lo)
        .map_collect(|&x, &y| add_down_f32(x, y));
    let raw_upper = ndarray::Zip::from(&a_hi)
        .and(&b_hi)
        .map_collect(|&x, &y| add_up_f32(x, y));
    let lower = ndarray::Zip::from(&raw_lower)
        .and(&raw_upper)
        .map_collect(|&l, &u| widen_for_binary32(l, u).0);
    let upper = ndarray::Zip::from(&raw_lower)
        .and(&raw_upper)
        .map_collect(|&l, &u| widen_for_binary32(l, u).1);
    IntervalTensor::checked(lower, upper)
}

/// Interval `Relu`. Monotone and exactly representable, so nothing rounds.
pub(crate) fn relu(input: &IntervalTensor) -> Option<IntervalTensor> {
    IntervalTensor::checked(
        input.lower.mapv(|v| if v > 0.0 { v } else { 0.0 }),
        input.upper.mapv(|v| if v > 0.0 { v } else { 0.0 }),
    )
}
