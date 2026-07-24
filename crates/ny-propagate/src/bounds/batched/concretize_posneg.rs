// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Positive/negative coefficient split helpers for batched concretization.
//!
//! Extracted from `concretize.rs` to stay within file-size limits.
//! Part of #2220 Packet B.

use super::BatchedLinearBounds;
use crate::bounds::safe_math::{
    nan_propagating_max_zero, nan_propagating_min_zero, safe_mul_for_bounds_f64,
};
use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result, CROWN_COEFF_MAX};
use ny_tensor::{next_down_f32, next_up_f32};

/// Directed-round one f64-accumulated concrete bound at the final f32 cast.
///
/// The BLAS path accumulates the CROWN dot product entirely in f64 (operands
/// cast f32→f64, so each product `a_j * x_j` is EXACT, and the running sum has
/// only sub-f64-ULP error over `n` terms — utterly negligible against an f32
/// ULP). The ONLY rounding that can move the bound the wrong way is the single
/// final f64→f32 cast. We absorb it with one directed step: `next_down_f32` for
/// a lower bound (toward −∞), `next_up_f32` for an upper bound (toward +∞).
///
/// This is the SAME soundness basis as the f64-scalar fallback
/// (`concretize_scalar_posneg`): f64 accumulation + a single 1-ULP directed
/// cast. It is sound AND tight — no absolute-envelope over-widening is needed,
/// because there is no f32 accumulation error to cover.
///
/// `signed_sum` is the f64 dot result plus the (f64) bias. A NaN accumulator
/// (only reachable if an Inf coefficient × 0 input leaked through, which the
/// `all_finite_for_blas` gate already excludes) is degraded to the sound ±∞
/// sentinel, matching the scalar path.
#[inline]
fn round_blas_element(signed_sum: f64, round_down: bool) -> f32 {
    if round_down {
        if signed_sum.is_nan() {
            f32::NEG_INFINITY
        } else {
            next_down_f32(signed_sum as f32)
        }
    } else if signed_sum.is_nan() {
        f32::INFINITY
    } else {
        next_up_f32(signed_sum as f32)
    }
}

impl BatchedLinearBounds {
    /// Apply the certified coefficient-error penalty to already-concretized bounds
    /// (#vnncomp-aw-soundness).
    ///
    /// For each output position `p = [...batch, i]`, subtracts
    /// `penalty_p = Σ_j max(|in_l_j|, |in_u_j|) · err[...batch, i, j]` from the
    /// concrete lower bound and adds it to the concrete upper bound, where `err` is
    /// `self.lower_a_err` (lower) / `self.upper_a_err` (upper). This is exactly the
    /// scalar `LinearBounds::concretize` penalty: it is sound for ANY true
    /// coefficient inside `[stored - err, stored + err]` over the input box. The
    /// directed `next_down_f32`/`next_up_f32` absorbs the final f32 cast. No-ops
    /// when no error is carried. The input box is broadcast across batch positions
    /// (each position uses the per-column worst-case magnitude `max(|in_l|,|in_u|)`).
    pub(super) fn apply_coeff_err_penalty(
        &self,
        concrete_lower: ArrayD<f32>,
        concrete_upper: ArrayD<f32>,
        in_lower: &ArrayD<f32>,
        in_upper: &ArrayD<f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
        if !self.has_coeff_err() {
            return Ok((concrete_lower, concrete_upper));
        }
        let a_shape = self.lower_a.shape();
        if a_shape.len() < 2 {
            return Ok((concrete_lower, concrete_upper));
        }
        let m = a_shape[a_shape.len() - 2];
        let n = a_shape[a_shape.len() - 1];
        let coeff_batch = checked_shape_product(&a_shape[..a_shape.len() - 2])
            .unwrap_or(0)
            .max(1);
        // Output batch positions = concrete result element count / m. When the
        // coefficients are unbatched but the input is batched, the result has more
        // batch positions than the coefficient array; the error broadcasts across
        // them (each output position re-uses the same per-coefficient error).
        if m == 0 {
            return Ok((concrete_lower, concrete_upper));
        }
        let total_batch = (concrete_lower.len() / m).max(1);
        let coeff_broadcast = coeff_batch == 1 && total_batch > 1;
        let out_batch_dims: Vec<usize> = if concrete_lower.ndim() >= 1 {
            concrete_lower.shape()[..concrete_lower.ndim() - 1].to_vec()
        } else {
            Vec::new()
        };

        // Per-column worst-case input magnitude (broadcast). The input box may be
        // batched; collapse to the per-column max over ALL input positions so the
        // penalty is a sound upper bound regardless of which batch a coefficient
        // multiplies. (Coefficients are per-position; using the global per-column
        // max only widens, preserving soundness.)
        let x_n = *in_lower.shape().last().unwrap_or(&0);
        if x_n != n {
            // Cannot align the box to the coefficient columns; widen everything to
            // conservative rather than risk an unsound (too-tight) penalty.
            let lo = concrete_lower.mapv(|_| f32::NEG_INFINITY);
            let hi = concrete_upper.mapv(|_| f32::INFINITY);
            return Ok((lo, hi));
        }
        let x_batch_elems = in_lower.len() / x_n.max(1);
        let xl = in_lower
            .view()
            .into_shape_with_order((x_batch_elems, x_n))
            .map_err(|e| NyError::InternalError(format!("err penalty in_lower reshape: {e}")))?;
        let xu = in_upper
            .view()
            .into_shape_with_order((x_batch_elems, x_n))
            .map_err(|e| NyError::InternalError(format!("err penalty in_upper reshape: {e}")))?;

        // Worst-case magnitude per column, per input batch position.
        let mut mag = Array2::<f64>::zeros((x_batch_elems, x_n));
        for b in 0..x_batch_elems {
            for j in 0..x_n {
                mag[[b, j]] = (xl[[b, j]] as f64).abs().max((xu[[b, j]] as f64).abs());
            }
        }

        let err_batch = if coeff_broadcast { 1 } else { total_batch };
        let le = self.lower_a_err.as_ref();
        let ue = self.upper_a_err.as_ref();
        let le_flat = le
            .map(|e| e.view().into_shape_with_order((err_batch, m, n)))
            .transpose()
            .map_err(|e| NyError::InternalError(format!("lower_a_err reshape: {e}")))?;
        let ue_flat = ue
            .map(|e| e.view().into_shape_with_order((err_batch, m, n)))
            .transpose()
            .map_err(|e| NyError::InternalError(format!("upper_a_err reshape: {e}")))?;

        let mut lower2d = concrete_lower
            .into_shape_with_order((total_batch, m))
            .map_err(|e| NyError::InternalError(format!("concrete_lower reshape: {e}")))?;
        let mut upper2d = concrete_upper
            .into_shape_with_order((total_batch, m))
            .map_err(|e| NyError::InternalError(format!("concrete_upper reshape: {e}")))?;

        for b in 0..total_batch {
            let xb = if x_batch_elems == 1 {
                0
            } else {
                b.min(x_batch_elems - 1)
            };
            let eb = if coeff_broadcast { 0 } else { b };
            for i in 0..m {
                let mut pen_l = 0.0f64;
                let mut pen_u = 0.0f64;
                for j in 0..n {
                    let mg = mag[[xb, j]];
                    if let Some(ref l) = le_flat {
                        pen_l += l[[eb, i, j]] as f64 * mg;
                    }
                    if let Some(ref u) = ue_flat {
                        pen_u += u[[eb, i, j]] as f64 * mg;
                    }
                }
                if pen_l != 0.0 {
                    let lo = lower2d[[b, i]];
                    if lo.is_finite() {
                        lower2d[[b, i]] = if pen_l.is_finite() {
                            next_down_f32((lo as f64 - pen_l) as f32)
                        } else {
                            f32::NEG_INFINITY
                        };
                    }
                }
                if pen_u != 0.0 {
                    let hi = upper2d[[b, i]];
                    if hi.is_finite() {
                        upper2d[[b, i]] = if pen_u.is_finite() {
                            next_up_f32((hi as f64 + pen_u) as f32)
                        } else {
                            f32::INFINITY
                        };
                    }
                }
            }
        }

        let mut out_shape: Vec<usize> = out_batch_dims;
        out_shape.push(m);
        let (vl, _) = lower2d.into_raw_vec_and_offset();
        let (vu, _) = upper2d.into_raw_vec_and_offset();
        Ok((
            ArrayD::from_shape_vec(IxDyn(&out_shape), vl)
                .map_err(|e| NyError::InternalError(format!("err penalty lower reshape: {e}")))?,
            ArrayD::from_shape_vec(IxDyn(&out_shape), vu)
                .map_err(|e| NyError::InternalError(format!("err penalty upper reshape: {e}")))?,
        ))
    }

    /// Check if all inputs/coefficients/biases are finite for the fast path.
    pub(super) fn all_finite_for_blas(
        lower_a: &ArrayD<f32>,
        upper_a: &ArrayD<f32>,
        lower_b: &ArrayD<f32>,
        upper_b: &ArrayD<f32>,
        in_lower: &ArrayD<f32>,
        in_upper: &ArrayD<f32>,
    ) -> bool {
        let coeff_ok = |v: &f32| v.is_finite() && v.abs() <= CROWN_COEFF_MAX;
        in_lower.iter().all(|v| v.is_finite())
            && in_upper.iter().all(|v| v.is_finite())
            && lower_b.iter().all(|v| v.is_finite())
            && upper_b.iter().all(|v| v.is_finite())
            && lower_a.iter().all(coeff_ok)
            && upper_a.iter().all(coeff_ok)
    }

    /// Fast concretize using BLAS-accelerated positive/negative coefficient split.
    ///
    /// For the lower bound function f_L(x) = A_L @ x + b_L:
    ///   min_{x in [x_l, x_u]} f_L(x) = A_L_pos @ x_l + A_L_neg @ x_u + b_L
    ///
    /// For the upper bound function f_U(x) = A_U @ x + b_U:
    ///   max_{x in [x_l, x_u]} f_U(x) = A_U_pos @ x_u + A_U_neg @ x_l + b_U
    ///
    /// Splits coefficients into positive/negative parts, then casts the operands
    /// to f64 and uses ndarray `.dot()` which dispatches to BLAS DGEMV/DGEMM
    /// (Accelerate on macOS). When all batches share the same input (broadcast),
    /// fuses into a single DGEMV over [batch*m, n].
    ///
    /// SOUNDNESS / TIGHTNESS: the dot products are accumulated entirely in f64.
    /// Each operand is an exact f32 value widened to f64, so every product
    /// `a_j * x_j` is EXACT (f32 × f32 fits in f64 with room to spare), and the
    /// f64 running sum over `n` terms has only sub-f64-ULP relative error
    /// (~`n * 2^-53`) — utterly negligible against an f32 ULP. The result is
    /// therefore EXACT up to a single f64→f32 cast. We absorb only that one cast
    /// with a directed `next_down_f32` (lower) / `next_up_f32` (upper) in
    /// [`round_blas_element`].
    ///
    /// This is the SAME soundness basis as the f64-scalar fallback
    /// ([`Self::concretize_scalar_posneg`]) — exact f64 products, f64
    /// accumulation, one directed f32-cast ULP — so the two paths are equally
    /// sound and agree to within the summation order, which is sub-f64-ULP and
    /// vanishes under the f32 cast. Forming the products in f64 is load-bearing
    /// on BOTH paths, not a tightness nicety: an f32 product rounds to nearest,
    /// i.e. INWARD by up to 0.5 f32-ULP at the *term* magnitude, and under
    /// cancellation (|term| ≫ |result|) that bias is unbounded relative to the
    /// 1-ULP widening applied at the *result* magnitude. No absolute-envelope
    /// over-widening is needed, because f64 accumulation leaves no
    /// f32-accumulation error to cover — replacing the conservative
    /// `gamma_{2n+2} * sum|term|` envelope of the previous f32-BLAS path, which
    /// loosened bounds under cancellation.
    ///
    /// Reference: alpha-beta-CROWN bound_general.py:1140-1160 for the pos/neg
    /// split; the f64-accumulate + single directed-cast rounding is ny's sound,
    /// tight concretization (matching the trusted scalar path).
    pub(super) fn concretize_blas_posneg(
        lower_a: &ArrayD<f32>,
        upper_a: &ArrayD<f32>,
        lower_b: &ArrayD<f32>,
        upper_b: &ArrayD<f32>,
        in_lower: &ArrayD<f32>,
        in_upper: &ArrayD<f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
        let a_shape = lower_a.shape();
        let x_shape = in_lower.shape();

        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(format!(
                "concretize_blas_posneg requires coefficient array with >= 2 dims, got {}",
                a_shape.len()
            )));
        }

        let m = a_shape[a_shape.len() - 2];
        let n = a_shape[a_shape.len() - 1];
        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch = checked_shape_product(batch_dims)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "concretize_blas: batch dims overflow: {:?}",
                    batch_dims
                ))
            })?
            .max(1);

        let mut out_shape: Vec<usize> = batch_dims.to_vec();
        out_shape.push(m);

        if m == 0 || n == 0 || out_shape.contains(&0) {
            return Ok((
                ArrayD::zeros(IxDyn(&out_shape)),
                ArrayD::zeros(IxDyn(&out_shape)),
            ));
        }

        let x_n = *x_shape.last().unwrap_or(&0);
        let x_batch_elems = in_lower.len() / x_n.max(1);

        let expected_b_elems = total_batch * m;
        if lower_b.len() != expected_b_elems || upper_b.len() != expected_b_elems {
            return Err(NyError::shape_mismatch(
                vec![total_batch, m],
                lower_b.shape().to_vec(),
            ));
        }
        let lb_flat = lower_b
            .view()
            .into_shape_with_order((total_batch, m))
            .map_err(|e| NyError::InternalError(format!("lower_b reshape: {e}")))?;
        let ub_flat = upper_b
            .view()
            .into_shape_with_order((total_batch, m))
            .map_err(|e| NyError::InternalError(format!("upper_b reshape: {e}")))?;

        let mut result_lower = Array2::<f32>::zeros((total_batch, m));
        let mut result_upper = Array2::<f32>::zeros((total_batch, m));

        if x_batch_elems == 1 {
            // Broadcast: all batches share the same input bounds.
            // Fuse [batch, m, n] → [batch*m, n] for a single BLAS DGEMV call.
            // Cast coefficients to f64 so accumulation (and the pos/neg split
            // products) are exact-then-f64-summed; the dot routes through DGEMV.
            let la_2d = lower_a
                .view()
                .into_shape_with_order((total_batch * m, n))
                .map_err(|e| NyError::InternalError(format!("lower_a fused reshape: {e}")))?
                .mapv(|v| v as f64);
            let ua_2d = upper_a
                .view()
                .into_shape_with_order((total_batch * m, n))
                .map_err(|e| NyError::InternalError(format!("upper_a fused reshape: {e}")))?
                .mapv(|v| v as f64);

            let xl_flat = in_lower
                .view()
                .into_shape_with_order(n)
                .map_err(|e| NyError::InternalError(format!("in_lower reshape: {e}")))?
                .mapv(|v| v as f64);
            let xu_flat = in_upper
                .view()
                .into_shape_with_order(n)
                .map_err(|e| NyError::InternalError(format!("in_upper reshape: {e}")))?
                .mapv(|v| v as f64);

            // Pos/neg split in f64 + BLAS DGEMV: 4 calls over fused [batch*m, n].
            // Products are EXACT in f64, sums accumulate in f64 → no f32
            // accumulation error to widen for (only the final cast is rounded).
            let la_pos = la_2d.mapv(|v| v.max(0.0));
            let la_neg = la_2d.mapv(|v| v.min(0.0));
            let ua_pos = ua_2d.mapv(|v| v.max(0.0));
            let ua_neg = ua_2d.mapv(|v| v.min(0.0));

            let lower_flat = la_pos.dot(&xl_flat) + la_neg.dot(&xu_flat);
            let upper_flat = ua_pos.dot(&xu_flat) + ua_neg.dot(&xl_flat);

            // Reshape f64 dot results to [batch, m], add the (f64) bias, then a
            // SINGLE directed f64→f32 cast (next_down for lower, next_up upper).
            let lower_2d = lower_flat
                .into_shape_with_order((total_batch, m))
                .map_err(|e| NyError::InternalError(format!("lower result reshape: {e}")))?;
            let upper_2d = upper_flat
                .into_shape_with_order((total_batch, m))
                .map_err(|e| NyError::InternalError(format!("upper result reshape: {e}")))?;

            for b in 0..total_batch {
                for i in 0..m {
                    let sum_l = lower_2d[[b, i]] + lb_flat[[b, i]] as f64;
                    let sum_u = upper_2d[[b, i]] + ub_flat[[b, i]] as f64;
                    result_lower[[b, i]] = round_blas_element(sum_l, true);
                    result_upper[[b, i]] = round_blas_element(sum_u, false);
                }
            }
        } else {
            // Non-broadcast: per-batch BLAS DGEMV with f64-cast operands.
            let la_flat = lower_a
                .view()
                .into_shape_with_order((total_batch, m, n))
                .map_err(|e| NyError::InternalError(format!("lower_a reshape: {e}")))?
                .mapv(|v| v as f64);
            let ua_flat = upper_a
                .view()
                .into_shape_with_order((total_batch, m, n))
                .map_err(|e| NyError::InternalError(format!("upper_a reshape: {e}")))?
                .mapv(|v| v as f64);
            let xl_flat = in_lower
                .view()
                .into_shape_with_order((x_batch_elems, x_n))
                .map_err(|e| NyError::InternalError(format!("in_lower reshape: {e}")))?
                .mapv(|v| v as f64);
            let xu_flat = in_upper
                .view()
                .into_shape_with_order((x_batch_elems, x_n))
                .map_err(|e| NyError::InternalError(format!("in_upper reshape: {e}")))?
                .mapv(|v| v as f64);

            for b in 0..total_batch {
                let la_b = la_flat.index_axis(ndarray::Axis(0), b);
                let ua_b = ua_flat.index_axis(ndarray::Axis(0), b);
                let xl_b = xl_flat.index_axis(ndarray::Axis(0), b);
                let xu_b = xu_flat.index_axis(ndarray::Axis(0), b);

                // Pos/neg split in f64: products exact, accumulation in f64.
                let la_pos = la_b.mapv(|v| v.max(0.0));
                let la_neg = la_b.mapv(|v| v.min(0.0));
                let lower_dot = la_pos.dot(&xl_b) + la_neg.dot(&xu_b);

                let ua_pos = ua_b.mapv(|v| v.max(0.0));
                let ua_neg = ua_b.mapv(|v| v.min(0.0));
                let upper_dot = ua_pos.dot(&xu_b) + ua_neg.dot(&xl_b);

                // Add (f64) bias, then a SINGLE directed f64→f32 cast.
                for i in 0..m {
                    let sum_l = lower_dot[i] + lb_flat[[b, i]] as f64;
                    let sum_u = upper_dot[i] + ub_flat[[b, i]] as f64;
                    result_lower[[b, i]] = round_blas_element(sum_l, true);
                    result_upper[[b, i]] = round_blas_element(sum_u, false);
                }
            }
        }

        let (vec_l, _) = result_lower.into_raw_vec_and_offset();
        let (vec_u, _) = result_upper.into_raw_vec_and_offset();
        Ok((
            ArrayD::from_shape_vec(IxDyn(&out_shape), vec_l)
                .map_err(|e| NyError::InternalError(format!("lower reshape back: {e}")))?,
            ArrayD::from_shape_vec(IxDyn(&out_shape), vec_u)
                .map_err(|e| NyError::InternalError(format!("upper reshape back: {e}")))?,
        ))
    }

    /// Scalar fallback for concretize with per-element NaN/Inf/overflow handling.
    ///
    /// Uses the same pos/neg coefficient split but with `safe_mul_for_bounds_f64`
    /// (0*inf=0) and NaN-propagating max/min to handle edge cases soundly. Both
    /// operands are promoted to f64 before the product, so — exactly as on the BLAS
    /// path — every term is exact and only the final directed f32 cast rounds.
    ///
    /// Rows whose bias is already at ±inf, or whose coefficients exceed
    /// `CROWN_COEFF_MAX` (including the ±Inf conservative NaN guards `compose`
    /// emits), are degraded per-direction to the sound saturating bound,
    /// mirroring `LinearBounds::concretize_scalar_f64`. This path is the only
    /// one such rows can reach: `all_finite_for_blas` excludes them from BLAS.
    pub(super) fn concretize_scalar_posneg(
        lower_a: &ArrayD<f32>,
        upper_a: &ArrayD<f32>,
        lower_b: &ArrayD<f32>,
        upper_b: &ArrayD<f32>,
        in_lower: &ArrayD<f32>,
        in_upper: &ArrayD<f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
        let a_shape = lower_a.shape();
        let x_shape = in_lower.shape();

        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(format!(
                "concretize_scalar_posneg requires coefficient array with >= 2 dims, got {}",
                a_shape.len()
            )));
        }

        let m = a_shape[a_shape.len() - 2];
        let n = a_shape[a_shape.len() - 1];
        let batch_dims = &a_shape[..a_shape.len() - 2];
        let total_batch = checked_shape_product(batch_dims)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "concretize_scalar: batch dims overflow: {:?}",
                    batch_dims
                ))
            })?
            .max(1);

        let mut out_shape: Vec<usize> = batch_dims.to_vec();
        out_shape.push(m);

        if m == 0 || n == 0 || out_shape.contains(&0) {
            return Ok((
                ArrayD::zeros(IxDyn(&out_shape)),
                ArrayD::zeros(IxDyn(&out_shape)),
            ));
        }

        let la_flat = lower_a
            .view()
            .into_shape_with_order((total_batch, m, n))
            .map_err(|e| NyError::InternalError(format!("lower_a reshape: {e}")))?;
        let ua_flat = upper_a
            .view()
            .into_shape_with_order((total_batch, m, n))
            .map_err(|e| NyError::InternalError(format!("upper_a reshape: {e}")))?;

        let x_n = *x_shape.last().unwrap_or(&0);
        let x_batch_elems = in_lower.len() / x_n.max(1);
        let xl_flat = in_lower
            .view()
            .into_shape_with_order((x_batch_elems, x_n))
            .map_err(|e| NyError::InternalError(format!("in_lower reshape: {e}")))?;
        let xu_flat = in_upper
            .view()
            .into_shape_with_order((x_batch_elems, x_n))
            .map_err(|e| NyError::InternalError(format!("in_upper reshape: {e}")))?;

        let expected_b_elems = total_batch * m;
        if lower_b.len() != expected_b_elems || upper_b.len() != expected_b_elems {
            return Err(NyError::shape_mismatch(
                vec![total_batch, m],
                lower_b.shape().to_vec(),
            ));
        }
        let lb_flat = lower_b
            .view()
            .into_shape_with_order((total_batch, m))
            .map_err(|e| NyError::InternalError(format!("lower_b reshape: {e}")))?;
        let ub_flat = upper_b
            .view()
            .into_shape_with_order((total_batch, m))
            .map_err(|e| NyError::InternalError(format!("upper_b reshape: {e}")))?;

        let mut result_lower = Array2::<f32>::zeros((total_batch, m));
        let mut result_upper = Array2::<f32>::zeros((total_batch, m));

        for b in 0..total_batch {
            let x_b = if x_batch_elems == 1 { 0 } else { b };

            for i in 0..m {
                let lb = lb_flat[[b, i]];
                let ub = ub_flat[[b, i]];

                // Per-direction row degradation (mirrors
                // `LinearBounds::concretize_scalar_f64`, #1932): a bias already at
                // ±inf, or any coefficient with |a| > CROWN_COEFF_MAX — which
                // includes the ±Inf conservative NaN guards `compose` emits —
                // degrades that direction to its sound saturating bound. The guard
                // is load-bearing here, not defense-in-depth: the pos/neg split
                // multiplies a guard coefficient by an input endpoint whose sign
                // (or exact zero, via 0·inf = 0) can flip or silently drop the
                // poison, turning a degraded row into a confidently wrong finite
                // bound. Lower and upper are independent: a degraded lower does
                // not loosen a well-behaved upper, and vice versa.
                let mut lower_degraded = lb == f32::NEG_INFINITY;
                let mut upper_degraded = ub == f32::INFINITY;
                for j in 0..n {
                    if lower_degraded && upper_degraded {
                        break;
                    }
                    if !lower_degraded && la_flat[[b, i, j]].abs() > CROWN_COEFF_MAX {
                        lower_degraded = true;
                    }
                    if !upper_degraded && ua_flat[[b, i, j]].abs() > CROWN_COEFF_MAX {
                        upper_degraded = true;
                    }
                }
                if lower_degraded {
                    result_lower[[b, i]] = f32::NEG_INFINITY;
                }
                if upper_degraded {
                    result_upper[[b, i]] = f32::INFINITY;
                }
                if lower_degraded && upper_degraded {
                    continue;
                }

                let mut sum_l = lb as f64;
                let mut sum_u = ub as f64;

                for j in 0..n {
                    let xl = xl_flat[[x_b, j]];
                    let xu = xu_flat[[x_b, j]];
                    let la = la_flat[[b, i, j]];
                    let ua = ua_flat[[b, i, j]];

                    // Pos/neg coefficient split with safe 0*inf=0 handling.
                    // NaN-propagating max/min preserves NaN so it poisons the accumulator.
                    //
                    // SOUNDNESS (#concretize-soundness-hardening): the per-term product is
                    // formed in **f64** (`safe_mul_for_bounds_f64` after promoting both
                    // operands), NOT in f32. A f32 product rounds to nearest and can land
                    // up to 0.5 f32-ULP *inside* the true product; with cancellation
                    // between large positive and negative products (|product| ≫ |result|),
                    // the inward bias accumulated over the `n` terms can exceed the final
                    // `next_*_f32` 1-ULP widening at the result's (small) magnitude —
                    // making the caller's `concretize_sound` itself unsound. f32×f32
                    // promoted to f64 is EXACT (48 < 53 significand bits), so the only
                    // residual error is the (negligible) f64 sum, which the final directed
                    // cast covers. The split itself (max/min with 0) is exact in f32 and
                    // preserved.
                    let (la_pos, la_neg) =
                        (nan_propagating_max_zero(la), nan_propagating_min_zero(la));
                    let (ua_pos, ua_neg) =
                        (nan_propagating_max_zero(ua), nan_propagating_min_zero(ua));
                    sum_l += safe_mul_for_bounds_f64(la_pos as f64, xl as f64)
                        + safe_mul_for_bounds_f64(la_neg as f64, xu as f64);
                    sum_u += safe_mul_for_bounds_f64(ua_pos as f64, xu as f64)
                        + safe_mul_for_bounds_f64(ua_neg as f64, xl as f64);
                }

                // Only write back non-degraded directions: overwriting would
                // silently defeat the row guard above (#3202).
                if !lower_degraded {
                    result_lower[[b, i]] = if sum_l.is_nan() {
                        f32::NEG_INFINITY
                    } else {
                        next_down_f32(sum_l as f32)
                    };
                }
                if !upper_degraded {
                    result_upper[[b, i]] = if sum_u.is_nan() {
                        f32::INFINITY
                    } else {
                        next_up_f32(sum_u as f32)
                    };
                }
            }
        }

        let (vec_l, _) = result_lower.into_raw_vec_and_offset();
        let (vec_u, _) = result_upper.into_raw_vec_and_offset();
        Ok((
            ArrayD::from_shape_vec(IxDyn(&out_shape), vec_l)
                .map_err(|e| NyError::InternalError(format!("lower reshape back: {e}")))?,
            ArrayD::from_shape_vec(IxDyn(&out_shape), vec_u)
                .map_err(|e| NyError::InternalError(format!("upper reshape back: {e}")))?,
        ))
    }
}
