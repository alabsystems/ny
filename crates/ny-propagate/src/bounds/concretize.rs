// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Concretization methods for `LinearBounds`.
//!
//! Concretization computes concrete numerical bounds from linear bounds
//! given input bounds. This is the final step of CROWN backward propagation.

use super::safe_math::{
    nan_propagating_max_zero, nan_propagating_min_zero, safe_mul_for_bounds_f64,
};
use ndarray::Array1;
use ny_core::{NyError, Result, CROWN_COEFF_MAX};
use ny_tensor::{
    next_down_f32, next_up_f32, repair_inverted_bounds_nd, BoundedTensor, InversionRepair,
};

use crate::contiguous_flat_slice;

use super::LinearBounds;

/// Certified per-row concretization core (directed f64→f32 rounding), shared by
/// the patches-native sparse concretize
/// (`PatchesLinearBounds::concretize_sound_sparse`).
///
/// Computes the concrete `[lower, upper]` for ONE output row from that row's
/// per-active-column data, applying the SAME arithmetic as one row of
/// [`LinearBounds::concretize_scalar_f64`] followed by one element of
/// [`LinearBounds::f64_to_bounded_tensor`]'s directed cast + repair:
///
/// - f64 dot accumulation with `safe_mul_for_bounds_f64` (0·inf=0) and the
///   NaN-propagating positive/negative split,
/// - the certified coefficient-error penalty (`le`/`ue`, mirroring
///   `lower_a_err`/`upper_a_err`; `Some` ⇒ that side carries error, the slice is
///   this row's per-active-column error already materialized exactly as
///   `to_dense` would, `None` ⇒ exact side; the err pass runs iff either is
///   `Some`, matching the dense gate),
/// - the `lower_b == -inf` / `upper_b == +inf` degrade and the `CROWN_COEFF_MAX`
///   row-overflow guard,
/// - the NaN→±inf fallback, the `next_down_f32`/`next_up_f32` directed cast, and
///   the non-finite / inversion repair to `[-inf, +inf]`.
///
/// SOUNDNESS / BIT-IDENTITY: the caller MUST pass exactly the row's active
/// columns (a superset of its nonzero-coefficient / nonzero-err columns is also
/// fine) in **strictly increasing global column index order**. Every omitted
/// column has coefficient `0` and error `0`, so its dense contribution
/// `safe_mul(0,·)+safe_mul(0,·)` and `0·mag` are exactly `0.0` — an f64 no-op
/// add — and it cannot trip the overflow guard (`0.abs() ≤ CROWN_COEFF_MAX`).
/// Hence the running f64 accumulator matches the full `j = 0..n` dense loop at
/// every active column, and the result is bit-for-bit identical to
/// `to_dense().concretize_sound(input)` for that row. Pinned by
/// `sparse_concretize_matches_dense_bit_identical`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn concretize_row_directed(
    lb: f32,
    ub: f32,
    in_l: &[f32],
    in_u: &[f32],
    la: &[f32],
    ua: &[f32],
    le: Option<&[f32]>,
    ue: Option<&[f32]>,
) -> (f32, f32) {
    let n = la.len();
    debug_assert_eq!(in_l.len(), n);
    debug_assert_eq!(in_u.len(), n);
    debug_assert_eq!(ua.len(), n);

    // Degrade / overflow guards (mirror concretize_scalar_f64:130-160). Inactive
    // columns have coefficient 0, so scanning only the active columns yields the
    // same overflow determination.
    let lower_degraded = lb == f32::NEG_INFINITY;
    let upper_degraded = ub == f32::INFINITY;
    let mut lower_row_overflow = false;
    let mut upper_row_overflow = false;
    if !lower_degraded || !upper_degraded {
        for j in 0..n {
            if !lower_degraded && la[j].abs() > CROWN_COEFF_MAX {
                lower_row_overflow = true;
            }
            if !upper_degraded && ua[j].abs() > CROWN_COEFF_MAX {
                upper_row_overflow = true;
            }
            if (lower_degraded || lower_row_overflow) && (upper_degraded || upper_row_overflow) {
                break;
            }
        }
    }

    let mut lower_f64 = 0.0f64;
    let mut upper_f64 = 0.0f64;
    if lower_degraded || lower_row_overflow {
        lower_f64 = f64::NEG_INFINITY;
    }
    if upper_degraded || upper_row_overflow {
        upper_f64 = f64::INFINITY;
    }
    if !((lower_degraded || lower_row_overflow) && (upper_degraded || upper_row_overflow)) {
        let mut sum_l = lb as f64;
        let mut sum_u = ub as f64;
        let mut err_penalty_l = 0.0f64;
        let mut err_penalty_u = 0.0f64;
        let have_err = le.is_some() || ue.is_some();
        for j in 0..n {
            let inl = in_l[j];
            let inu = in_u[j];
            let laj = la[j];
            let uaj = ua[j];
            let (la_pos, la_neg) = (nan_propagating_max_zero(laj), nan_propagating_min_zero(laj));
            let (ua_pos, ua_neg) = (nan_propagating_max_zero(uaj), nan_propagating_min_zero(uaj));
            sum_l += safe_mul_for_bounds_f64(la_pos as f64, inl as f64)
                + safe_mul_for_bounds_f64(la_neg as f64, inu as f64);
            sum_u += safe_mul_for_bounds_f64(ua_pos as f64, inu as f64)
                + safe_mul_for_bounds_f64(ua_neg as f64, inl as f64);
            if have_err {
                let mag = (inl as f64).abs().max((inu as f64).abs());
                if let Some(le) = le {
                    err_penalty_l += le[j] as f64 * mag;
                }
                if let Some(ue) = ue {
                    err_penalty_u += ue[j] as f64 * mag;
                }
            }
        }
        if err_penalty_l != 0.0 {
            sum_l -= err_penalty_l;
        }
        if err_penalty_u != 0.0 {
            sum_u += err_penalty_u;
        }
        if !(lower_degraded || lower_row_overflow) {
            lower_f64 = if sum_l.is_nan() {
                f64::NEG_INFINITY
            } else {
                sum_l
            };
        }
        if !(upper_degraded || upper_row_overflow) {
            upper_f64 = if sum_u.is_nan() { f64::INFINITY } else { sum_u };
        }
    }

    // Directed cast + per-element repair (mirror f64_to_bounded_tensor): a
    // non-finite endpoint OR an inversion widens to the sound [-inf, +inf].
    let mut l = next_down_f32(lower_f64 as f32);
    let mut u = next_up_f32(upper_f64 as f32);
    if !l.is_finite() || !u.is_finite() || l > u {
        l = f32::NEG_INFINITY;
        u = f32::INFINITY;
    }
    (l, u)
}

impl LinearBounds {
    fn conservative_unbounded(num_outputs: usize) -> BoundedTensor {
        BoundedTensor::new_conservative(&[num_outputs])
    }

    pub(crate) fn validate_internal_shapes(&self) -> Result<()> {
        let lower_shape = self.lower_a.shape().to_vec();
        let upper_shape = self.upper_a.shape().to_vec();
        if lower_shape != upper_shape {
            return Err(NyError::InvalidSpec(format!(
                "LinearBounds invariant violated: lower_a shape {:?} != upper_a shape {:?}",
                lower_shape, upper_shape
            )));
        }

        let expected_outputs = self.lower_a.nrows();
        if self.lower_b.len() != expected_outputs {
            return Err(NyError::InvalidSpec(format!(
                "LinearBounds invariant violated: lower_b len {} != lower_a.nrows() {}",
                self.lower_b.len(),
                expected_outputs
            )));
        }
        if self.upper_b.len() != expected_outputs {
            return Err(NyError::InvalidSpec(format!(
                "LinearBounds invariant violated: upper_b len {} != lower_a.nrows() {}",
                self.upper_b.len(),
                expected_outputs
            )));
        }
        // Certified coefficient-error matrices, when present, must match the
        // corresponding coefficient matrix shape (#vnncomp-aw-soundness).
        if let Some(le) = self.lower_a_err.as_ref() {
            if le.shape() != self.lower_a.shape() {
                return Err(NyError::InvalidSpec(format!(
                    "LinearBounds invariant violated: lower_a_err shape {:?} != lower_a shape {:?}",
                    le.shape(),
                    self.lower_a.shape()
                )));
            }
        }
        if let Some(ue) = self.upper_a_err.as_ref() {
            if ue.shape() != self.upper_a.shape() {
                return Err(NyError::InvalidSpec(format!(
                    "LinearBounds invariant violated: upper_a_err shape {:?} != upper_a shape {:?}",
                    ue.shape(),
                    self.upper_a.shape()
                )));
            }
        }

        Ok(())
    }

    /// Compute concretized bounds in f64 intermediates.
    ///
    /// Uses `safe_mul_for_bounds` for 0*inf=0 handling and f64 accumulation
    /// to minimize rounding error, matching `concretize_l2_ball` and
    /// alpha-beta-CROWN's `cuda_kernels.cu` (f64 intermediates).
    ///
    /// Handles ±Inf coefficients from `safe_add` accumulation (#3032):
    /// rows with Inf bias or CROWN_COEFF_MAX overflow are short-circuited
    /// to ±Inf, and any NaN from the dot product is replaced with ±Inf.
    fn concretize_f64_inner(
        &self,
        input_bounds: &BoundedTensor,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        self.validate_internal_shapes()?;

        let input_flat = input_bounds.flatten();
        let in_l = contiguous_flat_slice(input_flat.lower());
        let in_u = contiguous_flat_slice(input_flat.upper());
        let m = self.lower_a.nrows();
        let n = self.lower_a.ncols();

        self.concretize_scalar_f64(&in_l, &in_u, m, n)
    }

    /// Scalar concretize with per-element NaN/Inf/overflow handling.
    fn concretize_scalar_f64(
        &self,
        in_l: &[f32],
        in_u: &[f32],
        m: usize,
        n: usize,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let mut lower = Array1::<f64>::zeros(m);
        let mut upper = Array1::<f64>::zeros(m);
        // Certified per-coefficient error on the stored A·W coefficients
        // (#vnncomp-aw-soundness). When present, the lower bound is penalized by
        // -Σ_j max(|in_l|,|in_u|)·err and the upper bound by +Σ_j max(...)·err,
        // which is provably sound over the box for ANY true coefficient within
        // `[stored-err, stored+err]` (the corner is no longer chosen by a single
        // possibly-wrong f32 sign). Validated at 0 violations / 300k trials.
        let lower_err = self.lower_a_err.as_ref();
        let upper_err = self.upper_a_err.as_ref();
        for i in 0..m {
            // #1932: Defense-in-depth magnitude pre-check. If the bias is already
            // ±inf (from CROWN backward row degradation), skip the dot product for
            // that bound — the row is already maximally loose. Also check for any
            // A coefficient exceeding CROWN_COEFF_MAX, which should not happen if
            // backward paths are working correctly but could occur from unprotected
            // secondary paths.
            //
            // Lower and upper are handled independently: a degraded lower bound
            // (lower_b = -inf) does not force the upper bound to +inf if the upper
            // A-row and bias are well-behaved, and vice versa.
            let lb = self.lower_b[i];
            let ub = self.upper_b[i];
            let lower_degraded = lb == f32::NEG_INFINITY;
            let upper_degraded = ub == f32::INFINITY;
            // Check A-row coefficients for magnitude overflow (secondary path defense).
            let mut lower_row_overflow = false;
            let mut upper_row_overflow = false;
            if !lower_degraded || !upper_degraded {
                for j in 0..n {
                    if !lower_degraded && self.lower_a[[i, j]].abs() > CROWN_COEFF_MAX {
                        lower_row_overflow = true;
                    }
                    if !upper_degraded && self.upper_a[[i, j]].abs() > CROWN_COEFF_MAX {
                        upper_row_overflow = true;
                    }
                    if (lower_degraded || lower_row_overflow)
                        && (upper_degraded || upper_row_overflow)
                    {
                        break;
                    }
                }
            }
            if lower_degraded || lower_row_overflow {
                lower[i] = f64::NEG_INFINITY;
            }
            if upper_degraded || upper_row_overflow {
                upper[i] = f64::INFINITY;
            }
            if (lower_degraded || lower_row_overflow) && (upper_degraded || upper_row_overflow) {
                continue;
            }

            let mut sum_l = lb as f64;
            let mut sum_u = ub as f64;
            // S-scaled coefficient-error penalty (subtracted from lower, added to
            // upper). Accumulated in f64; folded into sum_l/sum_u after the dot.
            let mut err_penalty_l = 0.0f64;
            let mut err_penalty_u = 0.0f64;
            for j in 0..n {
                let in_l = in_l[j];
                let in_u = in_u[j];
                let la = self.lower_a[[i, j]];
                let ua = self.upper_a[[i, j]];
                // Positive/negative split for interval arithmetic, with safe 0*inf=0.
                // Uses NaN-propagating max/min (#2415): Rust's f32::max/min follow
                // IEEE 754-2008, returning the non-NaN argument. This silently absorbs
                // NaN coefficients into 0.0, bypassing the NaN guard below. The
                // nan_propagating variants preserve NaN so it poisons the accumulator.
                //
                // SOUNDNESS (#concretize-soundness-hardening): the per-term product is
                // formed in **f64** (`safe_mul_for_bounds_f64` after promoting both
                // operands), NOT in f32. A f32 product rounds to nearest and can land
                // up to 0.5 f32-ULP *inside* the true product; with cancellation
                // between large positive and negative products (|product| ≫ |result|),
                // the accumulated inward bias can exceed the final `next_*_f32` 1-ULP
                // widening at the result's (small) magnitude — making `concretize_sound`
                // itself unsound. f32×f32 promoted to f64 is EXACT (48 < 53 significand
                // bits), so the only residual error is the (negligible) f64 sum, which
                // the final directed cast covers. The split itself (max/min with 0) is
                // exact in f32 and preserved.
                let (la_pos, la_neg) = (nan_propagating_max_zero(la), nan_propagating_min_zero(la));
                let (ua_pos, ua_neg) = (nan_propagating_max_zero(ua), nan_propagating_min_zero(ua));
                sum_l += safe_mul_for_bounds_f64(la_pos as f64, in_l as f64)
                    + safe_mul_for_bounds_f64(la_neg as f64, in_u as f64);
                sum_u += safe_mul_for_bounds_f64(ua_pos as f64, in_u as f64)
                    + safe_mul_for_bounds_f64(ua_neg as f64, in_l as f64);
                // Worst-case input magnitude on coord j: max(|in_l|, |in_u|).
                if lower_err.is_some() || upper_err.is_some() {
                    let mag = (in_l as f64).abs().max((in_u as f64).abs());
                    if let Some(le) = lower_err {
                        err_penalty_l += le[[i, j]] as f64 * mag;
                    }
                    if let Some(ue) = upper_err {
                        err_penalty_u += ue[[i, j]] as f64 * mag;
                    }
                }
            }
            // Apply the certified-error penalty: lower goes DOWN, upper goes UP.
            // A non-finite penalty (from a degraded err entry) drives the bound to
            // ±inf, which f64_to_bounded_tensor repairs to [-inf, +inf] (sound).
            if err_penalty_l != 0.0 {
                sum_l -= err_penalty_l;
            }
            if err_penalty_u != 0.0 {
                sum_u += err_penalty_u;
            }
            // NaN guard: if NaN entered the accumulator (e.g., from NaN input bounds
            // or NaN coefficients via safe_mul_for_bounds), fall back to conservative
            // bounds matching BatchedLinearBounds::concretize / interval_mul_for_bounds.
            //
            // #3202: Only write back dot-product results for non-degraded bounds.
            // When lower_degraded/lower_row_overflow is set, lower[i] was already
            // set to -inf above. Writing sum_l here would silently overwrite that
            // defense, defeating the CROWN_COEFF_MAX guard. Same for upper.
            if !(lower_degraded || lower_row_overflow) {
                lower[i] = if sum_l.is_nan() {
                    tracing::warn!(
                        "NaN in CROWN concretization lower sum, falling back to -inf: row={i}"
                    );
                    f64::NEG_INFINITY
                } else {
                    sum_l
                };
            }
            if !(upper_degraded || upper_row_overflow) {
                upper[i] = if sum_u.is_nan() {
                    tracing::warn!(
                        "NaN in CROWN concretization upper sum, falling back to +inf: row={i}"
                    );
                    f64::INFINITY
                } else {
                    sum_u
                };
            }
        }
        Ok((lower, upper))
    }

    /// Convert f64 concretization results to a BoundedTensor.
    ///
    /// Guarantees output has no NaN and no inversions (lower > upper).
    /// NaN is already replaced with ±Inf in `concretize_f64_inner`.
    /// Non-finite f32 values (from Inf coefficients produced by `safe_add`
    /// accumulation, #3032) are repaired to `[-inf, +inf]` per-element.
    /// Any remaining inversions (from numerical instability in CROWN backward)
    /// are repaired per-element to `[-inf, +inf]`, which is always a sound
    /// overapproximation. This eliminates the need for post-concretize guards
    /// at every call site (#2287).
    fn f64_to_bounded_tensor(
        &self,
        lower_f64: Array1<f64>,
        upper_f64: Array1<f64>,
        cast_lower: fn(f64) -> f32,
        cast_upper: fn(f64) -> f32,
    ) -> BoundedTensor {
        let mut lower = lower_f64.mapv(cast_lower).into_dyn();
        let mut upper = upper_f64.mapv(cast_upper).into_dyn();
        let lower_shape = lower.shape().to_vec();
        let upper_shape = upper.shape().to_vec();

        if lower_shape != upper_shape {
            tracing::warn!(
                lower_shape = ?lower_shape,
                upper_shape = ?upper_shape,
                num_outputs = self.num_outputs(),
                "LinearBounds::concretize produced mismatched output shapes; returning conservative [-inf, +inf] fallback"
            );
            return Self::conservative_unbounded(self.num_outputs());
        }

        // NaN is replaced with ±Inf in concretize_f64_inner.
        // Fix any remaining inversions per-element: if lower > upper (from numerical
        // instability in CROWN backward), widen that element to [-inf, +inf].
        // This is sound because [-inf, +inf] is always a valid overapproximation.
        let mut repaired = 0usize;
        ndarray::Zip::from(&mut lower)
            .and(&mut upper)
            .for_each(|l, u| {
                if !l.is_finite() || !u.is_finite() {
                    *l = f32::NEG_INFINITY;
                    *u = f32::INFINITY;
                    repaired += 1;
                }
            });
        repaired += repair_inverted_bounds_nd(&mut lower, &mut upper, InversionRepair::WidenToInf);
        if repaired > 0 {
            tracing::debug!(
                repaired,
                num_outputs = self.num_outputs(),
                "LinearBounds::concretize_sound repaired {repaired} non-finite/inverted elements to [-inf, +inf]"
            );
        }

        // After sanitization, new_allow_infinite should always succeed.
        match BoundedTensor::new_allow_infinite(lower, upper) {
            Ok(bt) => bt,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    lower_shape = ?lower_shape,
                    upper_shape = ?upper_shape,
                    num_outputs = self.num_outputs(),
                    "LinearBounds::concretize failed to construct BoundedTensor after sanitization; returning conservative [-inf, +inf] fallback"
                );
                Self::conservative_unbounded(self.num_outputs())
            }
        }
    }

    /// Concretize linear bounds given input bounds (plain, round-to-nearest cast).
    ///
    /// Uses f64 intermediate accumulation for dot products, then a plain
    /// round-to-nearest `v as f32` cast on the final endpoints. This cast is NOT
    /// directed: an endpoint can land up to 0.5 ULP *inside* the true f64 range, so
    /// the returned bound is NOT guaranteed to be a sound over-approximation at the
    /// f32 boundary.
    ///
    /// SOUNDNESS (#concretize-soundness-hardening): every verdict-relevant /
    /// output-spec / intermediate-relaxation-constraining caller MUST use
    /// [`concretize_sound`](Self::concretize_sound) (directed outward rounding)
    /// instead. As of the soundness-hardening sweep there are NO production callers
    /// of this plain method — the sole former production caller
    /// (`network/core/graph/forward_linear.rs::concretize_to_node_shape`, whose
    /// concretized node bounds are intersected with IBP and used to constrain
    /// downstream relaxations) was routed to `concretize_sound`. This method is
    /// retained for tests and for the explicitly non-binding tightening case where
    /// the result is later widened by a sound operation; if you reach for it on a
    /// verdict path, you almost certainly want `concretize_sound`.
    ///
    /// REQUIRES: `input_bounds.numel() == self.num_inputs()`.
    /// ENSURES: `result.shape() == [self.num_outputs()]`.
    pub fn concretize(&self, input_bounds: &BoundedTensor) -> BoundedTensor {
        if let Err(err) = self.validate_internal_shapes() {
            tracing::warn!(
                error = %err,
                lower_a_shape = ?self.lower_a.shape(),
                upper_a_shape = ?self.upper_a.shape(),
                lower_b_len = self.lower_b.len(),
                upper_b_len = self.upper_b.len(),
                "LinearBounds::concretize called with malformed LinearBounds; returning conservative [-inf, +inf] fallback"
            );
            return Self::conservative_unbounded(self.num_outputs());
        }
        let input_numel = input_bounds.len();
        if input_numel != self.num_inputs() {
            tracing::warn!(
                expected = self.num_inputs(),
                got = input_numel,
                "LinearBounds::concretize input dimension mismatch; returning conservative [-inf, +inf] fallback"
            );
            return Self::conservative_unbounded(self.num_outputs());
        }
        // Shape already validated above; concretize_f64_inner re-checks as defense-in-depth.
        let (lower, upper) = match self.concretize_f64_inner(input_bounds) {
            Ok(pair) => pair,
            Err(err) => {
                tracing::warn!(error = %err, "concretize_f64_inner failed despite pre-validation");
                return Self::conservative_unbounded(self.num_outputs());
            }
        };
        self.f64_to_bounded_tensor(lower, upper, |v| v as f32, |v| v as f32)
    }

    /// Concretize linear bounds with a flattened shape check.
    ///
    /// # Errors
    /// - `NyError::ShapeMismatch` if the input length does not match `num_inputs`.
    pub fn concretize_checked(&self, input_bounds: &BoundedTensor) -> Result<BoundedTensor> {
        self.validate_internal_shapes()?;
        if input_bounds.len() != self.num_inputs() {
            return Err(NyError::shape_mismatch(
                vec![self.num_inputs()],
                input_bounds.shape().to_vec(),
            ));
        }
        // #2239: directed rounding on f64→f32 for soundness.
        Ok(self.concretize_sound(input_bounds))
    }

    /// Concretize with directed rounding on the f64→f32 boundary for soundness.
    ///
    /// Uses f64 intermediates and applies `next_down_f32`/`next_up_f32` on the
    /// f64→f32 cast, matching alpha-beta-CROWN's `__double2float_rd`/`__double2float_ru`.
    ///
    /// REQUIRES: `input_bounds.numel() == self.num_inputs()`.
    /// ENSURES: `result.lower()[i]` is a sound lower bound (rounded toward -∞).
    /// ENSURES: `result.upper()[i]` is a sound upper bound (rounded toward +∞).
    pub fn concretize_sound(&self, input_bounds: &BoundedTensor) -> BoundedTensor {
        if let Err(err) = self.validate_internal_shapes() {
            tracing::warn!(
                error = %err,
                lower_a_shape = ?self.lower_a.shape(),
                upper_a_shape = ?self.upper_a.shape(),
                lower_b_len = self.lower_b.len(),
                upper_b_len = self.upper_b.len(),
                "LinearBounds::concretize_sound called with malformed LinearBounds; returning conservative [-inf, +inf] fallback"
            );
            return Self::conservative_unbounded(self.num_outputs());
        }
        let input_numel = input_bounds.len();
        if input_numel != self.num_inputs() {
            tracing::warn!(
                expected = self.num_inputs(),
                got = input_numel,
                "LinearBounds::concretize_sound input dimension mismatch; returning conservative [-inf, +inf] fallback"
            );
            return Self::conservative_unbounded(self.num_outputs());
        }
        // Shape already validated above; concretize_f64_inner re-checks as defense-in-depth.
        let (lower, upper) = match self.concretize_f64_inner(input_bounds) {
            Ok(pair) => pair,
            Err(err) => {
                tracing::warn!(error = %err, "concretize_f64_inner failed despite pre-validation");
                return Self::conservative_unbounded(self.num_outputs());
            }
        };
        self.f64_to_bounded_tensor(
            lower,
            upper,
            |v| next_down_f32(v as f32),
            |v| next_up_f32(v as f32),
        )
    }

    /// Concretize linear bounds over an ℓ2 ball input set.
    ///
    /// For bounds of the form:
    /// - Lower: y >= a_L^T x + b_L
    /// - Upper: y <= a_U^T x + b_U
    ///
    /// and input constraint `||x - x_hat||_2 <= rho`, the extrema of a linear function
    /// occur in the direction of the coefficient vector:
    /// - min_x a^T x = a^T x_hat - rho * ||a||_2
    /// - max_x a^T x = a^T x_hat + rho * ||a||_2
    ///
    /// REQUIRES: `rho >= 0.0`.
    /// REQUIRES: `x_hat.len() == self.num_inputs()` (dimension match).
    ///     ENSURES: `result.shape() == [self.num_outputs()]`.
    /// ENSURES: For each output i and any `x` s.t. `||x - x_hat||_2 <= rho`:
    ///   - `result.lower()[i] <= lower_a[i]^T x + lower_b[i]`,
    ///   - `result.upper()[i] >= upper_a[i]^T x + upper_b[i]`.
    pub fn concretize_l2_ball(&self, x_hat: &Array1<f32>, rho: f32) -> Result<BoundedTensor> {
        self.validate_internal_shapes()?;
        if rho < 0.0 {
            return Err(NyError::InvalidSpec(format!(
                "rho must be >= 0 (got {rho})"
            )));
        }
        if self.num_inputs() != x_hat.len() {
            return Err(NyError::shape_mismatch(
                vec![self.num_inputs()],
                vec![x_hat.len()],
            ));
        }

        let m = self.num_outputs();
        let n = self.num_inputs();
        let rho_f64 = rho as f64;

        let mut lower = Array1::<f32>::zeros(m);
        let mut upper = Array1::<f32>::zeros(m);

        for i in 0..m {
            let mut dot_l = self.lower_b[i] as f64;
            let mut dot_u = self.upper_b[i] as f64;
            let mut norm_l2_l = 0.0f64;
            let mut norm_l2_u = 0.0f64;
            for j in 0..n {
                let xj = x_hat[j] as f64;
                let al = self.lower_a[[i, j]] as f64;
                let au = self.upper_a[[i, j]] as f64;
                dot_l += al * xj;
                dot_u += au * xj;
                norm_l2_l += al * al;
                norm_l2_u += au * au;
            }
            let norm_l2_l = norm_l2_l.sqrt();
            let norm_l2_u = norm_l2_u.sqrt();
            // Apply directed rounding on f64→f32 cast for soundness.
            // Lower bound rounds toward -∞, upper bound rounds toward +∞.
            // Reference: alpha-beta-CROWN uses __double2float_rd/__double2float_ru
            // CUDA intrinsics for the same purpose (cuda_kernels.cu:8-22).
            let l_val = next_down_f32((dot_l - rho_f64 * norm_l2_l) as f32);
            let u_val = next_up_f32((dot_u + rho_f64 * norm_l2_u) as f32);
            // Guard: NaN from Inf-Inf subtraction → conservative [-Inf, +Inf].
            lower[i] = if l_val.is_nan() {
                f32::NEG_INFINITY
            } else {
                l_val
            };
            upper[i] = if u_val.is_nan() { f32::INFINITY } else { u_val };
        }

        // Repair inversions: if lower > upper (from numerical instability in CROWN
        // backward coefficients), widen that element to [-inf, +inf].
        // Note: f64_to_bounded_tensor also repairs non-finite values; here we skip
        // that because the NaN guard above already ensures lower ∈ {finite, -Inf}
        // and upper ∈ {finite, +Inf} — both are valid conservative bounds.
        let mut lower = lower.into_dyn();
        let mut upper = upper.into_dyn();
        let repaired =
            repair_inverted_bounds_nd(&mut lower, &mut upper, InversionRepair::WidenToInf);
        if repaired > 0 {
            tracing::debug!(
                repaired,
                num_outputs = m,
                "LinearBounds::concretize_l2_ball repaired {repaired} inverted elements to [-inf, +inf]"
            );
        }

        // Inf bounds are sound (conservative); NaN and inversions have been repaired above.
        BoundedTensor::new_allow_infinite(lower, upper)
    }
}
