// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{
    checked_shape_product,
    dd::{next_down_f64, next_up_f64},
    NyError, Result,
};
use tracing::warn;

use super::{certified_affine_sum_f32, OutwardDirection};

/// Fast scan for any NaN element in an `f32` array.
///
/// Behaviourally equivalent to `arr.iter().any(|v| v.is_nan())`, but scans a
/// flat `&[f32]` slice when the array is in standard (contiguous, C-order)
/// layout — the common case for CROWN coefficient tensors. The slice loop has
/// no per-element stride bookkeeping and autovectorizes, which matters because
/// this firewall runs on every `BatchedLinearBounds` construction. Strided
/// arrays fall back to the element iterator, preserving identical results
/// (`.any()` over a pure predicate is order-independent). Unlike the dense
/// `LinearBounds` firewall, ±Inf is permitted here (legitimate conservative
/// coefficients from `compose()`), so only NaN is matched.
#[inline]
fn any_nan(arr: &ArrayD<f32>) -> bool {
    match arr.as_slice() {
        Some(slice) => slice.iter().any(|v| v.is_nan()),
        None => arr.iter().any(|v| v.is_nan()),
    }
}

mod compose;
mod compose_blas;
mod concretize;
mod concretize_posneg;
mod interval;
mod matvec;
mod reshape;

// Test-only re-exports: used by bounds/mod.rs → bounds/tests/interval_arithmetic.rs.
// Production use of batched_interval_matvec_checked is in concretize.rs via direct
// super::interval:: import, not through this re-export.
#[cfg(test)]
pub(crate) use interval::batched_interval_matvec;
#[cfg(test)]
pub(crate) use interval::batched_interval_matvec_checked;
#[cfg(test)]
pub(crate) use matvec::batched_matvec;

/// N-D batched linear bounds for transformer verification.
///
/// Unlike `LinearBounds` which flattens everything to 2D, this maintains
/// the batch structure (e.g., [batch, seq, hidden]) and operates on the
/// last dimension only, following Auto-LiRPA's approach.
///
/// For input shape [...batch_dims, in_dim] and output shape [...batch_dims, out_dim]:
/// - lower_a: [...batch_dims, out_dim, in_dim] - coefficient matrix per position
/// - lower_b: [...batch_dims, out_dim] - bias per position
///
/// The backward pass broadcasts correctly: for y = Wx + b,
/// new_A = A @ W broadcasts over batch dimensions.
#[derive(Debug, Clone)]
#[must_use = "BatchedLinearBounds from CROWN propagation should not be silently discarded"]
pub struct BatchedLinearBounds {
    /// Lower bound coefficient matrix: shape [...batch_dims, out_dim, in_dim]
    /// Represents: lower(y) >= A_L @ x + b_L
    pub(crate) lower_a: ArrayD<f32>,
    /// Lower bound bias: shape [...batch_dims, out_dim]
    pub(crate) lower_b: ArrayD<f32>,
    /// Upper bound coefficient matrix: shape [...batch_dims, out_dim, in_dim]
    /// Represents: upper(y) <= A_U @ x + b_U
    pub(crate) upper_a: ArrayD<f32>,
    /// Upper bound bias: shape [...batch_dims, out_dim]
    pub(crate) upper_b: ArrayD<f32>,
    /// Input shape (what x represents): e.g., [batch, seq, hidden]
    pub(crate) input_shape: Vec<usize>,
    /// Output shape (what y represents): e.g., [batch, seq, hidden]
    pub(crate) output_shape: Vec<usize>,
    /// Certified per-coefficient error on `lower_a` (#vnncomp-aw-soundness). Same
    /// shape as `lower_a` when present; `None` means exact (error 0). Mirrors
    /// [`LinearBounds::lower_a_err`](crate::LinearBounds). Populated by the batched
    /// Linear CROWN backward (the `A·W` f64-accumulation + f32-cast error,
    /// `γ_n·S`-scaled) and consumed at [`concretize`](Self::concretize) /
    /// [`concretize_sound`](Self::concretize_sound) via the same
    /// `max(|in_l|,|in_u|)`-scaled OUTWARD penalty as the scalar path — making the
    /// batched (β-CROWN / BaB) verdict path carry the SAME certified soundness
    /// margin as the scalar path rather than being ~1 ULP optimistic.
    pub(crate) lower_a_err: Option<ArrayD<f32>>,
    /// Certified per-coefficient error on `upper_a`. Mirror of `lower_a_err`.
    pub(crate) upper_a_err: Option<ArrayD<f32>>,
}

impl BatchedLinearBounds {
    /// Create BatchedLinearBounds with shape and NaN validation.
    ///
    /// # Invariants
    /// - `lower_a.ndim() >= 2` (at least [out_dim, in_dim])
    /// - `upper_a.shape() == lower_a.shape()`
    /// - `lower_b.shape() == lower_a.shape()[..ndim-1]` (bias = A without last dim)
    /// - `upper_b.shape() == lower_b.shape()`
    /// - No NaN in any array (±Inf allowed — compose() legitimately produces
    ///   ±Inf coefficients as conservative NaN guards)
    ///
    /// # Errors
    /// - Returns `NyError::InvalidSpec` when the coefficient rank is invalid.
    /// - Returns `NyError::ShapeMismatch` when paired coefficient, bias, or
    ///   error-carrier shapes disagree.
    /// - Returns `NyError::NumericalInstability` on NaN detection.
    pub fn new(
        lower_a: ArrayD<f32>,
        lower_b: ArrayD<f32>,
        upper_a: ArrayD<f32>,
        upper_b: ArrayD<f32>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    ) -> Result<Self> {
        let bounds = Self {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            input_shape,
            output_shape,
            lower_a_err: None,
            upper_a_err: None,
        };
        bounds.validate_internal_shapes()?;
        bounds.validate_no_nan()?;
        Ok(bounds)
    }

    /// Validate shape consistency of coefficient and bias arrays.
    pub(crate) fn validate_internal_shapes(&self) -> Result<()> {
        if self.lower_a.ndim() < 2 {
            return Err(NyError::InvalidSpec(format!(
                "BatchedLinearBounds: lower_a must have ndim >= 2, got {}",
                self.lower_a.ndim()
            )));
        }
        if self.lower_a.shape() != self.upper_a.shape() {
            return Err(NyError::shape_mismatch(
                self.lower_a.shape().to_vec(),
                self.upper_a.shape().to_vec(),
            ));
        }
        if self.lower_b.shape() != self.upper_b.shape() {
            return Err(NyError::shape_mismatch(
                self.lower_b.shape().to_vec(),
                self.upper_b.shape().to_vec(),
            ));
        }
        // bias shape = A shape without last dimension
        let expected_b_shape: Vec<usize> = self.lower_a.shape()[..self.lower_a.ndim() - 1].to_vec();
        if self.lower_b.shape() != expected_b_shape.as_slice() {
            return Err(NyError::shape_mismatch(
                expected_b_shape,
                self.lower_b.shape().to_vec(),
            ));
        }
        if let Some(error) = self
            .lower_a_err
            .as_ref()
            .filter(|error| error.shape() != self.lower_a.shape())
        {
            return Err(NyError::shape_mismatch(
                self.lower_a.shape().to_vec(),
                error.shape().to_vec(),
            ));
        }
        if let Some(error) = self
            .upper_a_err
            .as_ref()
            .filter(|error| error.shape() != self.upper_a.shape())
        {
            return Err(NyError::shape_mismatch(
                self.upper_a.shape().to_vec(),
                error.shape().to_vec(),
            ));
        }
        Ok(())
    }

    /// Validate that no array contains NaN.
    ///
    /// Unlike LinearBounds which rejects Inf in coefficients, BatchedLinearBounds
    /// allows ±Inf because `compose()` legitimately produces ±Inf coefficients
    /// as conservative NaN fallbacks (lines 560-569 of compose()). Only NaN is
    /// rejected — it represents a computation error, not a conservative bound.
    ///
    /// Reference: designs/2026-02-26-batched-linear-bounds-encapsulation.md
    pub(crate) fn validate_no_nan(&self) -> Result<()> {
        if any_nan(&self.lower_a) {
            return Err(NyError::NumericalInstability(
                "BatchedLinearBounds lower_a contains NaN".into(),
            ));
        }
        if any_nan(&self.upper_a) {
            return Err(NyError::NumericalInstability(
                "BatchedLinearBounds upper_a contains NaN".into(),
            ));
        }
        if any_nan(&self.lower_b) {
            return Err(NyError::NumericalInstability(
                "BatchedLinearBounds lower_b contains NaN".into(),
            ));
        }
        if any_nan(&self.upper_b) {
            return Err(NyError::NumericalInstability(
                "BatchedLinearBounds upper_b contains NaN".into(),
            ));
        }
        if self
            .lower_a_err
            .iter()
            .chain(self.upper_a_err.iter())
            .flat_map(|error| error.iter())
            .any(|&value| value.is_nan() || value < 0.0)
        {
            return Err(NyError::NumericalInstability(
                "BatchedLinearBounds coefficient error must be non-negative and non-NaN".into(),
            ));
        }
        Ok(())
    }

    /// Conservative fallback bounds: coefficients = 0, biases = ±Inf.
    ///
    /// Used when CROWN backward fails or produces numerically unstable results.
    /// The resulting bounds are `(-inf, +inf)` for all outputs — always sound
    /// but maximally imprecise. Analogous to [`LinearBounds::conservative()`].
    ///
    /// Known-safe by construction: zero coefficients and ±Inf biases contain
    /// no NaN.
    pub fn conservative(input_shape: Vec<usize>, output_shape: Vec<usize>) -> Self {
        let in_dim = input_shape.last().copied().unwrap_or(1);
        let out_dim = output_shape.last().copied().unwrap_or(1);
        let batch_dims: Vec<usize> = output_shape[..output_shape.len().saturating_sub(1)].to_vec();

        let mut a_shape = batch_dims.clone();
        a_shape.push(out_dim);
        a_shape.push(in_dim);

        let mut b_shape = batch_dims;
        b_shape.push(out_dim);

        Self {
            lower_a: ArrayD::zeros(IxDyn(&a_shape)),
            lower_b: ArrayD::from_elem(IxDyn(&b_shape), f32::NEG_INFINITY),
            upper_a: ArrayD::zeros(IxDyn(&a_shape)),
            upper_b: ArrayD::from_elem(IxDyn(&b_shape), f32::INFINITY),
            input_shape,
            output_shape,
            lower_a_err: None,
            upper_a_err: None,
        }
    }

    /// Validated construction that falls back to conservative bounds on NaN.
    ///
    /// This is the batched CROWN backward NaN firewall (Tier 2 of the NaN
    /// strategy, designs/2026-02-25-nan-strategy-unification.md). When CROWN
    /// backward produces NaN coefficients due to numerical instability, instead
    /// of returning `Err(NumericalInstability)`, this returns conservative
    /// bounds (A=0, b=±∞) that are sound but maximally imprecise.
    ///
    /// Use this at CROWN backward output points where the caller would
    /// otherwise discard the entire verification attempt on NaN. For
    /// construction sites where NaN should be a hard error, use
    /// [`new()`](Self::new) instead.
    ///
    /// Reference: #2812 (IBP-vs-CROWN NaN defense asymmetry)
    pub fn new_or_conservative(
        lower_a: ArrayD<f32>,
        lower_b: ArrayD<f32>,
        upper_a: ArrayD<f32>,
        upper_b: ArrayD<f32>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    ) -> Result<Self> {
        let bounds = Self {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            input_shape,
            output_shape,
            lower_a_err: None,
            upper_a_err: None,
        };
        bounds.validate_internal_shapes()?;
        match bounds.validate_no_nan() {
            Ok(()) => Ok(bounds),
            Err(_) => {
                warn!(
                    "BatchedLinearBounds NaN firewall: CROWN backward produced NaN \
                     coefficients (input {:?}, output {:?}), falling back to conservative bounds",
                    bounds.input_shape, bounds.output_shape
                );
                Ok(Self::conservative(bounds.input_shape, bounds.output_shape))
            }
        }
    }

    // --- Read-only accessors ---

    /// Lower bound coefficient matrix: shape [...batch_dims, out_dim, in_dim].
    pub fn lower_a(&self) -> &ArrayD<f32> {
        &self.lower_a
    }

    /// Upper bound coefficient matrix: shape [...batch_dims, out_dim, in_dim].
    pub fn upper_a(&self) -> &ArrayD<f32> {
        &self.upper_a
    }

    /// Lower bound bias: shape [...batch_dims, out_dim].
    pub fn lower_b(&self) -> &ArrayD<f32> {
        &self.lower_b
    }

    /// Upper bound bias: shape [...batch_dims, out_dim].
    pub fn upper_b(&self) -> &ArrayD<f32> {
        &self.upper_b
    }

    /// Whether this object carries any certified coefficient error.
    pub(crate) fn has_coeff_err(&self) -> bool {
        self.lower_a_err.is_some() || self.upper_a_err.is_some()
    }

    /// Attach certified coefficient-error matrices (#vnncomp-aw-soundness).
    ///
    /// Shapes must match `lower_a`/`upper_a`; a mismatch degrades the whole
    /// carrier conservatively. Negative or non-finite entries become `+inf`,
    /// which degrades affected rows when the error is discharged.
    pub(crate) fn set_coeff_err(&mut self, lower_err: ArrayD<f32>, upper_err: ArrayD<f32>) {
        if lower_err.shape() != self.lower_a.shape() || upper_err.shape() != self.upper_a.shape() {
            *self = Self::conservative(self.input_shape.clone(), self.output_shape.clone());
            return;
        }
        let sanitize = |value: f32| {
            if value.is_finite() && value >= 0.0 {
                value
            } else {
                f32::INFINITY
            }
        };
        self.lower_a_err = Some(lower_err.mapv(sanitize));
        self.upper_a_err = Some(upper_err.mapv(sanitize));
    }

    /// Build a "carrier" whose coefficient matrices ARE this object's certified
    /// error matrices (`|err|`, non-negative) and whose bias is zero
    /// (#vnncomp-aw-soundness). Batched analogue of
    /// [`LinearBounds::coeff_err_carrier`] — used to propagate the error through an
    /// EXACT-linear batched op by re-running the op on the carrier. Returns `None`
    /// when there is no error to carry.
    pub(crate) fn coeff_err_carrier(&self) -> Option<BatchedLinearBounds> {
        if !self.has_coeff_err() {
            return None;
        }
        let lower_carrier = self
            .lower_a_err
            .as_ref()
            .map(|e| e.mapv(|v| v.abs()))
            .unwrap_or_else(|| ArrayD::zeros(self.lower_a.raw_dim()));
        let upper_carrier = self
            .upper_a_err
            .as_ref()
            .map(|e| e.mapv(|v| v.abs()))
            .unwrap_or_else(|| ArrayD::zeros(self.upper_a.raw_dim()));
        Some(BatchedLinearBounds {
            lower_a: lower_carrier,
            lower_b: ArrayD::zeros(self.lower_b.raw_dim()),
            upper_a: upper_carrier,
            upper_b: ArrayD::zeros(self.upper_b.raw_dim()),
            input_shape: self.input_shape.clone(),
            output_shape: self.output_shape.clone(),
            lower_a_err: None,
            upper_a_err: None,
        })
    }

    /// Attach the certified error derived from running an exact-linear batched op
    /// on a [`coeff_err_carrier`](Self::coeff_err_carrier) (#vnncomp-aw-soundness).
    /// `carried`'s coefficient magnitudes become the new per-coefficient error
    /// (`|coeff|`, next_up-rounded) and its bias magnitudes widen `self`'s bias
    /// OUTWARD. Shapes must match `self`; on mismatch the error is dropped after a
    /// conservative degrade (so no tightness is claimed without the penalty).
    pub(crate) fn attach_err_from_carried(&mut self, carried: &BatchedLinearBounds) {
        if self.validate_internal_shapes().is_err()
            || self.validate_no_nan().is_err()
            || carried.validate_internal_shapes().is_err()
            || carried.validate_no_nan().is_err()
            || carried.lower_a.shape() != self.lower_a.shape()
            || carried.upper_a.shape() != self.upper_a.shape()
            || carried.lower_b.shape() != self.lower_b.shape()
            || carried.upper_b.shape() != self.upper_b.shape()
        {
            // The real result may not itself carry a fresh error.  Calling
            // `discharge_coeff_err_to_conservative` in that case is a no-op and
            // would silently lose the incoming carrier.  Degrade explicitly.
            *self = Self::conservative(self.input_shape.clone(), self.output_shape.clone());
            return;
        }
        // ADD the carried error to any error the op already attached on its plain
        // run (Linear/Conv produce a fresh `γ_n·S` error AND must compose the
        // incoming error `err_in @ |W|` carried here). Pure carriers have no plain
        // error, so this is just the carried value.
        let add_err = |existing: &Option<ArrayD<f32>>, carried_a: &ArrayD<f32>| -> ArrayD<f32> {
            let mut out = carried_a.mapv(|v| v.abs());
            if let Some(e) = existing {
                if e.shape() == out.shape() {
                    out = &out + e;
                }
            }
            out.mapv(|v| {
                if v.is_finite() {
                    ny_tensor::next_up_f32(v)
                } else {
                    f32::INFINITY
                }
            })
        };
        let new_lower = add_err(&self.lower_a_err, &carried.lower_a);
        let new_upper = add_err(&self.upper_a_err, &carried.upper_a);
        self.lower_a_err = Some(new_lower);
        self.upper_a_err = Some(new_upper);
        // Fold the carried bias magnitude OUTWARD into self's bias.
        for (((lb, ub), cl), cu) in self
            .lower_b
            .iter_mut()
            .zip(self.upper_b.iter_mut())
            .zip(carried.lower_b.iter())
            .zip(carried.upper_b.iter())
        {
            let mag = cl.abs().max(cu.abs());
            if mag != 0.0 && mag.is_finite() {
                *lb = ny_tensor::next_down_f32(*lb - mag);
                *ub = ny_tensor::next_up_f32(*ub + mag);
            } else if !mag.is_finite() {
                *lb = f32::NEG_INFINITY;
                *ub = f32::INFINITY;
            }
        }
    }

    /// EAGERLY discharge the certified coefficient error over a box, per output
    /// position, keeping positions whose penalty is non-finite carried
    /// (#cgan-conv-err-compose). Batched counterpart of
    /// `LinearBounds::fold_coeff_err_over_box_eager` — the same fold identity as
    /// [`fold_coeff_err_into_bias`](Self::fold_coeff_err_into_bias), different
    /// POLICY: called right after an elementwise-activation batched backward,
    /// where the coefficient columns multiply the (typically CROWN-tightened)
    /// pre-activation cut, so the u-scale relative coefficient error is
    /// discharged over the tightest box it will ever see instead of compounding
    /// through the ABS-composition of every remaining backward layer (IBP-scale
    /// magnitudes). Keeps the batched (BaB multi-domain rebound) path consistent
    /// with the scalar path's eager fold (multi_objective_parity moat).
    ///
    /// On any shape mismatch, or for positions with a non-finite penalty, the
    /// error entries are KEPT (carried) — byte-identical to the prior behavior,
    /// never a new degrade.
    pub(crate) fn fold_coeff_err_over_box_eager(&mut self, in_l: &[f32], in_u: &[f32]) {
        if !self.has_coeff_err() {
            return;
        }
        if self.validate_internal_shapes().is_err() || self.validate_no_nan().is_err() {
            *self = Self::conservative(self.input_shape.clone(), self.output_shape.clone());
            return;
        }
        let a_shape = self.lower_a.shape().to_vec();
        if a_shape.len() < 2 {
            return; // keep carrying
        }
        let n = a_shape[a_shape.len() - 1];
        if in_l.len() != n || in_u.len() != n {
            return; // keep carrying
        }
        let total_pos: usize = self.lower_b.len();
        let mut mag = vec![0.0f32; n];
        for j in 0..n {
            mag[j] = in_l[j].abs().max(in_u[j].abs());
        }
        let fold_side =
            |err_opt: &mut Option<ArrayD<f32>>, bias: &mut ArrayD<f32>, lower_side: bool| {
                // Pre-check both views before taking anything, so a failure keeps
                // the error carried instead of silently dropping it.
                let ok = err_opt
                    .as_ref()
                    .is_some_and(|e| e.len() == total_pos * n && e.as_slice().is_some())
                    && bias.as_slice_mut().is_some();
                if !ok {
                    return;
                }
                let e = err_opt.take().expect("checked above");
                let err_shape = e.raw_dim();
                let mut e2d = match e.into_shape_with_order((total_pos, n)) {
                    Ok(e2d) => e2d,
                    Err(_) => return, // unreachable given the len check; keep None-safe
                };
                let b = bias.as_slice_mut().expect("checked above");
                let mut any_kept = false;
                for p in 0..total_pos {
                    let pen = certified_affine_sum_f32(
                        0.0,
                        (0..n).map(|j| (e2d[[p, j]], mag[j])),
                        OutwardDirection::Upper,
                    );
                    if pen.is_finite() {
                        if pen != 0.0 {
                            b[p] = if lower_side {
                                ny_tensor::next_down_f32(next_down_f64(b[p] as f64 - pen) as f32)
                            } else {
                                ny_tensor::next_up_f32(next_up_f64(b[p] as f64 + pen) as f32)
                            };
                        }
                        for j in 0..n {
                            e2d[[p, j]] = 0.0;
                        }
                    } else {
                        any_kept = true;
                    }
                }
                if any_kept {
                    if let Ok(back) = e2d.into_dyn().into_shape_with_order(err_shape) {
                        *err_opt = Some(back);
                    } else {
                        // Shape restore cannot fail (same element count); if it ever
                        // does, degrade OUTWARD rather than dropping the error.
                        for v in bias.iter_mut() {
                            *v = if lower_side {
                                f32::NEG_INFINITY
                            } else {
                                f32::INFINITY
                            };
                        }
                    }
                }
            };
        {
            let (l_err, lower_b) = (&mut self.lower_a_err, &mut self.lower_b);
            fold_side(l_err, lower_b, true);
        }
        {
            let (u_err, upper_b) = (&mut self.upper_a_err, &mut self.upper_b);
            fold_side(u_err, upper_b, false);
        }
    }

    /// Soundly discharge any certified coefficient error by folding it into the
    /// BIAS over a known input box, then clearing the error (#vnncomp-aw-soundness).
    ///
    /// The precise (tight) analogue of [`LinearBounds::fold_coeff_err_into_bias`]:
    /// the coefficients multiply the value whose box is `in_l`/`in_u` (flattened to
    /// the coefficient's last dim), so the certified interval `[A-err, A+err]`
    /// contributes at most `penalty = Σ_j max(|in_l_j|,|in_u_j|)·err_ij` per output
    /// position, which we subtract from `lower_b` and add to `upper_b` OUTWARD. The
    /// resulting bounds are error-free and sound when any downstream backward op
    /// further transforms them. A length mismatch (cannot align the box to the
    /// coefficient columns) falls back to the conservative position degrade.
    pub(crate) fn fold_coeff_err_into_bias(&mut self, in_l: &[f32], in_u: &[f32]) {
        if !self.has_coeff_err() {
            return;
        }
        if self.validate_internal_shapes().is_err() || self.validate_no_nan().is_err() {
            *self = Self::conservative(self.input_shape.clone(), self.output_shape.clone());
            return;
        }
        let fold_storage_is_contiguous = self.lower_b.as_slice().is_some()
            && self.upper_b.as_slice().is_some()
            && self
                .lower_a_err
                .as_ref()
                .is_none_or(|error| error.as_slice().is_some())
            && self
                .upper_a_err
                .as_ref()
                .is_none_or(|error| error.as_slice().is_some());
        if !fold_storage_is_contiguous {
            // The implementation below consumes the carrier before reshaping it.
            // A strided owned array can be shape-valid yet non-reshapeable; do
            // not let that exceptional layout silently discard the proof error.
            *self = Self::conservative(self.input_shape.clone(), self.output_shape.clone());
            return;
        }
        let a_shape = self.lower_a.shape().to_vec();
        if a_shape.len() < 2 {
            self.discharge_coeff_err_to_conservative();
            return;
        }
        let n = a_shape[a_shape.len() - 1];
        if in_l.len() != n || in_u.len() != n {
            self.discharge_coeff_err_to_conservative();
            return;
        }
        let total_pos: usize = self.lower_b.len();
        let mut mag = vec![0.0f32; n];
        for j in 0..n {
            mag[j] = in_l[j].abs().max(in_u[j].abs());
        }
        if let Some(le) = self.lower_a_err.take() {
            if let (Ok(e2d), Some(lb)) = (
                le.into_shape_with_order((total_pos, n)),
                self.lower_b.as_slice_mut(),
            ) {
                for p in 0..total_pos {
                    let pen = certified_affine_sum_f32(
                        0.0,
                        (0..n).map(|j| (e2d[[p, j]], mag[j])),
                        OutwardDirection::Upper,
                    );
                    if pen != 0.0 {
                        lb[p] = if pen.is_finite() {
                            ny_tensor::next_down_f32(next_down_f64(lb[p] as f64 - pen) as f32)
                        } else {
                            f32::NEG_INFINITY
                        };
                    }
                }
            }
        }
        if let Some(ue) = self.upper_a_err.take() {
            if let (Ok(e2d), Some(ub)) = (
                ue.into_shape_with_order((total_pos, n)),
                self.upper_b.as_slice_mut(),
            ) {
                for p in 0..total_pos {
                    let pen = certified_affine_sum_f32(
                        0.0,
                        (0..n).map(|j| (e2d[[p, j]], mag[j])),
                        OutwardDirection::Upper,
                    );
                    if pen != 0.0 {
                        ub[p] = if pen.is_finite() {
                            ny_tensor::next_up_f32(next_up_f64(ub[p] as f64 + pen) as f32)
                        } else {
                            f32::INFINITY
                        };
                    }
                }
            }
        }
    }

    /// Soundly discharge any certified coefficient error by degrading every
    /// affected output position to a conservative `[-inf, +inf]` bound, then
    /// clearing the error (#vnncomp-aw-soundness). Mirrors
    /// [`LinearBounds::discharge_coeff_err_to_conservative`](crate::LinearBounds::discharge_coeff_err_to_conservative).
    ///
    /// Used by `propagate_crown_backward_batched` before handing an error-carrying
    /// batched bounds object to a layer whose batched backward reconstructs the
    /// coefficient matrix WITHOUT the error matrices (which would silently drop the
    /// soundness penalty — making bounds optimistically tight). For each output
    /// position `[...batch, i]` with any nonzero lower (resp. upper) error, the
    /// position's lower (resp. upper) coefficients are zeroed and the bias set to
    /// `-inf` (resp. `+inf`). Positions with zero error stay fully precise.
    pub(crate) fn discharge_coeff_err_to_conservative(&mut self) {
        if !self.has_coeff_err() {
            return;
        }
        if self.validate_internal_shapes().is_err() || self.validate_no_nan().is_err() {
            *self = Self::conservative(self.input_shape.clone(), self.output_shape.clone());
            return;
        }
        let discharge_storage_is_contiguous = self.lower_a.as_slice().is_some()
            && self.upper_a.as_slice().is_some()
            && self.lower_b.as_slice().is_some()
            && self.upper_b.as_slice().is_some()
            && self
                .lower_a_err
                .as_ref()
                .is_none_or(|error| error.as_slice().is_some())
            && self
                .upper_a_err
                .as_ref()
                .is_none_or(|error| error.as_slice().is_some());
        if !discharge_storage_is_contiguous {
            *self = Self::conservative(self.input_shape.clone(), self.output_shape.clone());
            return;
        }
        let a_shape = self.lower_a.shape().to_vec();
        if a_shape.len() < 2 {
            self.lower_a_err = None;
            self.upper_a_err = None;
            return;
        }
        let m = a_shape[a_shape.len() - 2];
        let n = a_shape[a_shape.len() - 1];
        let total_pos: usize = self.lower_b.len();
        if let Some(le) = self.lower_a_err.take() {
            if let (Ok(mut a2d), Ok(e2d)) = (
                std::mem::take(&mut self.lower_a).into_shape_with_order((total_pos, n)),
                le.into_shape_with_order((total_pos, n)),
            ) {
                if let Some(lb) = self.lower_b.as_slice_mut() {
                    for p in 0..total_pos {
                        let has_err = (0..n).any(|j| e2d[[p, j]] != 0.0);
                        if has_err {
                            for j in 0..n {
                                a2d[[p, j]] = 0.0;
                            }
                            lb[p] = f32::NEG_INFINITY;
                        }
                    }
                }
                self.lower_a = a2d
                    .into_shape_with_order(IxDyn(&a_shape))
                    .unwrap_or_else(|_| ArrayD::zeros(IxDyn(&a_shape)));
            }
        }
        if let Some(ue) = self.upper_a_err.take() {
            if let (Ok(mut a2d), Ok(e2d)) = (
                std::mem::take(&mut self.upper_a).into_shape_with_order((total_pos, n)),
                ue.into_shape_with_order((total_pos, n)),
            ) {
                if let Some(ub) = self.upper_b.as_slice_mut() {
                    for p in 0..total_pos {
                        let has_err = (0..n).any(|j| e2d[[p, j]] != 0.0);
                        if has_err {
                            for j in 0..n {
                                a2d[[p, j]] = 0.0;
                            }
                            ub[p] = f32::INFINITY;
                        }
                    }
                }
                self.upper_a = a2d
                    .into_shape_with_order(IxDyn(&a_shape))
                    .unwrap_or_else(|_| ArrayD::zeros(IxDyn(&a_shape)));
            }
        }
        let _ = m;
    }

    /// Input shape (what x represents).
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    /// Output shape (what y represents).
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    // --- Destructuring ---

    /// Consume self and return the six components.
    ///
    /// Returns `(lower_a, lower_b, upper_a, upper_b, input_shape, output_shape)`.
    // Justification: this is a destructuring method returning all struct fields as a tuple.
    // A named struct would add indirection for a pattern that's only used at destructuring sites.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        ArrayD<f32>,
        ArrayD<f32>,
        ArrayD<f32>,
        ArrayD<f32>,
        Vec<usize>,
        Vec<usize>,
    ) {
        (
            self.lower_a,
            self.lower_b,
            self.upper_a,
            self.upper_b,
            self.input_shape,
            self.output_shape,
        )
    }

    // --- Unchecked factory ---

    /// Unchecked construction for performance-critical inner loops where
    /// inputs are already validated upstream.
    ///
    /// In debug builds, asserts that no array contains NaN (matching
    /// `validate_no_nan()` contract). In release builds, skips all validation.
    ///
    /// # Safety (logical)
    /// Caller must ensure:
    /// - Shapes are consistent (see `validate_internal_shapes()`)
    /// - No NaN in any array
    ///
    /// Tracked: grep for `from_parts_unchecked` to audit usage sites.
    /// Reference: designs/2026-02-26-batched-linear-bounds-encapsulation.md §Step 1
    pub(crate) fn from_parts_unchecked(
        lower_a: ArrayD<f32>,
        lower_b: ArrayD<f32>,
        upper_a: ArrayD<f32>,
        upper_b: ArrayD<f32>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    ) -> Self {
        debug_assert!(
            !lower_a.iter().any(|v| v.is_nan()),
            "from_parts_unchecked: lower_a contains NaN"
        );
        debug_assert!(
            !upper_a.iter().any(|v| v.is_nan()),
            "from_parts_unchecked: upper_a contains NaN"
        );
        debug_assert!(
            !lower_b.iter().any(|v| v.is_nan()),
            "from_parts_unchecked: lower_b contains NaN"
        );
        debug_assert!(
            !upper_b.iter().any(|v| v.is_nan()),
            "from_parts_unchecked: upper_b contains NaN"
        );
        Self {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            input_shape,
            output_shape,
            lower_a_err: None,
            upper_a_err: None,
        }
    }

    /// Create identity linear bounds (output = input) for given shape.
    ///
    /// For shape [..., dim], creates bounds where A = I (identity) and b = 0.
    /// This represents y >= x and y <= x, i.e., y = x.
    pub fn identity(shape: &[usize]) -> Result<Self> {
        if shape.is_empty()
            || checked_shape_product(shape).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "BatchedLinearBounds::identity: shape {shape:?} overflows usize",
                ))
            })? == 0
        {
            // KEEP unchecked: locally-allocated identity / zero arrays are NaN-free
            // and the degenerate [1, 1] / [1] shapes match by construction.
            return Ok(Self::from_parts_unchecked(
                ArrayD::ones(IxDyn(&[1, 1])),
                ArrayD::zeros(IxDyn(&[1])),
                ArrayD::ones(IxDyn(&[1, 1])),
                ArrayD::zeros(IxDyn(&[1])),
                vec![1],
                vec![1],
            ));
        }

        let dim = shape[shape.len() - 1];
        let batch_dims: Vec<usize> = shape[..shape.len() - 1].to_vec();

        // Coefficient shape: [batch..., out_dim, in_dim]
        let mut a_shape = batch_dims.clone();
        a_shape.push(dim);
        a_shape.push(dim);

        // Bias shape: [batch..., out_dim]
        let mut b_shape = batch_dims.clone();
        b_shape.push(dim);

        // Create eye matrix [dim, dim]
        let eye = Array2::eye(dim);

        // Broadcast to [batch..., dim, dim]
        let total_batch = checked_shape_product(&batch_dims).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "BatchedLinearBounds identity: batch dims product overflows: {:?}",
                batch_dims
            ))
        })?;
        let total_batch = total_batch.max(1);

        // Stack identity matrices — build each array's backing vec separately
        // to avoid a .clone() that doubles peak allocation (#2220 F6).
        // For dim=4096 × 16 batch, this saves ~1 GB of transient allocation.
        let build_identity_vec = |cap: usize| -> Vec<f32> {
            let mut v = Vec::with_capacity(cap);
            for _ in 0..total_batch {
                v.extend(eye.iter());
            }
            v
        };
        let total_elems = total_batch * dim * dim;

        let lower_a = ArrayD::from_shape_vec(IxDyn(&a_shape), build_identity_vec(total_elems))
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "identity: failed to construct lower_a from shape {a_shape:?}: {e}"
                ))
            })?;
        let upper_a = ArrayD::from_shape_vec(IxDyn(&a_shape), build_identity_vec(total_elems))
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "identity: failed to construct upper_a from shape {a_shape:?}: {e}"
                ))
            })?;

        // KEEP unchecked: eye matrices and zero biases are allocated locally,
        // so only the validated shape assembly above can fail.
        Ok(Self::from_parts_unchecked(
            lower_a,
            ArrayD::zeros(IxDyn(&b_shape)),
            upper_a,
            ArrayD::zeros(IxDyn(&b_shape)),
            shape.to_vec(),
            shape.to_vec(),
        ))
    }

    /// Create identity bounds for attention-shaped output.
    ///
    /// For attention output with shape [batch, heads, seq, seq], creates bounds with
    /// the last two dimensions flattened to enable McCormick relaxation for Q@K^T.
    ///
    /// The resulting bounds have shape [batch, heads, seq*seq, seq*seq] for the A matrix,
    /// which matches the flattened c_size = seq * seq expected by McCormick CROWN.
    ///
    /// Returns None if the shape doesn't match attention pattern (4D with last two dims equal)
    /// or if the flattened size would be too large (> 16M elements per identity matrix).
    pub fn identity_for_attention(shape: &[usize]) -> Option<Self> {
        // Attention output shape: [batch, heads, seq, seq] (4D with last two dims equal)
        if shape.len() != 4 {
            return None;
        }

        let batch = shape[0];
        let heads = shape[1];
        let seq_out = shape[2];
        let seq_in = shape[3];

        // Must be square attention output
        if seq_out != seq_in {
            return None;
        }

        let seq = seq_out;
        let flat_size = seq * seq;

        // Memory limit: this dense identity is O(seq^4) elements.
        // For seq > 64, flat_size > 4096, and the identity would exceed 16M elements.
        // This limit only affects the MatMul attention retry path (crown_batched.rs).
        //
        // For BilinearCrown nodes, BilinearRelaxation (#286 Approach A) handles
        // arbitrary seq without needing this dense identity — it stores O(batch*m*n*k)
        // per-batch coefficients and composes via broadcast einsum.
        if flat_size > 4096 {
            return None;
        }

        let batch_size = batch * heads;
        let total_elements = batch_size * flat_size * flat_size;

        // Total memory check: batch_size * flat_size^2 * 4 bytes * 2 (lower + upper)
        // Allow up to 256MB total for coefficient matrices
        let max_elements = 256 * 1024 * 1024 / 4 / 2; // ~32M elements per matrix
        if total_elements > max_elements {
            return None;
        }

        // Create identity matrix [flat_size, flat_size]
        let eye = Array2::<f32>::eye(flat_size);

        // Stack identity matrices for each batch position
        let mut lower_a_data = Vec::with_capacity(total_elements);
        for _ in 0..batch_size {
            lower_a_data.extend(eye.iter());
        }

        // A shape: [batch, heads, flat_size, flat_size]
        let a_shape = vec![batch, heads, flat_size, flat_size];
        // Bias shape: [batch, heads, flat_size]
        let b_shape = vec![batch, heads, flat_size];

        let lower_a = ArrayD::from_shape_vec(IxDyn(&a_shape), lower_a_data.clone()).ok()?;
        let upper_a = ArrayD::from_shape_vec(IxDyn(&a_shape), lower_a_data).ok()?;

        // KEEP unchecked: this helper builds eye matrices and zero biases from
        // validated dimensions only; reshaping cannot inject NaN.
        Some(Self::from_parts_unchecked(
            lower_a,
            ArrayD::zeros(IxDyn(&b_shape)),
            upper_a,
            ArrayD::zeros(IxDyn(&b_shape)),
            vec![batch, heads, flat_size], // Flattened shape
            vec![batch, heads, flat_size], // Flattened shape
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conservative_shape_2d() {
        let c = BatchedLinearBounds::conservative(vec![4], vec![3]);
        assert_eq!(c.input_shape(), &[4]);
        assert_eq!(c.output_shape(), &[3]);
        assert_eq!(c.lower_a().shape(), &[3, 4]);
        assert_eq!(c.lower_b().shape(), &[3]);
        assert!(c.lower_a().iter().all(|v| *v == 0.0));
        assert!(c.upper_a().iter().all(|v| *v == 0.0));
        assert!(c.lower_b().iter().all(|v| *v == f32::NEG_INFINITY));
        assert!(c.upper_b().iter().all(|v| *v == f32::INFINITY));
    }

    #[test]
    fn test_conservative_shape_batched() {
        let c = BatchedLinearBounds::conservative(vec![2, 4], vec![2, 3]);
        assert_eq!(c.lower_a().shape(), &[2, 3, 4]);
        assert_eq!(c.lower_b().shape(), &[2, 3]);
    }

    #[test]
    fn test_new_or_conservative_finite_passthrough() {
        let la = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
        let lb = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5; 2]).unwrap();
        let ua = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0; 6]).unwrap();
        let ub = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.6; 2]).unwrap();
        let result = BatchedLinearBounds::new_or_conservative(
            la.clone(),
            lb.clone(),
            ua,
            ub,
            vec![3],
            vec![2],
        )
        .unwrap();
        assert_eq!(result.lower_a(), &la);
        assert_eq!(result.lower_b(), &lb);
    }

    #[test]
    fn test_new_or_conservative_nan_falls_back() {
        let mut la = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
        la[[0, 0]] = f32::NAN;
        let lb = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5; 2]).unwrap();
        let ua = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0; 6]).unwrap();
        let ub = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.6; 2]).unwrap();
        let result =
            BatchedLinearBounds::new_or_conservative(la, lb, ua, ub, vec![3], vec![2]).unwrap();
        // NaN → conservative fallback
        assert!(result.lower_a().iter().all(|v| *v == 0.0));
        assert!(result.upper_a().iter().all(|v| *v == 0.0));
        assert!(result.lower_b().iter().all(|v| *v == f32::NEG_INFINITY));
        assert!(result.upper_b().iter().all(|v| *v == f32::INFINITY));
    }

    #[test]
    fn test_new_or_conservative_inf_passthrough() {
        // BatchedLinearBounds allows ±Inf (compose() produces it)
        let mut la = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
        la[[0, 0]] = f32::INFINITY;
        let lb = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5; 2]).unwrap();
        let ua = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0; 6]).unwrap();
        let ub = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.6; 2]).unwrap();
        let result =
            BatchedLinearBounds::new_or_conservative(la.clone(), lb, ua, ub, vec![3], vec![2])
                .unwrap();
        // ±Inf is allowed in BatchedLinearBounds, should pass through
        assert_eq!(result.lower_a(), &la);
    }

    #[test]
    fn malformed_coefficient_error_fails_closed() {
        let mut bounds = BatchedLinearBounds::new(
            ArrayD::zeros(IxDyn(&[1, 2])),
            ArrayD::zeros(IxDyn(&[1])),
            ArrayD::zeros(IxDyn(&[1, 2])),
            ArrayD::zeros(IxDyn(&[1])),
            vec![2],
            vec![1],
        )
        .unwrap();
        bounds.set_coeff_err(
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-1.0, 0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![f32::NAN, 0.0]).unwrap(),
        );
        assert_eq!(bounds.lower_a_err.as_ref().unwrap()[[0, 0]], f32::INFINITY);
        assert_eq!(bounds.upper_a_err.as_ref().unwrap()[[0, 0]], f32::INFINITY);

        bounds.set_coeff_err(ArrayD::zeros(IxDyn(&[2, 2])), ArrayD::zeros(IxDyn(&[2, 2])));
        assert!(bounds.lower_b().iter().all(|&v| v == f32::NEG_INFINITY));
        assert!(bounds.upper_b().iter().all(|&v| v == f32::INFINITY));

        let mut directly_malformed = BatchedLinearBounds::new(
            ArrayD::zeros(IxDyn(&[1, 2])),
            ArrayD::zeros(IxDyn(&[1])),
            ArrayD::zeros(IxDyn(&[1, 2])),
            ArrayD::zeros(IxDyn(&[1])),
            vec![2],
            vec![1],
        )
        .unwrap();
        directly_malformed.lower_a_err = Some(ArrayD::from_elem(IxDyn(&[1, 2]), -1.0));
        assert!(directly_malformed.validate_no_nan().is_err());
        directly_malformed.fold_coeff_err_into_bias(&[1.0, 1.0], &[1.0, 1.0]);
        assert!(directly_malformed
            .lower_b()
            .iter()
            .all(|&value| value == f32::NEG_INFINITY));
        assert!(directly_malformed
            .upper_b()
            .iter()
            .all(|&value| value == f32::INFINITY));

        // A carrier-layout mismatch must also degrade when the real result has
        // no fresh coefficient error of its own; the old discharge helper was a
        // no-op in precisely that case.
        let mut real = BatchedLinearBounds::new(
            ArrayD::zeros(IxDyn(&[1, 2])),
            ArrayD::zeros(IxDyn(&[1])),
            ArrayD::zeros(IxDyn(&[1, 2])),
            ArrayD::zeros(IxDyn(&[1])),
            vec![2],
            vec![1],
        )
        .unwrap();
        let carried = BatchedLinearBounds::new(
            ArrayD::zeros(IxDyn(&[1, 1])),
            ArrayD::zeros(IxDyn(&[1])),
            ArrayD::zeros(IxDyn(&[1, 1])),
            ArrayD::zeros(IxDyn(&[1])),
            vec![1],
            vec![1],
        )
        .unwrap();
        real.attach_err_from_carried(&carried);
        assert!(real
            .lower_b()
            .iter()
            .all(|&value| value == f32::NEG_INFINITY));
        assert!(real.upper_b().iter().all(|&value| value == f32::INFINITY));

        let mut strided_error = BatchedLinearBounds::new(
            ArrayD::zeros(IxDyn(&[2, 2])),
            ArrayD::zeros(IxDyn(&[2])),
            ArrayD::zeros(IxDyn(&[2, 2])),
            ArrayD::zeros(IxDyn(&[2])),
            vec![2],
            vec![2],
        )
        .unwrap();
        strided_error.lower_a_err =
            Some(ArrayD::from_elem(IxDyn(&[2, 2]), 1.0).permuted_axes(IxDyn(&[1, 0])));
        strided_error.upper_a_err = Some(ArrayD::zeros(IxDyn(&[2, 2])));
        assert!(strided_error
            .lower_a_err
            .as_ref()
            .unwrap()
            .as_slice()
            .is_none());
        strided_error.fold_coeff_err_into_bias(&[1.0, 1.0], &[1.0, 1.0]);
        assert!(strided_error
            .lower_b()
            .iter()
            .all(|&value| value == f32::NEG_INFINITY));
        assert!(strided_error
            .upper_b()
            .iter()
            .all(|&value| value == f32::INFINITY));
    }
}

/// Soundness and tightness tests for certified BLAS concretization.
///
/// The fast path attaches a complete binary64 BLAS error envelope and switches
/// cancellation-heavy rows to the shared self-checked DD reducer. These tests
/// require both routes to bracket exact-product references without making the
/// former absolute envelope vacuous.
///
/// These tests build f64 ground truth and assert (a) the BLAS bounds always
/// bracket it (soundness) and (b) the BLAS path matches the trusted f64-scalar
/// path to within the unavoidable f32-cast ULP (tightness), including the wide
/// n=4096 and heavy-cancellation cases that the previous f32-BLAS + envelope
/// path over-widened.
#[cfg(test)]
mod blas_accum_widening_soundness {
    use super::*;
    use ny_tensor::BoundedTensor;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// f64 ground-truth concretization of one output row over a shared input box.
    ///
    /// Products `la*x` of two f32 values are EXACT in f64; the f64 running sum
    /// has only sub-f64-ULP error (~1e-13 relative), so `(lo64, hi64)` is the
    /// true mathematical (min, max) of the linear bound functions over the box,
    /// up to error utterly negligible against f32 ULPs. A sound BLAS lower bound
    /// must be <= `lo64`; a sound upper bound must be >= `hi64`.
    fn f64_truth(la: &[f32], ua: &[f32], xl: &[f32], xu: &[f32], lb: f32, ub: f32) -> (f64, f64) {
        let mut lo = lb as f64;
        let mut hi = ub as f64;
        for j in 0..la.len() {
            let (laj, uaj) = (la[j] as f64, ua[j] as f64);
            let (xlj, xuj) = (xl[j] as f64, xu[j] as f64);
            // Lower of A_L @ x: pos coeff * x_l + neg coeff * x_u.
            lo += laj.max(0.0) * xlj + laj.min(0.0) * xuj;
            // Upper of A_U @ x: pos coeff * x_u + neg coeff * x_l.
            hi += uaj.max(0.0) * xuj + uaj.min(0.0) * xlj;
        }
        (lo, hi)
    }

    /// Run the f64-accumulate BLAS path for a single output row (m=1) over a
    /// shared input box of width `n`, returning the (lower, upper) concrete
    /// bounds after certified BLAS-error enclosure and directed publication.
    fn blas_row(la: &[f32], ua: &[f32], xl: &[f32], xu: &[f32], lb: f32, ub: f32) -> (f32, f32) {
        let n = la.len();
        let lower_a = ArrayD::from_shape_vec(IxDyn(&[1, n]), la.to_vec()).unwrap();
        let upper_a = ArrayD::from_shape_vec(IxDyn(&[1, n]), ua.to_vec()).unwrap();
        let lower_b = ArrayD::from_shape_vec(IxDyn(&[1]), vec![lb]).unwrap();
        let upper_b = ArrayD::from_shape_vec(IxDyn(&[1]), vec![ub]).unwrap();
        let in_lower = ArrayD::from_shape_vec(IxDyn(&[n]), xl.to_vec()).unwrap();
        let in_upper = ArrayD::from_shape_vec(IxDyn(&[n]), xu.to_vec()).unwrap();
        let (lo, hi) = BatchedLinearBounds::concretize_blas_posneg(
            &lower_a, &upper_a, &lower_b, &upper_b, &in_lower, &in_upper,
        )
        .unwrap();
        (lo[[0]], hi[[0]])
    }

    /// Run the TRUSTED f64-scalar fallback for the same single-row instance, so
    /// tests can assert the f64-BLAS path matches it (tightness equivalence).
    fn scalar_row(la: &[f32], ua: &[f32], xl: &[f32], xu: &[f32], lb: f32, ub: f32) -> (f32, f32) {
        let n = la.len();
        let lower_a = ArrayD::from_shape_vec(IxDyn(&[1, n]), la.to_vec()).unwrap();
        let upper_a = ArrayD::from_shape_vec(IxDyn(&[1, n]), ua.to_vec()).unwrap();
        let lower_b = ArrayD::from_shape_vec(IxDyn(&[1]), vec![lb]).unwrap();
        let upper_b = ArrayD::from_shape_vec(IxDyn(&[1]), vec![ub]).unwrap();
        let in_lower = ArrayD::from_shape_vec(IxDyn(&[n]), xl.to_vec()).unwrap();
        let in_upper = ArrayD::from_shape_vec(IxDyn(&[n]), xu.to_vec()).unwrap();
        let (lo, hi) = BatchedLinearBounds::concretize_scalar_posneg(
            &lower_a, &upper_a, &lower_b, &upper_b, &in_lower, &in_upper,
        )
        .unwrap();
        (lo[[0]], hi[[0]])
    }

    #[test]
    fn three_term_binary64_cancellation_is_tight_on_blas_and_scalar_seams() {
        let large = 2.0_f32.powi(30);
        let coefficients = [large, 1.0, -large];
        let point = [large, 1.0, large];

        for (name, (lower, upper)) in [
            (
                "blas",
                blas_row(&coefficients, &coefficients, &point, &point, 0.0, 0.0),
            ),
            (
                "scalar",
                scalar_row(&coefficients, &coefficients, &point, &point, 0.0, 0.0),
            ),
        ] {
            assert!(
                lower <= 1.0 && upper >= 1.0,
                "{name}: [{lower:e}, {upper:e}]"
            );
            assert!(
                lower > 0.99 && upper < 1.01,
                "{name}: [{lower:e}, {upper:e}]"
            );
        }
    }

    #[test]
    fn public_concretize_sound_is_tight_on_blas_and_scalar_dispatches() {
        let large = 2.0_f32.powi(30);
        let cancellation = [large, 1.0, -large];
        let point_values = [large, 1.0, large];
        let point = ArrayD::from_shape_vec(IxDyn(&[3]), point_values.to_vec()).unwrap();
        let input = BoundedTensor::new(point.clone(), point).unwrap();

        for (name, rows) in [
            ("blas", vec![cancellation.to_vec()]),
            (
                "scalar fallback",
                vec![cancellation.to_vec(), vec![f32::INFINITY, 0.0, 0.0]],
            ),
        ] {
            let m = rows.len();
            let coefficients =
                ArrayD::from_shape_vec(IxDyn(&[m, 3]), rows.into_iter().flatten().collect())
                    .unwrap();
            let biases = ArrayD::zeros(IxDyn(&[m]));
            let bounds = BatchedLinearBounds::new(
                coefficients.clone(),
                biases.clone(),
                coefficients,
                biases,
                vec![3],
                vec![m],
            )
            .unwrap();
            let result = bounds.concretize_sound(&input).unwrap();
            let lower = result.lower()[[0]];
            let upper = result.upper()[[0]];
            assert!(
                lower <= 1.0 && upper >= 1.0,
                "{name}: [{lower:e}, {upper:e}]"
            );
            assert!(
                lower > 0.99 && upper < 1.01,
                "{name}: [{lower:e}, {upper:e}]"
            );
        }
    }

    /// Assert both concretization paths are SOUND against the exact-product f64
    /// truth (gap to truth >= 0), and that the f64-BLAS bound is additionally
    /// TIGHT (gap <= `budget`, i.e. no absolute-envelope over-widening).
    /// `lower = true` for a lower bound (truth - bound), `false` for an upper
    /// bound (bound - truth).
    ///
    /// SOUNDNESS is required of BOTH paths, including under the heavy-cancellation
    /// regime below. They share exact binary32 products and a complete reduction
    /// error channel, so neither may f32-round a product before
    /// summing: round-to-nearest biases each term INWARD at the TERM magnitude,
    /// whereas the compensating widening is one ULP at the (under cancellation,
    /// far smaller) RESULT magnitude, which cannot cover it.
    ///
    /// `budget` is calibrated against the BLAS path's summation order and is
    /// asserted for it alone; the two paths' f64 dot residuals differ by sub-ULP
    /// reordering, which the neighbouring `close_tol` checks pin down directly.
    fn assert_tight(blas: f32, scal: f32, truth: f64, budget: f64, lower: bool) {
        let gap = |v: f32| {
            if lower {
                truth - v as f64
            } else {
                v as f64 - truth
            }
        };
        let dir = if lower { "lower" } else { "upper" };
        for (name, bound) in [("blas", blas), ("scalar", scal)] {
            let g = gap(bound);
            assert!(
                g >= 0.0,
                "UNSOUND ({dir}, {name}): {bound} on wrong side of truth={truth} (gap {g})",
            );
        }
        let blas_gap = gap(blas);
        assert!(
            blas_gap <= budget,
            "over-widened ({dir}): blas={blas} gap={blas_gap} budget={budget}",
        );
    }

    /// DENSE SAMPLING, WIDE n: with n=4096 and many same-magnitude mixed-sign
    /// terms (the classic f32-accumulation-drift regime), the f64-accumulate
    /// BLAS lower bound stays <= the f64 truth and the upper >= the f64 truth,
    /// AND the BLAS path is at least as tight as the trusted f64-scalar path,
    /// with the gap to truth only a few result-ULPs. This is the case the
    /// previous f32-BLAS + envelope path over-widened; f64 accumulation makes it
    /// tight (the old envelope was ~`gamma_{2n+2}*S` ≈ n*eps*S, far larger).
    #[test]
    fn wide_n_blas_matches_scalar_and_is_sound_dense_sampling() {
        const N: usize = 4096;
        const TRIALS: usize = 400;
        let mut rng = StdRng::seed_from_u64(0xB1A5_ACC0_5EED);

        for _ in 0..TRIALS {
            // Coefficients ~O(1) of mixed sign so pos/neg split is exercised;
            // inputs are tight boxes around values ~O(1). Many same-magnitude
            // terms => classic f32 accumulation drift across 4096 additions
            // (which f64 accumulation eliminates).
            let la: Vec<f32> = (0..N).map(|_| rng.random_range(-1.0f32..1.0)).collect();
            let ua: Vec<f32> = (0..N).map(|_| rng.random_range(-1.0f32..1.0)).collect();
            let centers: Vec<f32> = (0..N).map(|_| rng.random_range(-1.0f32..1.0)).collect();
            let radii: Vec<f32> = (0..N).map(|_| rng.random_range(0.0f32..0.05)).collect();
            let xl: Vec<f32> = centers.iter().zip(&radii).map(|(c, r)| c - r).collect();
            let xu: Vec<f32> = centers.iter().zip(&radii).map(|(c, r)| c + r).collect();
            let lb = rng.random_range(-1.0f32..1.0);
            let ub = rng.random_range(-1.0f32..1.0);

            let (truth_lo, truth_hi) = f64_truth(&la, &ua, &xl, &xu, lb, ub);
            let (blas_lo, blas_hi) = blas_row(&la, &ua, &xl, &xu, lb, ub);
            let (scal_lo, scal_hi) = scalar_row(&la, &ua, &xl, &xu, lb, ub);

            // SOUNDNESS: f64-BLAS bounds must bracket the f64 truth.
            assert!(
                (blas_lo as f64) <= truth_lo,
                "UNSOUND lower: blas_lo={blas_lo} > truth_lo={truth_lo}",
            );
            assert!(
                (blas_hi as f64) >= truth_hi,
                "UNSOUND upper: blas_hi={blas_hi} < truth_hi={truth_hi}",
            );

            // TIGHTNESS: at least as tight as the scalar path, gap to truth only a
            // few result-ULPs (here |result| <= ~n => budget ~ a handful of ULPs).
            let result_ulp = |v: f64| (v.abs() as f32).max(f32::MIN_POSITIVE) * f32::EPSILON;
            assert_tight(
                blas_lo,
                scal_lo,
                truth_lo,
                8.0 * result_ulp(truth_lo) as f64,
                true,
            );
            assert_tight(
                blas_hi,
                scal_hi,
                truth_hi,
                8.0 * result_ulp(truth_hi) as f64,
                false,
            );
        }
    }

    /// SMALL n: at n=3 the f64-BLAS path is sound AND matches the trusted scalar
    /// path within the f32-cast ULP, with the gap to truth no larger than the
    /// scalar path's own (i.e. no over-widening — the previous envelope path was
    /// looser here).
    #[test]
    fn small_n_blas_matches_scalar_and_is_sound() {
        const N: usize = 3;
        const TRIALS: usize = 5000;
        let mut rng = StdRng::seed_from_u64(0x5A11_5EED);

        for _ in 0..TRIALS {
            let la: Vec<f32> = (0..N).map(|_| rng.random_range(-3.0f32..3.0)).collect();
            let ua: Vec<f32> = (0..N).map(|_| rng.random_range(-3.0f32..3.0)).collect();
            let centers: Vec<f32> = (0..N).map(|_| rng.random_range(-3.0f32..3.0)).collect();
            let radii: Vec<f32> = (0..N).map(|_| rng.random_range(0.0f32..1.0)).collect();
            let xl: Vec<f32> = centers.iter().zip(&radii).map(|(c, r)| c - r).collect();
            let xu: Vec<f32> = centers.iter().zip(&radii).map(|(c, r)| c + r).collect();
            let lb = rng.random_range(-3.0f32..3.0);
            let ub = rng.random_range(-3.0f32..3.0);

            let (truth_lo, truth_hi) = f64_truth(&la, &ua, &xl, &xu, lb, ub);
            let (blas_lo, blas_hi) = blas_row(&la, &ua, &xl, &xu, lb, ub);
            let (scal_lo, scal_hi) = scalar_row(&la, &ua, &xl, &xu, lb, ub);

            // SOUNDNESS.
            assert!(
                (blas_lo as f64) <= truth_lo,
                "UNSOUND lower (small n): blas_lo={blas_lo} > truth_lo={truth_lo}",
            );
            assert!(
                (blas_hi as f64) >= truth_hi,
                "UNSOUND upper (small n): blas_hi={blas_hi} < truth_hi={truth_hi}",
            );

            // CLOSE TO scalar: both paths form every product exactly in f64, so
            // they can differ only by f64 summation order and the two independent
            // f32 casts. `term_round` — 0.5 f32-ULP per term magnitude — is a
            // deliberately generous stand-in for that reordering slack: a fixed
            // result-ULP count is the WRONG metric under cancellation, where a
            // single |term|~12 product dwarfs the result.
            let term_round: f64 = (0..N)
                .map(|j| {
                    let tl = (la[j] as f64) * (xl[j] as f64);
                    let tu = (ua[j] as f64) * (xu[j] as f64);
                    0.5 * (tl.abs().max(tu.abs()) as f32 * f32::EPSILON) as f64
                })
                .sum();
            let result_floor = |v: f32| (v.abs()).max(f32::MIN_POSITIVE) * f32::EPSILON;
            let close_tol =
                2.0 * term_round + 4.0 * result_floor(scal_lo).max(result_floor(scal_hi)) as f64;
            assert!(
                (blas_lo as f64 - scal_lo as f64).abs() <= close_tol,
                "small-n lower not close to scalar: blas={blas_lo} scalar={scal_lo} tol={close_tol}",
            );
            assert!(
                (blas_hi as f64 - scal_hi as f64).abs() <= close_tol,
                "small-n upper not close to scalar: blas={blas_hi} scalar={scal_hi} tol={close_tol}",
            );

            // TIGHTNESS vs TRUTH: gap to the exact-product f64 truth is at most a
            // couple of result ULPs (no absolute-envelope over-widening). The f64
            // dot residual at n=3 is sub-f64-ULP and folds into the cast.
            let result_ulp = |v: f64| (v.abs() as f32).max(f32::MIN_POSITIVE) * f32::EPSILON;
            assert_tight(
                blas_lo,
                scal_lo,
                truth_lo,
                4.0 * result_ulp(truth_lo) as f64,
                true,
            );
            assert_tight(
                blas_hi,
                scal_hi,
                truth_hi,
                4.0 * result_ulp(truth_hi) as f64,
                false,
            );
        }
    }

    /// CANCELLATION: mixed-sign large coefficients (|coeff| ~1e4) with O(1)
    /// inputs make `|result| << sum_j |term_j|`. This is exactly where the
    /// previous f32-BLAS + ABSOLUTE-envelope path over-widened by `gamma*S`
    /// (huge S, tiny result) — vacuous bounds. The f64-accumulate path stays
    /// sound AND tight, with the gap to truth only a few result-ULPs plus the
    /// sub-f64-ULP dot residual — NOT the `gamma*S` envelope. This is also the
    /// regime that catches an f32 per-term product on EITHER path: the inward
    /// per-term bias sits at ~1e4 while the widening sits at the result, so a
    /// path that f32-rounds its products fails the soundness assert here.
    #[test]
    fn cancellation_blas_matches_scalar_and_is_tight_dense_sampling() {
        const N: usize = 64;
        const TRIALS: usize = 4000;
        let mut rng = StdRng::seed_from_u64(0xCA0_CE11_BEEF);

        for _ in 0..TRIALS {
            // Large-magnitude coefficients (well within CROWN_COEFF_MAX=1e10)
            // with mixed signs, paired with O(1) inputs => heavy cancellation:
            // the true result is small but term magnitudes are ~1e4.
            let scale: f32 = 1.0e4;
            let la: Vec<f32> = (0..N).map(|_| rng.random_range(-scale..scale)).collect();
            let ua: Vec<f32> = la.clone();
            let centers: Vec<f32> = (0..N).map(|_| rng.random_range(-1.0f32..1.0)).collect();
            let radii: Vec<f32> = (0..N).map(|_| rng.random_range(0.0f32..0.01)).collect();
            let xl: Vec<f32> = centers.iter().zip(&radii).map(|(c, r)| c - r).collect();
            let xu: Vec<f32> = centers.iter().zip(&radii).map(|(c, r)| c + r).collect();
            let lb = 0.0f32;
            let ub = 0.0f32;

            let (truth_lo, truth_hi) = f64_truth(&la, &ua, &xl, &xu, lb, ub);
            let (blas_lo, blas_hi) = blas_row(&la, &ua, &xl, &xu, lb, ub);
            let (scal_lo, scal_hi) = scalar_row(&la, &ua, &xl, &xu, lb, ub);

            // SOUNDNESS under cancellation.
            assert!(
                (blas_lo as f64) <= truth_lo,
                "UNSOUND lower (cancellation): blas_lo={blas_lo} > truth_lo={truth_lo}",
            );
            assert!(
                (blas_hi as f64) >= truth_hi,
                "UNSOUND upper (cancellation): blas_hi={blas_hi} < truth_hi={truth_hi}",
            );

            // NO OVER-WIDENING + AT LEAST AS TIGHT AS SCALAR. The BLAS gap to
            // truth is at most a few result-ULPs plus the sub-f64-ULP dot
            // residual (`~ n * S * 2^-53`, with S = sum|term| ~ n*scale*|x|).
            // This is DRAMATICALLY tighter than the rejected `gamma_{2n+2}*S`
            // envelope (~ n*2^-24 * S ≈ 4e-4 here), which `assert_tight`'s budget
            // would reject — proving the over-widening is gone.
            let result_ulp = |v: f64| (v.abs() as f32).max(f32::MIN_POSITIVE) * f32::EPSILON;
            let s = N as f64 * 2.0 * scale as f64 * 1.0; // over-estimate of sum|term|
            let dot_residual = (N as f64) * s * f64::EPSILON; // f64 accumulation error
            let lo_budget = 8.0 * result_ulp(truth_lo) as f64 + 4.0 * dot_residual;
            let hi_budget = 8.0 * result_ulp(truth_hi) as f64 + 4.0 * dot_residual;
            assert_tight(blas_lo, scal_lo, truth_lo, lo_budget, true);
            assert_tight(blas_hi, scal_hi, truth_hi, hi_budget, false);
        }
    }
}
