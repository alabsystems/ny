// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Concretization methods for `LinearBounds`.
//!
//! Concretization computes concrete numerical bounds from linear bounds
//! given input bounds. This is the final step of CROWN backward propagation.

use std::{borrow::Cow, mem::size_of, time::Instant};

use ndarray::{Array1, Array2, ArrayD};
use ny_core::{
    dd::{next_down_f64, next_up_f64},
    is_crown_coeff_safe, NyError, Result,
};
use ny_tensor::{
    next_down_f32, next_up_f32, repair_inverted_bounds_nd, BoundedTensor, InversionRepair,
};

use super::{
    certified_affine_sum_f32, certified_reduction::certified_affine_sum_f32_with_poll,
    patches::PatchesMaterializationDeadline, LinearBounds, OutwardDirection,
};

/// Binary32 sign test which does not let DAZ reinterpret a negative subnormal
/// coefficient as `-0.0`. Signed zero remains non-negative, matching the
/// historical `coefficient >= 0.0` branch.
#[inline]
fn coefficient_is_nonnegative(value: f32) -> bool {
    let bits = value.to_bits();
    bits >> 31 == 0 || bits & 0x7fff_ffff == 0
}

/// `max(|lower|, |upper|)` from binary32 representations, without executing a
/// DAZ-sensitive f32 absolute value or comparison.
#[inline]
fn max_abs_f32_bits(lower: f32, upper: f32) -> f32 {
    let lower_magnitude = lower.to_bits() & 0x7fff_ffff;
    let upper_magnitude = upper.to_bits() & 0x7fff_ffff;
    let lower_nan = lower_magnitude > f32::INFINITY.to_bits();
    let upper_nan = upper_magnitude > f32::INFINITY.to_bits();
    if lower_nan || upper_nan {
        f32::NAN
    } else {
        f32::from_bits(lower_magnitude.max(upper_magnitude))
    }
}

/// Defense-in-depth sanitizer for an error carrier. In particular, a negative
/// subnormal must not pass `value < 0.0` as signed zero under DAZ and become a
/// negative penalty (an inward move).
#[inline]
fn nonnegative_error_or_infinity(value: f32) -> f32 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    let exponent = magnitude >> 23;
    if exponent == 0xff || (bits >> 31 != 0 && magnitude != 0) {
        f32::INFINITY
    } else {
        value
    }
}

/// Directed binary64-to-binary32 lower publication which never emits a
/// subnormal endpoint that a later FTZ consumer could move inward.
#[inline]
fn publish_lower_zero_or_normal(value: f64) -> f32 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == 0.0 {
        return value as f32;
    }
    let min_normal = ny_core::f32_to_f64_exact(f32::MIN_POSITIVE);
    if value.abs() < min_normal {
        return if value.is_sign_negative() {
            -f32::MIN_POSITIVE
        } else {
            0.0
        };
    }
    let published = next_down_f32(value as f32);
    let magnitude = published.to_bits() & 0x7fff_ffff;
    if magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits() {
        if value.is_sign_negative() {
            -f32::MIN_POSITIVE
        } else {
            0.0
        }
    } else {
        published
    }
}

/// Directed binary64-to-binary32 upper publication; mirror of
/// [`publish_lower_zero_or_normal`].
#[inline]
fn publish_upper_zero_or_normal(value: f64) -> f32 {
    if value.is_nan() || value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if value == 0.0 {
        return value as f32;
    }
    let min_normal = ny_core::f32_to_f64_exact(f32::MIN_POSITIVE);
    if value.abs() < min_normal {
        return if value.is_sign_negative() {
            f32::from_bits(0x8000_0000)
        } else {
            f32::MIN_POSITIVE
        };
    }
    let published = next_up_f32(value as f32);
    let magnitude = published.to_bits() & 0x7fff_ffff;
    if magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits() {
        if value.is_sign_negative() {
            f32::from_bits(0x8000_0000)
        } else {
            f32::MIN_POSITIVE
        }
    } else {
        published
    }
}

/// Sound concrete bounds plus proof provenance that must survive defensive
/// interval repair.
///
/// `certified_finite_inversion` is true only when the certified, outward f64
/// endpoints produced before `BoundedTensor` construction contain `lower >
/// upper` with both endpoints finite. The public bounds remain repaired to
/// `[-inf, +inf]`; callers must use this typed flag rather than trying to infer
/// infeasibility from that conservative repair.
pub(crate) struct SoundConcretization {
    pub(crate) bounds: BoundedTensor,
    pub(crate) certified_finite_inversion: bool,
    /// Per-output pre-repair `lower - upper` gaps. A non-finite endpoint is
    /// represented by `None`. Keeping row provenance prevents a hard-max
    /// optimizer score from masking progress on a different output row.
    pub(crate) row_finite_gaps: Vec<Option<f64>>,
    /// Largest finite pre-repair `lower - upper` gap. This is a heuristic
    /// optimization score, not proof authority; only the boolean above can
    /// authorize infeasibility.
    pub(crate) max_finite_gap: Option<f64>,
}

/// One conservative total-live receipt for a `LinearBounds` box
/// concretization. Existing ndarray buffers are charged by logical payload;
/// every `Vec` allocated by this operation reconciles its actual capacity
/// before any element is touched. The immutable caller-owned input is part of
/// the enclosing graph-state headroom, while a non-contiguous flatten copy is
/// charged here.
struct LinearConcretizationAdmission {
    nominal_required_bytes: usize,
    capacity_overage_bytes: usize,
    budget_bytes: usize,
}

impl LinearConcretizationAdmission {
    fn new(
        bounds: &LinearBounds,
        input_bounds: &BoundedTensor,
        retained_base_bytes: usize,
        include_row_provenance: bool,
        site: &'static str,
    ) -> Result<Self> {
        let outputs = bounds.num_outputs();
        let input_scratch_bytes = [input_bounds.lower(), input_bounds.upper()]
            .into_iter()
            .filter(|array| array.as_slice().is_none())
            .fold(0usize, |bytes, array| {
                bytes.saturating_add(array.len().saturating_mul(size_of::<f32>()))
            });
        let endpoint_bytes = outputs
            .saturating_mul(2)
            .saturating_mul(size_of::<f64>())
            .saturating_add(outputs.saturating_mul(2).saturating_mul(size_of::<f32>()));
        let provenance_bytes = if include_row_provenance {
            outputs.saturating_mul(size_of::<Option<f64>>())
        } else {
            0
        };
        let nominal_required_bytes = retained_base_bytes
            .saturating_add(bounds.memory_bytes())
            .saturating_add(input_scratch_bytes)
            .saturating_add(endpoint_bytes)
            .saturating_add(provenance_bytes);
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        if nominal_required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: nominal_required_bytes,
                budget_bytes,
                site,
            });
        }
        Ok(Self {
            nominal_required_bytes,
            capacity_overage_bytes: 0,
            budget_bytes,
        })
    }

    fn allocation_error(&self, site: &'static str) -> NyError {
        NyError::CpuMemoryExceeded {
            required_bytes: self
                .nominal_required_bytes
                .saturating_add(self.capacity_overage_bytes),
            budget_bytes: self.budget_bytes,
            site,
        }
    }

    fn reconcile_capacity<T>(
        &mut self,
        requested_elements: usize,
        actual_capacity: usize,
        site: &'static str,
    ) -> Result<()> {
        let requested_bytes = requested_elements.saturating_mul(size_of::<T>());
        let actual_bytes = actual_capacity.saturating_mul(size_of::<T>());
        self.capacity_overage_bytes = self
            .capacity_overage_bytes
            .saturating_add(actual_bytes.saturating_sub(requested_bytes));
        if self
            .nominal_required_bytes
            .saturating_add(self.capacity_overage_bytes)
            > self.budget_bytes
        {
            return Err(self.allocation_error(site));
        }
        Ok(())
    }
}

fn try_filled_f64_array1(
    len: usize,
    fill: f64,
    deadline: &mut PatchesMaterializationDeadline,
    admission: &mut LinearConcretizationAdmission,
    site: &'static str,
) -> Result<Array1<f64>> {
    deadline.checkpoint(site)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| admission.allocation_error(site))?;
    admission.reconcile_capacity::<f64>(len, values.capacity(), site)?;
    deadline.checkpoint(site)?;
    let mut filled = 0usize;
    while filled < len {
        let end = filled
            .saturating_add(PatchesMaterializationDeadline::CHECK_STRIDE)
            .min(len);
        values.resize(end, fill);
        deadline.work(end - filled, site)?;
        filled = end;
    }
    deadline.checkpoint(site)?;
    Ok(Array1::from_vec(values))
}

fn try_flatten_f32<'a>(
    array: &'a ArrayD<f32>,
    deadline: &mut PatchesMaterializationDeadline,
    admission: &mut LinearConcretizationAdmission,
    site: &'static str,
) -> Result<Cow<'a, [f32]>> {
    deadline.checkpoint(site)?;
    if let Some(slice) = array.as_slice() {
        return Ok(Cow::Borrowed(slice));
    }
    let len = array.len();
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| admission.allocation_error(site))?;
    admission.reconcile_capacity::<f32>(len, values.capacity(), site)?;
    deadline.checkpoint(site)?;
    for &value in array {
        values.push(value);
        deadline.work(1, site)?;
    }
    deadline.checkpoint(site)?;
    Ok(Cow::Owned(values))
}

fn try_publish_f64_array(
    source: &Array1<f64>,
    publish: fn(f64) -> f32,
    deadline: &mut PatchesMaterializationDeadline,
    admission: &mut LinearConcretizationAdmission,
    site: &'static str,
) -> Result<ArrayD<f32>> {
    let len = source.len();
    deadline.checkpoint(site)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| admission.allocation_error(site))?;
    admission.reconcile_capacity::<f32>(len, values.capacity(), site)?;
    deadline.checkpoint(site)?;
    for &value in source {
        values.push(publish(value));
        deadline.work(1, site)?;
    }
    deadline.checkpoint(site)?;
    Ok(Array1::from_vec(values).into_dyn())
}

fn try_filled_gap_vec(
    len: usize,
    deadline: &mut PatchesMaterializationDeadline,
    admission: &mut LinearConcretizationAdmission,
    site: &'static str,
) -> Result<Vec<Option<f64>>> {
    deadline.checkpoint(site)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| admission.allocation_error(site))?;
    admission.reconcile_capacity::<Option<f64>>(len, values.capacity(), site)?;
    deadline.checkpoint(site)?;
    let mut filled = 0usize;
    while filled < len {
        let end = filled
            .saturating_add(PatchesMaterializationDeadline::CHECK_STRIDE)
            .min(len);
        values.resize(end, None);
        deadline.work(end - filled, site)?;
        filled = end;
    }
    deadline.checkpoint(site)?;
    Ok(values)
}

/// Certified per-row concretization core (directed f64→f32 rounding), shared by
/// the patches-native sparse concretize
/// (`PatchesLinearBounds::concretize_sound_sparse`).
///
/// Computes the concrete `[lower, upper]` for ONE output row from that row's
/// per-active-column data, applying the SAME arithmetic as one row of
/// [`LinearBounds::concretize_scalar_f64`] followed by one element of
/// [`LinearBounds::f64_to_bounded_tensor`]'s directed cast + repair:
///
/// - certified exact-binary32-product reduction with a self-checked DD fast
///   path and directed-per-add non-finite fallback,
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
/// add — and it cannot trip the overflow guard (`0` is a safe coefficient).
/// Hence the reduction sees the same nonzero terms as the dense row and the
/// result is identical to `to_dense().concretize_sound(input)` for that row.
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
            if !lower_degraded && !is_crown_coeff_safe(la[j]) {
                lower_row_overflow = true;
            }
            if !upper_degraded && !is_crown_coeff_safe(ua[j]) {
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
        let mut sum_l = certified_affine_sum_f32(
            lb,
            (0..n).map(|j| {
                let endpoint = if coefficient_is_nonnegative(la[j]) {
                    in_l[j]
                } else {
                    in_u[j]
                };
                (la[j], endpoint)
            }),
            OutwardDirection::Lower,
        );
        let mut sum_u = certified_affine_sum_f32(
            ub,
            (0..n).map(|j| {
                let endpoint = if coefficient_is_nonnegative(ua[j]) {
                    in_u[j]
                } else {
                    in_l[j]
                };
                (ua[j], endpoint)
            }),
            OutwardDirection::Upper,
        );
        let have_err = le.is_some() || ue.is_some();
        if have_err {
            if let Some(le) = le {
                let penalty = certified_affine_sum_f32(
                    0.0,
                    (0..n).map(|j| {
                        let mag = max_abs_f32_bits(in_l[j], in_u[j]);
                        (nonnegative_error_or_infinity(le[j]), mag)
                    }),
                    OutwardDirection::Upper,
                );
                sum_l = next_down_f64(sum_l - penalty);
            }
            if let Some(ue) = ue {
                let penalty = certified_affine_sum_f32(
                    0.0,
                    (0..n).map(|j| {
                        let mag = max_abs_f32_bits(in_l[j], in_u[j]);
                        (nonnegative_error_or_infinity(ue[j]), mag)
                    }),
                    OutwardDirection::Upper,
                );
                sum_u = next_up_f64(sum_u + penalty);
            }
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
    let mut l = publish_lower_zero_or_normal(lower_f64);
    let mut u = publish_upper_zero_or_normal(upper_f64);
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

    fn conservative_unbounded_with_deadline(
        num_outputs: usize,
        deadline: &mut PatchesMaterializationDeadline,
        admission: &mut LinearConcretizationAdmission,
    ) -> Result<BoundedTensor> {
        let lower_f64 = try_filled_f64_array1(
            num_outputs,
            f64::NEG_INFINITY,
            deadline,
            admission,
            "during conservative concretization lower allocation",
        )?;
        let upper_f64 = try_filled_f64_array1(
            num_outputs,
            f64::INFINITY,
            deadline,
            admission,
            "during conservative concretization upper allocation",
        )?;
        let lower = try_publish_f64_array(
            &lower_f64,
            publish_lower_zero_or_normal,
            deadline,
            admission,
            "during conservative concretization lower publication",
        )?;
        let upper = try_publish_f64_array(
            &upper_f64,
            publish_upper_zero_or_normal,
            deadline,
            admission,
            "during conservative concretization upper publication",
        )?;
        deadline.checkpoint("before conservative concretization wrapping")?;
        let bounded = BoundedTensor::new_allow_infinite_with_poll(lower, upper, || {
            deadline.checkpoint("during conservative concretization wrapping")
        })?;
        deadline.checkpoint("after conservative concretization wrapping")?;
        Ok(bounded)
    }

    fn conservative_sound_concretization_with_deadline(
        num_outputs: usize,
        deadline: &mut PatchesMaterializationDeadline,
        admission: &mut LinearConcretizationAdmission,
    ) -> Result<SoundConcretization> {
        let bounds = Self::conservative_unbounded_with_deadline(num_outputs, deadline, admission)?;
        let row_finite_gaps = try_filled_gap_vec(
            num_outputs,
            deadline,
            admission,
            "during conservative concretization provenance allocation",
        )?;
        deadline.checkpoint("after conservative concretization provenance")?;
        Ok(SoundConcretization {
            bounds,
            certified_finite_inversion: false,
            row_finite_gaps,
            max_finite_gap: None,
        })
    }

    fn validate_no_nan_with_deadline(
        &self,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<()> {
        let scan_coefficients = |values: &Array2<f32>,
                                 message: &'static str,
                                 deadline: &mut PatchesMaterializationDeadline|
         -> Result<()> {
            for &value in values {
                if !value.is_finite() {
                    return Err(NyError::NumericalInstability(message.into()));
                }
                deadline.work(1, "during LinearBounds coefficient validation")?;
            }
            Ok(())
        };
        scan_coefficients(
            &self.lower_a,
            "LinearBounds lower_a coefficients contain NaN or Inf",
            deadline,
        )?;
        scan_coefficients(
            &self.upper_a,
            "LinearBounds upper_a coefficients contain NaN or Inf",
            deadline,
        )?;
        for &value in &self.lower_b {
            if value.is_nan() {
                return Err(NyError::NumericalInstability(
                    "LinearBounds lower_b bias contains NaN".into(),
                ));
            }
            deadline.work(1, "during LinearBounds lower bias validation")?;
        }
        for &value in &self.upper_b {
            if value.is_nan() {
                return Err(NyError::NumericalInstability(
                    "LinearBounds upper_b bias contains NaN".into(),
                ));
            }
            deadline.work(1, "during LinearBounds upper bias validation")?;
        }
        for error in self.lower_a_err.iter().chain(self.upper_a_err.iter()) {
            for &value in error {
                let bits = value.to_bits();
                let magnitude = bits & 0x7fff_ffff;
                if magnitude > f32::INFINITY.to_bits() || (bits >> 31 != 0 && magnitude != 0) {
                    return Err(NyError::NumericalInstability(
                        "LinearBounds coefficient error must be non-negative and non-NaN".into(),
                    ));
                }
                deadline.work(1, "during LinearBounds coefficient-error validation")?;
            }
        }
        deadline.checkpoint("after LinearBounds numeric validation")?;
        Ok(())
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
    /// Uses a certified exact-binary32-product reduction. Cancellation-heavy
    /// rows use self-checked double-double accumulation; non-finite rows fall
    /// back to directing every binary64 addition outward.
    ///
    /// Handles ±Inf coefficients from `safe_add` accumulation (#3032):
    /// rows with Inf bias or CROWN_COEFF_MAX overflow are short-circuited
    /// to ±Inf, and any NaN from the dot product is replaced with ±Inf.
    fn concretize_f64_inner(
        &self,
        input_bounds: &BoundedTensor,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let mut deadline = PatchesMaterializationDeadline::new(None);
        let mut admission = LinearConcretizationAdmission::new(
            self,
            input_bounds,
            0,
            false,
            "LinearBounds concretization",
        )?;
        self.concretize_f64_inner_with_deadline(input_bounds, &mut deadline, &mut admission)
    }

    fn concretize_f64_inner_with_deadline(
        &self,
        input_bounds: &BoundedTensor,
        deadline: &mut PatchesMaterializationDeadline,
        admission: &mut LinearConcretizationAdmission,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        deadline.checkpoint("before LinearBounds f64 concretization")?;
        self.validate_internal_shapes()?;

        let in_l = try_flatten_f32(
            input_bounds.lower(),
            deadline,
            admission,
            "during LinearBounds lower-input flatten",
        )?;
        let in_u = try_flatten_f32(
            input_bounds.upper(),
            deadline,
            admission,
            "during LinearBounds upper-input flatten",
        )?;
        let m = self.lower_a.nrows();
        let n = self.lower_a.ncols();

        self.concretize_scalar_f64_with_deadline(&in_l, &in_u, m, n, deadline, admission)
    }

    /// Scalar concretize with per-element NaN/Inf/overflow handling.
    fn concretize_scalar_f64_with_deadline(
        &self,
        in_l: &[f32],
        in_u: &[f32],
        m: usize,
        n: usize,
        deadline: &mut PatchesMaterializationDeadline,
        admission: &mut LinearConcretizationAdmission,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let mut lower = try_filled_f64_array1(
            m,
            0.0,
            deadline,
            admission,
            "during LinearBounds lower f64 endpoint allocation",
        )?;
        let mut upper = try_filled_f64_array1(
            m,
            0.0,
            deadline,
            admission,
            "during LinearBounds upper f64 endpoint allocation",
        )?;
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
                    if !lower_degraded && !is_crown_coeff_safe(self.lower_a[[i, j]]) {
                        lower_row_overflow = true;
                    }
                    if !upper_degraded && !is_crown_coeff_safe(self.upper_a[[i, j]]) {
                        upper_row_overflow = true;
                    }
                    if (lower_degraded || lower_row_overflow)
                        && (upper_degraded || upper_row_overflow)
                    {
                        break;
                    }
                    deadline.work(1, "during LinearBounds coefficient guard")?;
                }
            }
            if lower_degraded || lower_row_overflow {
                lower[i] = f64::NEG_INFINITY;
            }
            if upper_degraded || upper_row_overflow {
                upper[i] = f64::INFINITY;
            }
            if (lower_degraded || lower_row_overflow) && (upper_degraded || upper_row_overflow) {
                deadline.work(1, "between LinearBounds concretization rows")?;
                continue;
            }

            let mut sum_l = certified_affine_sum_f32_with_poll(
                lb,
                (0..n).map(|j| {
                    let coefficient = self.lower_a[[i, j]];
                    let endpoint = if coefficient_is_nonnegative(coefficient) {
                        in_l[j]
                    } else {
                        in_u[j]
                    };
                    (coefficient, endpoint)
                }),
                OutwardDirection::Lower,
                |units| deadline.work(units, "during LinearBounds lower reduction"),
            )?;
            let mut sum_u = certified_affine_sum_f32_with_poll(
                ub,
                (0..n).map(|j| {
                    let coefficient = self.upper_a[[i, j]];
                    let endpoint = if coefficient_is_nonnegative(coefficient) {
                        in_u[j]
                    } else {
                        in_l[j]
                    };
                    (coefficient, endpoint)
                }),
                OutwardDirection::Upper,
                |units| deadline.work(units, "during LinearBounds upper reduction"),
            )?;
            // Apply the certified-error penalty: lower goes DOWN, upper goes UP.
            // A non-finite penalty (from a degraded err entry) drives the bound to
            // ±inf, which f64_to_bounded_tensor repairs to [-inf, +inf] (sound).
            if let Some(le) = lower_err {
                let penalty = certified_affine_sum_f32_with_poll(
                    0.0,
                    (0..n).map(|j| {
                        let mag = max_abs_f32_bits(in_l[j], in_u[j]);
                        (nonnegative_error_or_infinity(le[[i, j]]), mag)
                    }),
                    OutwardDirection::Upper,
                    |units| deadline.work(units, "during LinearBounds lower-error reduction"),
                )?;
                sum_l = next_down_f64(sum_l - penalty);
            }
            if let Some(ue) = upper_err {
                let penalty = certified_affine_sum_f32_with_poll(
                    0.0,
                    (0..n).map(|j| {
                        let mag = max_abs_f32_bits(in_l[j], in_u[j]);
                        (nonnegative_error_or_infinity(ue[[i, j]]), mag)
                    }),
                    OutwardDirection::Upper,
                    |units| deadline.work(units, "during LinearBounds upper-error reduction"),
                )?;
                sum_u = next_up_f64(sum_u + penalty);
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
            deadline.work(1, "between LinearBounds concretization rows")?;
        }
        deadline.checkpoint("after LinearBounds f64 concretization")?;
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

    fn f64_to_bounded_tensor_with_deadline(
        &self,
        lower_f64: Array1<f64>,
        upper_f64: Array1<f64>,
        cast_lower: fn(f64) -> f32,
        cast_upper: fn(f64) -> f32,
        deadline: &mut PatchesMaterializationDeadline,
        admission: &mut LinearConcretizationAdmission,
    ) -> Result<BoundedTensor> {
        deadline.checkpoint("before LinearBounds endpoint publication")?;
        let mut lower = try_publish_f64_array(
            &lower_f64,
            cast_lower,
            deadline,
            admission,
            "during LinearBounds lower endpoint publication",
        )?;
        let mut upper = try_publish_f64_array(
            &upper_f64,
            cast_upper,
            deadline,
            admission,
            "during LinearBounds upper endpoint publication",
        )?;
        if lower.shape() != upper.shape() {
            tracing::warn!(
                lower_shape = ?lower.shape(),
                upper_shape = ?upper.shape(),
                num_outputs = self.num_outputs(),
                "LinearBounds::concretize produced mismatched output shapes; returning conservative [-inf, +inf] fallback"
            );
            drop(lower);
            drop(upper);
            drop(lower_f64);
            drop(upper_f64);
            return Self::conservative_unbounded_with_deadline(
                self.num_outputs(),
                deadline,
                admission,
            );
        }

        let lower_slice = lower.as_slice_mut().ok_or_else(|| {
            NyError::InternalError(
                "LinearBounds lower publication unexpectedly became non-contiguous".into(),
            )
        })?;
        let upper_slice = upper.as_slice_mut().ok_or_else(|| {
            NyError::InternalError(
                "LinearBounds upper publication unexpectedly became non-contiguous".into(),
            )
        })?;
        let mut repaired = 0usize;
        for (lower_value, upper_value) in lower_slice.iter_mut().zip(upper_slice.iter_mut()) {
            if !lower_value.is_finite() || !upper_value.is_finite() || *lower_value > *upper_value {
                *lower_value = f32::NEG_INFINITY;
                *upper_value = f32::INFINITY;
                repaired = repaired.saturating_add(1);
            }
            deadline.work(1, "during LinearBounds endpoint repair")?;
        }
        if repaired > 0 {
            tracing::debug!(
                repaired,
                num_outputs = self.num_outputs(),
                "LinearBounds::concretize_sound repaired {repaired} non-finite/inverted elements to [-inf, +inf]"
            );
        }

        deadline.checkpoint("before LinearBounds bounded-tensor wrapping")?;
        let bounded = BoundedTensor::new_allow_infinite_with_poll(lower, upper, || {
            deadline.checkpoint("during LinearBounds bounded-tensor validation")
        })?;
        deadline.checkpoint("after LinearBounds bounded-tensor wrapping")?;
        Ok(bounded)
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
        if let Err(err) = self
            .validate_internal_shapes()
            .and_then(|()| self.validate_no_nan())
        {
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
        self.validate_no_nan()?;
        if input_bounds.len() != self.num_inputs() {
            return Err(NyError::shape_mismatch(
                vec![self.num_inputs()],
                input_bounds.shape().to_vec(),
            ));
        }
        // #2239: directed rounding on f64→f32 for soundness.
        Ok(self.concretize_sound(input_bounds))
    }

    /// Deadline-aware counterpart of [`Self::concretize_checked`].
    pub(crate) fn concretize_checked_with_deadline(
        &self,
        input_bounds: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        let mut deadline = PatchesMaterializationDeadline::new(deadline);
        deadline.checkpoint("before checked LinearBounds concretization")?;
        self.validate_internal_shapes()?;
        self.validate_no_nan_with_deadline(&mut deadline)?;
        if input_bounds.len() != self.num_inputs() {
            return Err(NyError::shape_mismatch(
                vec![self.num_inputs()],
                input_bounds.shape().to_vec(),
            ));
        }
        self.concretize_sound_with_deadline_state(input_bounds, &mut deadline)
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
        match self.concretize_sound_with_deadline(input_bounds, None) {
            Ok(bounds) => bounds,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "fallible no-deadline LinearBounds concretization refused; returning conservative [-inf, +inf] fallback"
                );
                Self::conservative_unbounded(self.num_outputs())
            }
        }
    }

    /// Fallible, cooperatively pollable form of [`Self::concretize_sound`].
    ///
    /// Every allocation, coefficient/error scan, certified reduction, endpoint
    /// publication, repair pass, and final [`BoundedTensor`] validation observes
    /// the same absolute deadline. `DeadlineExceeded` and allocation refusal are
    /// returned without publishing a partial result. Malformed linear relations
    /// retain the historical sound `[-inf, +inf]` fallback.
    pub(crate) fn concretize_sound_with_deadline(
        &self,
        input_bounds: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        self.concretize_sound_with_deadline_and_resident(input_bounds, deadline, 0)
    }

    /// Deadline-aware concretization while another request-owned logical
    /// payload remains live. `retained_base_bytes` excludes both this relation
    /// and all allocations made by concretization; those are charged here.
    pub(crate) fn concretize_sound_with_deadline_and_resident(
        &self,
        input_bounds: &BoundedTensor,
        deadline: Option<Instant>,
        retained_base_bytes: usize,
    ) -> Result<BoundedTensor> {
        let mut deadline = PatchesMaterializationDeadline::new(deadline);
        deadline.checkpoint("before LinearBounds concretization admission")?;
        let mut admission = LinearConcretizationAdmission::new(
            self,
            input_bounds,
            retained_base_bytes,
            false,
            "LinearBounds sound concretization",
        )?;
        self.concretize_sound_with_deadline_state_and_admission(
            input_bounds,
            &mut deadline,
            &mut admission,
        )
    }

    pub(crate) fn concretize_sound_with_deadline_state(
        &self,
        input_bounds: &BoundedTensor,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<BoundedTensor> {
        deadline.checkpoint("before LinearBounds concretization admission")?;
        let mut admission = LinearConcretizationAdmission::new(
            self,
            input_bounds,
            0,
            false,
            "LinearBounds sound concretization",
        )?;
        self.concretize_sound_with_deadline_state_and_admission(
            input_bounds,
            deadline,
            &mut admission,
        )
    }

    fn concretize_sound_with_deadline_state_and_admission(
        &self,
        input_bounds: &BoundedTensor,
        deadline: &mut PatchesMaterializationDeadline,
        admission: &mut LinearConcretizationAdmission,
    ) -> Result<BoundedTensor> {
        // Keep this ubiquitous CROWN/BaB path free of the additional per-row
        // provenance allocation and scan. Typed INVPROP provenance (including
        // its row-gap Vec) is collected only by the explicitly named method
        // below.
        deadline.checkpoint("before LinearBounds sound concretization")?;
        if let Err(err) = self.validate_internal_shapes() {
            tracing::warn!(
                error = %err,
                lower_a_shape = ?self.lower_a.shape(),
                upper_a_shape = ?self.upper_a.shape(),
                lower_b_len = self.lower_b.len(),
                upper_b_len = self.upper_b.len(),
                "LinearBounds::concretize_sound called with malformed LinearBounds; returning conservative [-inf, +inf] fallback"
            );
            return Self::conservative_unbounded_with_deadline(
                self.num_outputs(),
                deadline,
                admission,
            );
        }
        if let Err(error) = self.validate_no_nan_with_deadline(deadline) {
            if matches!(
                &error,
                NyError::DeadlineExceeded(_) | NyError::CpuMemoryExceeded { .. }
            ) {
                return Err(error);
            }
            tracing::warn!(
                error = %error,
                lower_a_shape = ?self.lower_a.shape(),
                upper_a_shape = ?self.upper_a.shape(),
                lower_b_len = self.lower_b.len(),
                upper_b_len = self.upper_b.len(),
                "LinearBounds::concretize_sound called with malformed LinearBounds; returning conservative [-inf, +inf] fallback"
            );
            return Self::conservative_unbounded_with_deadline(
                self.num_outputs(),
                deadline,
                admission,
            );
        }
        let input_numel = input_bounds.len();
        if input_numel != self.num_inputs() {
            tracing::warn!(
                expected = self.num_inputs(),
                got = input_numel,
                "LinearBounds::concretize_sound input dimension mismatch; returning conservative [-inf, +inf] fallback"
            );
            return Self::conservative_unbounded_with_deadline(
                self.num_outputs(),
                deadline,
                admission,
            );
        }
        // Shape already validated above; the f64 core re-checks as defense-in-depth.
        let (lower, upper) = match self.concretize_f64_inner_with_deadline(
            input_bounds,
            deadline,
            admission,
        ) {
            Ok(pair) => pair,
            Err(error) => {
                if matches!(
                    &error,
                    NyError::DeadlineExceeded(_) | NyError::CpuMemoryExceeded { .. }
                ) {
                    return Err(error);
                }
                tracing::warn!(error = %error, "concretize_f64_inner failed despite pre-validation");
                return Self::conservative_unbounded_with_deadline(
                    self.num_outputs(),
                    deadline,
                    admission,
                );
            }
        };
        self.f64_to_bounded_tensor_with_deadline(
            lower,
            upper,
            publish_lower_zero_or_normal,
            publish_upper_zero_or_normal,
            deadline,
            admission,
        )
    }

    /// Sound concretization that preserves a certified pre-repair inversion.
    ///
    /// This crate-private variant is reserved for algorithms that can turn an
    /// empty conditioned region into an infeasibility certificate. Validation
    /// failures, dimension mismatches, NaN/non-finite endpoints, and overflow
    /// fallbacks never set the proof flag.
    pub(crate) fn concretize_sound_with_infeasibility(
        &self,
        input_bounds: &BoundedTensor,
    ) -> SoundConcretization {
        if let Err(err) = self
            .validate_internal_shapes()
            .and_then(|()| self.validate_no_nan())
        {
            tracing::warn!(
                error = %err,
                lower_a_shape = ?self.lower_a.shape(),
                upper_a_shape = ?self.upper_a.shape(),
                lower_b_len = self.lower_b.len(),
                upper_b_len = self.upper_b.len(),
                "LinearBounds::concretize_sound called with malformed LinearBounds; returning conservative [-inf, +inf] fallback"
            );
            return SoundConcretization {
                bounds: Self::conservative_unbounded(self.num_outputs()),
                certified_finite_inversion: false,
                row_finite_gaps: vec![None; self.num_outputs()],
                max_finite_gap: None,
            };
        }
        let input_numel = input_bounds.len();
        if input_numel != self.num_inputs() {
            tracing::warn!(
                expected = self.num_inputs(),
                got = input_numel,
                "LinearBounds::concretize_sound input dimension mismatch; returning conservative [-inf, +inf] fallback"
            );
            return SoundConcretization {
                bounds: Self::conservative_unbounded(self.num_outputs()),
                certified_finite_inversion: false,
                row_finite_gaps: vec![None; self.num_outputs()],
                max_finite_gap: None,
            };
        }
        // Shape already validated above; concretize_f64_inner re-checks as defense-in-depth.
        let (lower, upper) = match self.concretize_f64_inner(input_bounds) {
            Ok(pair) => pair,
            Err(err) => {
                tracing::warn!(error = %err, "concretize_f64_inner failed despite pre-validation");
                return SoundConcretization {
                    bounds: Self::conservative_unbounded(self.num_outputs()),
                    certified_finite_inversion: false,
                    row_finite_gaps: vec![None; self.num_outputs()],
                    max_finite_gap: None,
                };
            }
        };
        let certified_finite_inversion = lower
            .iter()
            .zip(upper.iter())
            .any(|(&lower, &upper)| lower.is_finite() && upper.is_finite() && lower > upper);
        let row_finite_gaps: Vec<Option<f64>> = lower
            .iter()
            .zip(upper.iter())
            .map(|(&lower, &upper)| {
                (lower.is_finite() && upper.is_finite())
                    .then_some(lower - upper)
                    .filter(|gap| gap.is_finite())
            })
            .collect();
        let max_finite_gap = row_finite_gaps
            .iter()
            .filter_map(|gap| *gap)
            .reduce(f64::max);
        let bounds = self.f64_to_bounded_tensor(
            lower,
            upper,
            publish_lower_zero_or_normal,
            publish_upper_zero_or_normal,
        );
        SoundConcretization {
            bounds,
            certified_finite_inversion,
            row_finite_gaps,
            max_finite_gap,
        }
    }

    /// Deadline-aware counterpart of
    /// [`Self::concretize_sound_with_infeasibility`].
    #[cfg(test)]
    pub(crate) fn concretize_sound_with_infeasibility_and_deadline(
        &self,
        input_bounds: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<SoundConcretization> {
        self.concretize_sound_with_infeasibility_deadline_and_resident(input_bounds, deadline, 0)
    }

    pub(crate) fn concretize_sound_with_infeasibility_deadline_and_resident(
        &self,
        input_bounds: &BoundedTensor,
        deadline: Option<Instant>,
        retained_base_bytes: usize,
    ) -> Result<SoundConcretization> {
        let mut deadline = PatchesMaterializationDeadline::new(deadline);
        deadline.checkpoint("before infeasibility concretization admission")?;
        let mut admission = LinearConcretizationAdmission::new(
            self,
            input_bounds,
            retained_base_bytes,
            true,
            "LinearBounds infeasibility concretization",
        )?;
        deadline.checkpoint("before infeasibility concretization")?;
        if let Err(error) = self.validate_internal_shapes() {
            tracing::warn!(
                error = %error,
                "malformed LinearBounds in infeasibility concretization; returning conservative bounds"
            );
            return Self::conservative_sound_concretization_with_deadline(
                self.num_outputs(),
                &mut deadline,
                &mut admission,
            );
        }
        if let Err(error) = self.validate_no_nan_with_deadline(&mut deadline) {
            if matches!(
                &error,
                NyError::DeadlineExceeded(_) | NyError::CpuMemoryExceeded { .. }
            ) {
                return Err(error);
            }
            tracing::warn!(
                error = %error,
                "malformed LinearBounds in infeasibility concretization; returning conservative bounds"
            );
            return Self::conservative_sound_concretization_with_deadline(
                self.num_outputs(),
                &mut deadline,
                &mut admission,
            );
        }
        if input_bounds.len() != self.num_inputs() {
            tracing::warn!(
                expected = self.num_inputs(),
                got = input_bounds.len(),
                "LinearBounds infeasibility concretization input mismatch; returning conservative bounds"
            );
            return Self::conservative_sound_concretization_with_deadline(
                self.num_outputs(),
                &mut deadline,
                &mut admission,
            );
        }
        let (lower, upper) = match self.concretize_f64_inner_with_deadline(
            input_bounds,
            &mut deadline,
            &mut admission,
        ) {
            Ok(pair) => pair,
            Err(error) => {
                if matches!(
                    &error,
                    NyError::DeadlineExceeded(_) | NyError::CpuMemoryExceeded { .. }
                ) {
                    return Err(error);
                }
                tracing::warn!(
                    error = %error,
                    "infeasibility f64 concretization failed; returning conservative bounds"
                );
                return Self::conservative_sound_concretization_with_deadline(
                    self.num_outputs(),
                    &mut deadline,
                    &mut admission,
                );
            }
        };

        let mut row_finite_gaps = try_filled_gap_vec(
            lower.len(),
            &mut deadline,
            &mut admission,
            "during infeasibility row-provenance allocation",
        )?;
        let mut certified_finite_inversion = false;
        let mut max_finite_gap: Option<f64> = None;
        for (index, (&lower_value, &upper_value)) in lower.iter().zip(upper.iter()).enumerate() {
            let gap = (lower_value.is_finite() && upper_value.is_finite())
                .then_some(lower_value - upper_value)
                .filter(|value| value.is_finite());
            if lower_value.is_finite() && upper_value.is_finite() && lower_value > upper_value {
                certified_finite_inversion = true;
            }
            if let Some(value) = gap {
                max_finite_gap = Some(max_finite_gap.map_or(value, |current| current.max(value)));
            }
            row_finite_gaps[index] = gap;
            deadline.work(1, "during infeasibility row-provenance scan")?;
        }
        deadline.checkpoint("after infeasibility row-provenance scan")?;
        let bounds = self.f64_to_bounded_tensor_with_deadline(
            lower,
            upper,
            publish_lower_zero_or_normal,
            publish_upper_zero_or_normal,
            &mut deadline,
            &mut admission,
        )?;
        deadline.checkpoint("after infeasibility concretization")?;
        Ok(SoundConcretization {
            bounds,
            certified_finite_inversion,
            row_finite_gaps,
            max_finite_gap,
        })
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
        self.validate_no_nan()?;
        if !rho.is_finite() || rho < 0.0 {
            return Err(NyError::InvalidSpec(format!(
                "rho must be finite and >= 0 (got {rho})"
            )));
        }
        if self.num_inputs() != x_hat.len() {
            return Err(NyError::shape_mismatch(
                vec![self.num_inputs()],
                vec![x_hat.len()],
            ));
        }
        if x_hat.iter().any(|value| !value.is_finite()) {
            return Err(NyError::InvalidSpec(
                "l2-ball center must contain only finite values".into(),
            ));
        }

        let m = self.num_outputs();
        let n = self.num_inputs();
        let rho_f64 = f64::from(rho);

        // A carried coefficient interval `A±E` contributes at most
        // `Σ_j E_j max_{x in ball}|x_j|`.  Since every coordinate in an ℓ2
        // ball satisfies `|x_j| <= |x_hat_j| + rho`, this per-coordinate box is
        // a sound (slightly conservative) discharge of the coefficient error.
        // Round each magnitude upward before the certified product reduction.
        let error_magnitude: Vec<f32> = x_hat
            .iter()
            .map(|&center| next_up_f32(next_up_f64(f64::from(center).abs() + rho_f64) as f32))
            .collect();

        let mut lower = Array1::<f32>::zeros(m);
        let mut upper = Array1::<f32>::zeros(m);

        for i in 0..m {
            let dot_l = certified_affine_sum_f32(
                self.lower_b[i],
                (0..n).map(|j| (self.lower_a[[i, j]], x_hat[j])),
                OutwardDirection::Lower,
            );
            let dot_u = certified_affine_sum_f32(
                self.upper_b[i],
                (0..n).map(|j| (self.upper_a[[i, j]], x_hat[j])),
                OutwardDirection::Upper,
            );
            let norm_sq_l = certified_affine_sum_f32(
                0.0,
                (0..n).map(|j| {
                    let a = self.lower_a[[i, j]];
                    (a, a)
                }),
                OutwardDirection::Upper,
            );
            let norm_sq_u = certified_affine_sum_f32(
                0.0,
                (0..n).map(|j| {
                    let a = self.upper_a[[i, j]];
                    (a, a)
                }),
                OutwardDirection::Upper,
            );
            let norm_l2_l = next_up_f64(norm_sq_l.max(0.0).sqrt());
            let norm_l2_u = next_up_f64(norm_sq_u.max(0.0).sqrt());
            let radius_l = next_up_f64(rho_f64 * norm_l2_l);
            let radius_u = next_up_f64(rho_f64 * norm_l2_u);
            let coefficient_penalty_l = self.lower_a_err.as_ref().map_or(0.0, |error| {
                certified_affine_sum_f32(
                    0.0,
                    (0..n).map(|j| (error[[i, j]], error_magnitude[j])),
                    OutwardDirection::Upper,
                )
            });
            let coefficient_penalty_u = self.upper_a_err.as_ref().map_or(0.0, |error| {
                certified_affine_sum_f32(
                    0.0,
                    (0..n).map(|j| (error[[i, j]], error_magnitude[j])),
                    OutwardDirection::Upper,
                )
            });
            // Apply directed rounding on f64→f32 cast for soundness.
            // Lower bound rounds toward -∞, upper bound rounds toward +∞.
            // Reference: alpha-beta-CROWN uses __double2float_rd/__double2float_ru
            // CUDA intrinsics for the same purpose (cuda_kernels.cu:8-22).
            let l_with_radius = next_down_f64(dot_l - radius_l);
            let u_with_radius = next_up_f64(dot_u + radius_u);
            let l_val = next_down_f32(next_down_f64(l_with_radius - coefficient_penalty_l) as f32);
            let u_val = next_up_f32(next_up_f64(u_with_radius + coefficient_penalty_u) as f32);
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

#[cfg(test)]
mod ftz_daz_tests {
    use ndarray::{Array1, Array2, ArrayD, IxDyn};

    use super::*;

    fn is_zero_or_normal(value: f32) -> bool {
        let magnitude = value.to_bits() & 0x7fff_ffff;
        magnitude == 0 || magnitude >= f32::MIN_POSITIVE.to_bits()
    }

    fn flush_f32(value: f32) -> f32 {
        let bits = value.to_bits();
        if bits & 0x7f80_0000 == 0 {
            f32::from_bits(bits & 0x8000_0000)
        } else {
            value
        }
    }

    #[test]
    fn row_and_matrix_concretize_keep_negative_subnormal_sign_and_normal_publication() {
        let tiny = f32::from_bits(1);
        let scale = 2.0_f32.powi(120);
        assert_eq!(flush_f32(-tiny).to_bits(), (-0.0f32).to_bits());
        let exact = ny_core::f32_to_f64_exact(tiny) * ny_core::f32_to_f64_exact(scale);

        let (row_lower, row_upper) = concretize_row_directed(
            0.0,
            0.0,
            &[-scale],
            &[scale],
            &[-tiny],
            &[-tiny],
            None,
            None,
        );
        assert!(ny_core::f32_to_f64_exact(row_lower) <= -exact);
        assert!(ny_core::f32_to_f64_exact(row_upper) >= exact);
        assert!(is_zero_or_normal(row_lower) && is_zero_or_normal(row_upper));

        let bounds = LinearBounds::new(
            Array2::from_elem((1, 1), -tiny),
            Array1::zeros(1),
            Array2::from_elem((1, 1), -tiny),
            Array1::zeros(1),
        )
        .unwrap();
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1]), -scale),
            ArrayD::from_elem(IxDyn(&[1]), scale),
        )
        .unwrap();
        let matrix = bounds.concretize_sound(&input);
        let lower = matrix.lower()[[0]];
        let upper = matrix.upper()[[0]];
        assert!(ny_core::f32_to_f64_exact(lower) <= -exact);
        assert!(ny_core::f32_to_f64_exact(upper) >= exact);
        assert!(is_zero_or_normal(lower) && is_zero_or_normal(upper));
    }

    #[test]
    fn row_and_matrix_error_magnitude_decode_subnormal_input_bits() {
        let tiny = f32::from_bits(1);
        let error = 2.0_f32.powi(120);
        let exact_penalty = ny_core::f32_to_f64_exact(tiny) * ny_core::f32_to_f64_exact(error);

        let (row_lower, row_upper) = concretize_row_directed(
            0.0,
            0.0,
            &[-tiny],
            &[tiny],
            &[0.0],
            &[0.0],
            Some(&[error]),
            Some(&[error]),
        );
        assert!(ny_core::f32_to_f64_exact(row_lower) <= -exact_penalty);
        assert!(ny_core::f32_to_f64_exact(row_upper) >= exact_penalty);

        let bounds = LinearBounds::from_prevalidated_parts_with_optional_err(
            Array2::zeros((1, 1)),
            Array1::zeros(1),
            Array2::zeros((1, 1)),
            Array1::zeros(1),
            Some(Array2::from_elem((1, 1), error)),
            Some(Array2::from_elem((1, 1), error)),
        )
        .unwrap();
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1]), -tiny),
            ArrayD::from_elem(IxDyn(&[1]), tiny),
        )
        .unwrap();
        let matrix = bounds.concretize_sound(&input);
        assert!(ny_core::f32_to_f64_exact(matrix.lower()[[0]]) <= -exact_penalty);
        assert!(ny_core::f32_to_f64_exact(matrix.upper()[[0]]) >= exact_penalty);

        let (tiny_lower, tiny_upper) =
            concretize_row_directed(0.0, 0.0, &[1.0], &[1.0], &[tiny], &[tiny], None, None);
        let exact_tiny = ny_core::f32_to_f64_exact(tiny);
        assert!(ny_core::f32_to_f64_exact(tiny_lower) <= exact_tiny);
        assert!(ny_core::f32_to_f64_exact(tiny_upper) >= exact_tiny);
        assert!(is_zero_or_normal(tiny_lower) && is_zero_or_normal(tiny_upper));

        let tiny_bounds = LinearBounds::new(
            Array2::from_elem((1, 1), tiny),
            Array1::zeros(1),
            Array2::from_elem((1, 1), tiny),
            Array1::zeros(1),
        )
        .unwrap();
        let unit = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1]), 1.0),
            ArrayD::from_elem(IxDyn(&[1]), 1.0),
        )
        .unwrap();
        let tiny_matrix = tiny_bounds.concretize_sound(&unit);
        let matrix_lower = tiny_matrix.lower()[[0]];
        let matrix_upper = tiny_matrix.upper()[[0]];
        assert!(ny_core::f32_to_f64_exact(matrix_lower) <= exact_tiny);
        assert!(ny_core::f32_to_f64_exact(matrix_upper) >= exact_tiny);
        assert!(is_zero_or_normal(matrix_lower) && is_zero_or_normal(matrix_upper));
    }

    #[test]
    fn deadline_concretization_is_bitwise_legacy_when_live() {
        let bounds = LinearBounds::from_prevalidated_parts_with_optional_err(
            Array2::from_shape_vec((2, 3), vec![1.0, -2.0, 0.5, -0.25, 4.0, -8.0]).unwrap(),
            Array1::from_vec(vec![0.125, -0.5]),
            Array2::from_shape_vec((2, 3), vec![2.0, -1.0, 0.25, -0.5, 2.0, -4.0]).unwrap(),
            Array1::from_vec(vec![0.25, 0.75]),
            Some(Array2::from_elem((2, 3), f32::EPSILON)),
            Some(Array2::from_elem((2, 3), 2.0 * f32::EPSILON)),
        )
        .unwrap();
        let input = BoundedTensor::new(
            Array1::from_vec(vec![-1.0, -0.5, 0.25]).into_dyn(),
            Array1::from_vec(vec![0.5, 1.0, 2.0]).into_dyn(),
        )
        .unwrap();
        let legacy = bounds.concretize_sound(&input);
        let live = bounds
            .concretize_sound_with_deadline(
                &input,
                Some(Instant::now() + std::time::Duration::from_mins(1)),
            )
            .unwrap();
        assert_eq!(legacy.lower(), live.lower());
        assert_eq!(legacy.upper(), live.upper());

        let legacy_infeasibility = bounds.concretize_sound_with_infeasibility(&input);
        let live_infeasibility = bounds
            .concretize_sound_with_infeasibility_and_deadline(
                &input,
                Some(Instant::now() + std::time::Duration::from_mins(1)),
            )
            .unwrap();
        assert_eq!(
            legacy_infeasibility.bounds.lower(),
            live_infeasibility.bounds.lower()
        );
        assert_eq!(
            legacy_infeasibility.bounds.upper(),
            live_infeasibility.bounds.upper()
        );
        assert_eq!(
            legacy_infeasibility.certified_finite_inversion,
            live_infeasibility.certified_finite_inversion
        );
        assert_eq!(
            legacy_infeasibility.row_finite_gaps,
            live_infeasibility.row_finite_gaps
        );
        assert_eq!(
            legacy_infeasibility.max_finite_gap,
            live_infeasibility.max_finite_gap
        );

        let checked = bounds
            .concretize_checked_with_deadline(
                &input,
                Some(Instant::now() + std::time::Duration::from_mins(1)),
            )
            .unwrap();
        assert_eq!(legacy.lower(), checked.lower());
        assert_eq!(legacy.upper(), checked.upper());
    }

    #[test]
    fn deadline_concretization_refuses_before_and_during_private_work() {
        let bounds = LinearBounds::new(
            Array2::from_elem((2, 3), 1.0),
            Array1::zeros(2),
            Array2::from_elem((2, 3), 1.0),
            Array1::zeros(2),
        )
        .unwrap();
        let input = BoundedTensor::new(
            Array1::from_elem(3, -1.0).into_dyn(),
            Array1::from_elem(3, 1.0).into_dyn(),
        )
        .unwrap();

        assert!(matches!(
            bounds.concretize_sound_with_deadline(&input, Some(Instant::now())),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert!(matches!(
            bounds.concretize_checked_with_deadline(&input, Some(Instant::now())),
            Err(NyError::DeadlineExceeded(_))
        ));
        assert!(matches!(
            bounds.concretize_sound_with_infeasibility_and_deadline(&input, Some(Instant::now())),
            Err(NyError::DeadlineExceeded(_))
        ));

        for stage in [
            "during LinearBounds lower reduction",
            "during LinearBounds bounded-tensor validation",
        ] {
            let mut deadline = PatchesMaterializationDeadline::forced_at(stage);
            assert!(matches!(
                bounds.concretize_sound_with_deadline_state(&input, &mut deadline),
                Err(NyError::DeadlineExceeded(_))
            ));
            assert!(bounds.lower_a.iter().all(|&value| value == 1.0));
            assert!(bounds.upper_a.iter().all(|&value| value == 1.0));
            assert!(bounds.lower_b.iter().all(|&value| value == 0.0));
            assert!(bounds.upper_b.iter().all(|&value| value == 0.0));
        }
    }

    #[test]
    fn deadline_concretization_receipts_include_retained_payload() {
        let bounds = LinearBounds::new(
            Array2::from_elem((1, 2), 1.0),
            Array1::zeros(1),
            Array2::from_elem((1, 2), 1.0),
            Array1::zeros(1),
        )
        .unwrap();
        let input = BoundedTensor::new(
            Array1::from_elem(2, -1.0).into_dyn(),
            Array1::from_elem(2, 1.0).into_dyn(),
        )
        .unwrap();

        crate::tests::with_crown_dense_budget_mb("1", || {
            bounds
                .concretize_sound_with_deadline_and_resident(&input, None, 0)
                .unwrap();
            let error = bounds
                .concretize_sound_with_deadline_and_resident(&input, None, 1_048_576)
                .unwrap_err();
            assert!(matches!(
                error,
                NyError::CpuMemoryExceeded {
                    budget_bytes: 1_048_576,
                    ..
                }
            ));
        });
    }
}
