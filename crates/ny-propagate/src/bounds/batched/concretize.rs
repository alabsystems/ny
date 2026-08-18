// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Concretization of batched linear bounds given input intervals.
//!
//! Extracted from `mod.rs` as part of #4212.
//!
//! # Concretization via positive/negative coefficient split (#2220 Packet B)
//!
//! For separate lower/upper bound functions f_L(x) = A_L @ x + b_L and
//! f_U(x) = A_U @ x + b_U, concretization computes:
//!
//!   min_{x ∈ [x_l, x_u]} f_L(x) = A_L_pos @ x_l + A_L_neg @ x_u + b_L
//!   max_{x ∈ [x_l, x_u]} f_U(x) = A_U_pos @ x_u + A_U_neg @ x_l + b_U
//!
//! This is tighter than full interval matvec (which treats [A_L, A_U] as an
//! interval on a single coefficient) and BLAS-acceleratable via ndarray dot.
//!
//! Reference: alpha-beta-CROWN `bound_general.py:1140-1160` uses the same
//! pos/neg split: `lA.clamp(min=0) * x_L + lA.clamp(max=0) * x_U`.

use super::BatchedLinearBounds;
use crate::bounds::{
    certified_reduction::certified_affine_sum_f32_with_poll,
    patches::PatchesMaterializationDeadline, OutwardDirection,
};
use ndarray::{ArrayD, IxDyn};
use ny_core::{
    checked_shape_product,
    dd::{next_down_f64, next_up_f64},
    f32_to_f64_exact, NyError, Result, CROWN_COEFF_MAX,
};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};
use std::{borrow::Cow, mem::size_of, time::Instant};

/// Complete shape/index plan for the finite-authority scalar concretization.
/// It contains no borrowed array views and allocates no data buffer.
struct FiniteConcretizePlan {
    coefficient_batches: usize,
    output_batches: usize,
    input_batches: usize,
    m: usize,
    n: usize,
    output_elements: usize,
    output_shape_len: usize,
    broadcast_coefficients: bool,
    output_uses_input_batch_shape: bool,
}

/// One receipt for every operation-owned allocation that can coexist during a
/// finite batched concretization.
///
/// The resident [`BatchedLinearBounds`] logical payload is charged in full,
/// including optional coefficient-error arrays. The immutable caller-owned
/// input box is already part of the enclosing graph request and remains under
/// the dense budget's process-envelope headroom; any copy needed to flatten a
/// non-standard input layout is charged here. Existing ndarray backing-vector
/// slack and small ndarray dimension metadata are not observable through the
/// public ndarray API. As at the Patches materialization boundary, the global
/// one-eighth process-envelope clamp reserves the remaining headroom for that
/// allocator metadata/slack and the enclosing graph.
struct FiniteConcretizeAdmission {
    nominal_required_bytes: usize,
    capacity_overage_bytes: usize,
    budget_bytes: usize,
}

impl FiniteConcretizeAdmission {
    fn check(nominal_required_bytes: usize, budget_bytes: usize) -> Result<Self> {
        if nominal_required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: nominal_required_bytes,
                budget_bytes,
                site: "batched finite concretization",
            });
        }
        Ok(Self {
            nominal_required_bytes,
            capacity_overage_bytes: 0,
            budget_bytes,
        })
    }

    #[inline]
    fn required_bytes(&self) -> usize {
        self.nominal_required_bytes
            .saturating_add(self.capacity_overage_bytes)
    }

    fn allocation_error(&self, site: &'static str) -> NyError {
        NyError::CpuMemoryExceeded {
            required_bytes: self.required_bytes(),
            budget_bytes: self.budget_bytes,
            site,
        }
    }

    fn reconcile_vec_capacity<T>(
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
        if self.required_bytes() > self.budget_bytes {
            return Err(self.allocation_error(site));
        }
        Ok(())
    }
}

#[inline]
fn coefficient_is_nonnegative_bits(value: f32) -> bool {
    let bits = value.to_bits();
    bits >> 31 == 0 || bits & 0x7fff_ffff == 0
}

/// Bitwise form of `is_crown_coeff_safe`, so DAZ cannot turn a nonzero
/// subnormal coefficient into signed zero during the guard scan.
#[inline]
fn crown_coefficient_is_safe_bits(value: f32) -> bool {
    let magnitude = value.to_bits() & 0x7fff_ffff;
    magnitude < CROWN_COEFF_MAX.to_bits()
}

#[inline]
fn max_abs_f32_bits(lower: f32, upper: f32) -> f32 {
    let lower_magnitude = lower.to_bits() & 0x7fff_ffff;
    let upper_magnitude = upper.to_bits() & 0x7fff_ffff;
    if lower_magnitude > f32::INFINITY.to_bits() || upper_magnitude > f32::INFINITY.to_bits() {
        f32::NAN
    } else {
        f32::from_bits(lower_magnitude.max(upper_magnitude))
    }
}

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

/// Directed binary64-to-binary32 publication that cannot leave a subnormal
/// endpoint for an FTZ consumer to move inward.
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
    let min_normal = f32_to_f64_exact(f32::MIN_POSITIVE);
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
    let min_normal = f32_to_f64_exact(f32::MIN_POSITIVE);
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

/// The legacy batched `concretize_sound` adds one final ULP after its directed
/// f64 publication. Preserve that ordinary-normal result without publishing a
/// DAZ/FTZ-sensitive subnormal. An exact signed zero remains exact.
#[inline]
fn widen_lower_zero_or_normal(value: f32) -> f32 {
    if value.to_bits() & 0x7fff_ffff == 0 || !value.is_finite() {
        return value;
    }
    let widened = next_down_f32(value);
    let magnitude = widened.to_bits() & 0x7fff_ffff;
    if magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits() {
        if value.is_sign_negative() {
            -f32::MIN_POSITIVE
        } else {
            0.0
        }
    } else {
        widened
    }
}

#[inline]
fn widen_upper_zero_or_normal(value: f32) -> f32 {
    if value.to_bits() & 0x7fff_ffff == 0 || !value.is_finite() {
        return value;
    }
    let widened = next_up_f32(value);
    let magnitude = widened.to_bits() & 0x7fff_ffff;
    if magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits() {
        if value.is_sign_negative() {
            f32::from_bits(0x8000_0000)
        } else {
            f32::MIN_POSITIVE
        }
    } else {
        widened
    }
}

fn shape_product_with_deadline(
    shape: &[usize],
    deadline: &mut PatchesMaterializationDeadline,
    stage: &'static str,
) -> Result<usize> {
    let mut product = 1usize;
    for &dimension in shape {
        product = product.checked_mul(dimension).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "BatchedLinearBounds shape product overflows: {shape:?}"
            ))
        })?;
        deadline.work(1, stage)?;
    }
    deadline.checkpoint(stage)?;
    Ok(product)
}

fn shapes_equal_with_deadline(
    expected: &[usize],
    actual: &[usize],
    deadline: &mut PatchesMaterializationDeadline,
    stage: &'static str,
) -> Result<bool> {
    if expected.len() != actual.len() {
        deadline.checkpoint(stage)?;
        return Ok(false);
    }
    for (&expected_dimension, &actual_dimension) in expected.iter().zip(actual) {
        if expected_dimension != actual_dimension {
            deadline.checkpoint(stage)?;
            return Ok(false);
        }
        deadline.work(1, stage)?;
    }
    deadline.checkpoint(stage)?;
    Ok(true)
}

fn is_vector_like_with_deadline(
    shape: &[usize],
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<bool> {
    let mut nonunit_dimensions = 0usize;
    for &dimension in shape {
        if dimension > 1 {
            nonunit_dimensions = nonunit_dimensions.saturating_add(1);
        }
        deadline.work(1, "during batched input vector-shape validation")?;
    }
    deadline.checkpoint("after batched input vector-shape validation")?;
    Ok(nonunit_dimensions <= 1)
}

fn try_flatten_array<'a>(
    array: &'a ArrayD<f32>,
    admission: &mut FiniteConcretizeAdmission,
    deadline: &mut PatchesMaterializationDeadline,
    site: &'static str,
) -> Result<Cow<'a, [f32]>> {
    deadline.checkpoint(site)?;
    if let Some(slice) = array.as_slice() {
        return Ok(Cow::Borrowed(slice));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(array.len())
        .map_err(|_| admission.allocation_error(site))?;
    admission.reconcile_vec_capacity::<f32>(array.len(), values.capacity(), site)?;
    deadline.checkpoint(site)?;
    for &value in array {
        values.push(value);
        deadline.work(1, site)?;
    }
    deadline.checkpoint(site)?;
    Ok(Cow::Owned(values))
}

fn try_reserved_vec<T>(
    len: usize,
    admission: &mut FiniteConcretizeAdmission,
    deadline: &mut PatchesMaterializationDeadline,
    site: &'static str,
) -> Result<Vec<T>> {
    deadline.checkpoint(site)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| admission.allocation_error(site))?;
    admission.reconcile_vec_capacity::<T>(len, values.capacity(), site)?;
    deadline.checkpoint(site)?;
    Ok(values)
}

fn scan_no_nan_with_deadline(
    array: &ArrayD<f32>,
    label: &'static str,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<()> {
    for &value in array {
        if value.is_nan() {
            return Err(NyError::NumericalInstability(format!(
                "BatchedLinearBounds {label} contains NaN"
            )));
        }
        deadline.work(1, "during batched coefficient/bias numeric validation")?;
    }
    deadline.checkpoint("after batched coefficient/bias numeric validation")
}

fn scan_error_with_deadline(
    array: &ArrayD<f32>,
    deadline: &mut PatchesMaterializationDeadline,
) -> Result<()> {
    for &value in array {
        let bits = value.to_bits();
        let magnitude = bits & 0x7fff_ffff;
        if magnitude > f32::INFINITY.to_bits() || (bits >> 31 != 0 && magnitude != 0) {
            return Err(NyError::NumericalInstability(
                "BatchedLinearBounds coefficient error must be non-negative and non-NaN".into(),
            ));
        }
        deadline.work(1, "during batched coefficient-error numeric validation")?;
    }
    deadline.checkpoint("after batched coefficient-error numeric validation")
}

impl BatchedLinearBounds {
    fn finite_concretize_plan(
        &self,
        input_bounds: &BoundedTensor,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<FiniteConcretizePlan> {
        deadline.checkpoint("before batched finite shape validation")?;
        if self.input_shape.is_empty() || self.output_shape.is_empty() {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds finite concretization requires non-empty input/output shapes"
                    .into(),
            ));
        }
        if self.lower_a.ndim() < 2 {
            return Err(NyError::InvalidSpec(format!(
                "BatchedLinearBounds: lower_a must have ndim >= 2, got {}",
                self.lower_a.ndim()
            )));
        }
        if !shapes_equal_with_deadline(
            self.lower_a.shape(),
            self.upper_a.shape(),
            deadline,
            "during batched coefficient shape validation",
        )? {
            return Err(NyError::shape_mismatch(
                self.lower_a.shape().to_vec(),
                self.upper_a.shape().to_vec(),
            ));
        }
        if !shapes_equal_with_deadline(
            self.lower_b.shape(),
            self.upper_b.shape(),
            deadline,
            "during batched bias shape validation",
        )? {
            return Err(NyError::shape_mismatch(
                self.lower_b.shape().to_vec(),
                self.upper_b.shape().to_vec(),
            ));
        }
        let coefficient_shape = self.lower_a.shape();
        let bias_shape = &coefficient_shape[..coefficient_shape.len() - 1];
        if !shapes_equal_with_deadline(
            bias_shape,
            self.lower_b.shape(),
            deadline,
            "during batched coefficient/bias shape validation",
        )? {
            return Err(NyError::shape_mismatch(
                bias_shape.to_vec(),
                self.lower_b.shape().to_vec(),
            ));
        }
        if let Some(error) = &self.lower_a_err {
            if !shapes_equal_with_deadline(
                coefficient_shape,
                error.shape(),
                deadline,
                "during batched lower-error shape validation",
            )? {
                return Err(NyError::shape_mismatch(
                    coefficient_shape.to_vec(),
                    error.shape().to_vec(),
                ));
            }
        }
        if let Some(error) = &self.upper_a_err {
            if !shapes_equal_with_deadline(
                coefficient_shape,
                error.shape(),
                deadline,
                "during batched upper-error shape validation",
            )? {
                return Err(NyError::shape_mismatch(
                    coefficient_shape.to_vec(),
                    error.shape().to_vec(),
                ));
            }
        }

        // These metadata vectors do not participate in indexing below beyond
        // the validated shape relation, but still belong to the request schema
        // and therefore receive bounded deadline polling.
        for _ in &self.input_shape {
            deadline.work(1, "during batched input metadata validation")?;
        }
        for _ in &self.output_shape {
            deadline.work(1, "during batched output metadata validation")?;
        }
        deadline.checkpoint("after batched bounds metadata validation")?;

        let got_shape = input_bounds.shape();
        let expected_shape = self.input_shape.as_slice();
        if !shapes_equal_with_deadline(
            input_bounds.lower().shape(),
            input_bounds.upper().shape(),
            deadline,
            "during batched input endpoint shape validation",
        )? {
            return Err(NyError::shape_mismatch(
                input_bounds.lower().shape().to_vec(),
                input_bounds.upper().shape().to_vec(),
            ));
        }
        let input_shape_matches = shapes_equal_with_deadline(
            expected_shape,
            got_shape,
            deadline,
            "during batched expected-input shape validation",
        )?;
        let expected_elements = shape_product_with_deadline(
            expected_shape,
            deadline,
            "during batched expected-input shape product",
        )?;
        let got_elements = input_bounds.lower().len();
        if !input_shape_matches
            && !(expected_elements == got_elements
                && (is_vector_like_with_deadline(expected_shape, deadline)?
                    || is_vector_like_with_deadline(got_shape, deadline)?))
        {
            return Err(NyError::shape_mismatch(
                self.input_shape.clone(),
                got_shape.to_vec(),
            ));
        }
        if expected_elements != got_elements {
            return Err(NyError::shape_mismatch(
                self.input_shape.clone(),
                got_shape.to_vec(),
            ));
        }

        let rank = coefficient_shape.len();
        let m = coefficient_shape[rank - 2];
        let n = coefficient_shape[rank - 1];
        let coefficient_batches = shape_product_with_deadline(
            &coefficient_shape[..rank - 2],
            deadline,
            "during batched coefficient-batch shape product",
        )?;
        let input_last = expected_shape.last().copied().unwrap_or(0);
        let flat_attention =
            rank == 2 && expected_shape.len() > 1 && n == expected_elements && n != input_last;
        let broadcast_coefficients = rank == 2 && expected_shape.len() > 1 && !flat_attention;

        let input_batches = if flat_attention {
            1
        } else if n == 0 {
            if expected_shape.len() > 1 {
                shape_product_with_deadline(
                    &expected_shape[..expected_shape.len() - 1],
                    deadline,
                    "during batched zero-width input-batch shape product",
                )?
            } else {
                1
            }
        } else {
            if input_last != n || expected_elements % n != 0 {
                return Err(NyError::shape_mismatch(vec![n], expected_shape.to_vec()));
            }
            expected_elements / n
        };

        let output_batches = if broadcast_coefficients {
            shape_product_with_deadline(
                &expected_shape[..expected_shape.len() - 1],
                deadline,
                "during batched broadcast-output shape product",
            )?
        } else {
            coefficient_batches
        };
        if output_batches != 0 && input_batches != 1 && input_batches != output_batches {
            return Err(NyError::shape_mismatch(
                vec![output_batches, n],
                expected_shape.to_vec(),
            ));
        }
        let output_elements = output_batches.checked_mul(m).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "BatchedLinearBounds output size overflows: {output_batches} * {m}"
            ))
        })?;
        let output_shape_len = if broadcast_coefficients {
            expected_shape.len()
        } else {
            rank - 1
        };
        let planned_output_shape_matches_metadata = if flat_attention {
            shapes_equal_with_deadline(
                &self.output_shape,
                &[m],
                deadline,
                "during batched flat-output metadata shape validation",
            )?
        } else if broadcast_coefficients {
            if self.output_shape.len() == output_shape_len
                && self.output_shape.last().copied() == Some(m)
            {
                shapes_equal_with_deadline(
                    &self.output_shape[..self.output_shape.len().saturating_sub(1)],
                    &expected_shape[..expected_shape.len().saturating_sub(1)],
                    deadline,
                    "during batched output metadata shape validation",
                )?
            } else {
                deadline.checkpoint("during batched output metadata shape validation")?;
                false
            }
        } else {
            shapes_equal_with_deadline(
                &self.output_shape,
                &coefficient_shape[..rank - 1],
                deadline,
                "during batched output metadata shape validation",
            )?
        };
        if !planned_output_shape_matches_metadata {
            let planned = if broadcast_coefficients {
                let mut shape = expected_shape[..expected_shape.len().saturating_sub(1)].to_vec();
                shape.push(m);
                shape
            } else {
                coefficient_shape[..rank - 1].to_vec()
            };
            return Err(NyError::shape_mismatch(planned, self.output_shape.clone()));
        }
        deadline.checkpoint("after batched finite shape validation")?;
        Ok(FiniteConcretizePlan {
            coefficient_batches,
            output_batches,
            input_batches,
            m,
            n,
            output_elements,
            output_shape_len,
            broadcast_coefficients,
            output_uses_input_batch_shape: broadcast_coefficients,
        })
    }

    fn finite_concretize_nominal_bytes(
        &self,
        input_bounds: &BoundedTensor,
        plan: &FiniteConcretizePlan,
    ) -> usize {
        let mut scratch_elements = 0usize;
        let mut charge_if_nonstandard = |array: &ArrayD<f32>| {
            if array.as_slice().is_none() {
                scratch_elements = scratch_elements.saturating_add(array.len());
            }
        };
        charge_if_nonstandard(&self.lower_a);
        charge_if_nonstandard(&self.upper_a);
        charge_if_nonstandard(&self.lower_b);
        charge_if_nonstandard(&self.upper_b);
        if let Some(error) = &self.lower_a_err {
            charge_if_nonstandard(error);
        }
        if let Some(error) = &self.upper_a_err {
            charge_if_nonstandard(error);
        }
        charge_if_nonstandard(input_bounds.lower());
        charge_if_nonstandard(input_bounds.upper());

        self.memory_bytes()
            .saturating_add(scratch_elements.saturating_mul(size_of::<f32>()))
            .saturating_add(
                plan.output_elements
                    .saturating_mul(2)
                    .saturating_mul(size_of::<f32>()),
            )
            .saturating_add(plan.output_shape_len.saturating_mul(size_of::<usize>()))
    }

    fn validate_finite_concretize_values(
        &self,
        input_bounds: &BoundedTensor,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<()> {
        scan_no_nan_with_deadline(&self.lower_a, "lower_a", deadline)?;
        scan_no_nan_with_deadline(&self.upper_a, "upper_a", deadline)?;
        scan_no_nan_with_deadline(&self.lower_b, "lower_b", deadline)?;
        scan_no_nan_with_deadline(&self.upper_b, "upper_b", deadline)?;
        if let Some(error) = &self.lower_a_err {
            scan_error_with_deadline(error, deadline)?;
        }
        if let Some(error) = &self.upper_a_err {
            scan_error_with_deadline(error, deadline)?;
        }

        for (&lower, &upper) in input_bounds.lower().iter().zip(input_bounds.upper()) {
            if lower.is_nan() || upper.is_nan() {
                return Err(NyError::NumericalInstability(
                    "BatchedLinearBounds input bounds contain NaN".into(),
                ));
            }
            if f32_to_f64_exact(lower) > f32_to_f64_exact(upper) {
                return Err(NyError::InvalidSpec(
                    "BatchedLinearBounds input bounds are inverted".into(),
                ));
            }
            deadline.work(2, "during batched input numeric validation")?;
        }
        deadline.checkpoint("after batched input numeric validation")
    }

    fn try_finite_output_shape(
        &self,
        plan: &FiniteConcretizePlan,
        admission: &mut FiniteConcretizeAdmission,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<Vec<usize>> {
        let mut shape = try_reserved_vec::<usize>(
            plan.output_shape_len,
            admission,
            deadline,
            "during batched output-shape allocation",
        )?;
        let prefix = if plan.output_uses_input_batch_shape {
            &self.input_shape[..self.input_shape.len() - 1]
        } else {
            &self.lower_a.shape()[..self.lower_a.ndim() - 2]
        };
        for &dimension in prefix {
            shape.push(dimension);
            deadline.work(1, "during batched output-shape fill")?;
        }
        shape.push(plan.m);
        deadline.work(1, "during batched output-shape fill")?;
        deadline.checkpoint("after batched output-shape fill")?;
        debug_assert_eq!(shape.len(), plan.output_shape_len);
        Ok(shape)
    }

    #[allow(clippy::too_many_arguments)]
    fn concretize_finite_row(
        lower_a: &[f32],
        upper_a: &[f32],
        lower_b: &[f32],
        upper_b: &[f32],
        lower_a_err: Option<&[f32]>,
        upper_a_err: Option<&[f32]>,
        input_lower: &[f32],
        input_upper: &[f32],
        coefficient_batch: usize,
        input_batch: usize,
        output_index: usize,
        m: usize,
        n: usize,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<(f32, f32)> {
        let row = coefficient_batch
            .checked_mul(m)
            .and_then(|base| base.checked_add(output_index))
            .ok_or_else(|| {
                NyError::InvalidSpec("batched concretization row index overflow".into())
            })?;
        let coefficient_offset = row.checked_mul(n).ok_or_else(|| {
            NyError::InvalidSpec("batched concretization coefficient index overflow".into())
        })?;
        let input_offset = input_batch.checked_mul(n).ok_or_else(|| {
            NyError::InvalidSpec("batched concretization input index overflow".into())
        })?;
        let lb = lower_b[row];
        let ub = upper_b[row];
        let lower_degraded = lb == f32::NEG_INFINITY;
        let upper_degraded = ub == f32::INFINITY;
        let mut lower_overflow = false;
        let mut upper_overflow = false;
        if !lower_degraded || !upper_degraded {
            for j in 0..n {
                if !lower_degraded
                    && !lower_overflow
                    && !crown_coefficient_is_safe_bits(lower_a[coefficient_offset + j])
                {
                    lower_overflow = true;
                }
                if !upper_degraded
                    && !upper_overflow
                    && !crown_coefficient_is_safe_bits(upper_a[coefficient_offset + j])
                {
                    upper_overflow = true;
                }
                deadline.work(2, "during batched finite coefficient guard")?;
                if (lower_degraded || lower_overflow) && (upper_degraded || upper_overflow) {
                    break;
                }
            }
        }

        let lower_guarded = lower_degraded || lower_overflow;
        let upper_guarded = upper_degraded || upper_overflow;
        let lower_f64 = if lower_guarded {
            f64::NEG_INFINITY
        } else {
            certified_affine_sum_f32_with_poll(
                lb,
                (0..n).map(|j| {
                    let coefficient = lower_a[coefficient_offset + j];
                    let endpoint = if coefficient_is_nonnegative_bits(coefficient) {
                        input_lower[input_offset + j]
                    } else {
                        input_upper[input_offset + j]
                    };
                    (coefficient, endpoint)
                }),
                OutwardDirection::Lower,
                |units| deadline.work(units, "during batched finite lower reduction"),
            )?
        };
        let upper_f64 = if upper_guarded {
            f64::INFINITY
        } else {
            certified_affine_sum_f32_with_poll(
                ub,
                (0..n).map(|j| {
                    let coefficient = upper_a[coefficient_offset + j];
                    let endpoint = if coefficient_is_nonnegative_bits(coefficient) {
                        input_upper[input_offset + j]
                    } else {
                        input_lower[input_offset + j]
                    };
                    (coefficient, endpoint)
                }),
                OutwardDirection::Upper,
                |units| deadline.work(units, "during batched finite upper reduction"),
            )?
        };

        // Mirror the historical batched ordering: first publish the affine
        // result, then move that stored center outward by the certified
        // coefficient-error penalty, then apply concretize_sound's final ULP.
        let mut lower = publish_lower_zero_or_normal(lower_f64);
        let mut upper = publish_upper_zero_or_normal(upper_f64);
        if let Some(error) = lower_a_err.filter(|_| !lower_guarded) {
            let penalty = certified_affine_sum_f32_with_poll(
                0.0,
                (0..n).map(|j| {
                    let magnitude = max_abs_f32_bits(
                        input_lower[input_offset + j],
                        input_upper[input_offset + j],
                    );
                    (
                        nonnegative_error_or_infinity(error[coefficient_offset + j]),
                        magnitude,
                    )
                }),
                OutwardDirection::Upper,
                |units| deadline.work(units, "during batched finite lower-error reduction"),
            )?;
            if penalty != 0.0 && lower.is_finite() {
                lower = if penalty.is_finite() {
                    publish_lower_zero_or_normal(next_down_f64(f32_to_f64_exact(lower) - penalty))
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
        if let Some(error) = upper_a_err.filter(|_| !upper_guarded) {
            let penalty = certified_affine_sum_f32_with_poll(
                0.0,
                (0..n).map(|j| {
                    let magnitude = max_abs_f32_bits(
                        input_lower[input_offset + j],
                        input_upper[input_offset + j],
                    );
                    (
                        nonnegative_error_or_infinity(error[coefficient_offset + j]),
                        magnitude,
                    )
                }),
                OutwardDirection::Upper,
                |units| deadline.work(units, "during batched finite upper-error reduction"),
            )?;
            if penalty != 0.0 && upper.is_finite() {
                upper = if penalty.is_finite() {
                    publish_upper_zero_or_normal(next_up_f64(f32_to_f64_exact(upper) + penalty))
                } else {
                    f32::INFINITY
                };
            }
        }

        lower = widen_lower_zero_or_normal(lower);
        upper = widen_upper_zero_or_normal(upper);
        if lower > upper {
            lower = f32::NEG_INFINITY;
            upper = f32::INFINITY;
        }
        Ok((lower, upper))
    }

    fn concretize_sound_finite_with_budget(
        &self,
        input_bounds: &BoundedTensor,
        deadline: &mut PatchesMaterializationDeadline,
        budget_bytes: usize,
    ) -> Result<BoundedTensor> {
        let plan = self.finite_concretize_plan(input_bounds, deadline)?;
        let nominal_required_bytes = self.finite_concretize_nominal_bytes(input_bounds, &plan);
        let mut admission = FiniteConcretizeAdmission::check(nominal_required_bytes, budget_bytes)?;
        deadline.checkpoint("after batched finite memory admission")?;
        self.validate_finite_concretize_values(input_bounds, deadline)?;

        let lower_a = try_flatten_array(
            &self.lower_a,
            &mut admission,
            deadline,
            "during batched lower-a flatten",
        )?;
        let upper_a = try_flatten_array(
            &self.upper_a,
            &mut admission,
            deadline,
            "during batched upper-a flatten",
        )?;
        let lower_b = try_flatten_array(
            &self.lower_b,
            &mut admission,
            deadline,
            "during batched lower-b flatten",
        )?;
        let upper_b = try_flatten_array(
            &self.upper_b,
            &mut admission,
            deadline,
            "during batched upper-b flatten",
        )?;
        let lower_a_err = match &self.lower_a_err {
            Some(error) => Some(try_flatten_array(
                error,
                &mut admission,
                deadline,
                "during batched lower-error flatten",
            )?),
            None => None,
        };
        let upper_a_err = match &self.upper_a_err {
            Some(error) => Some(try_flatten_array(
                error,
                &mut admission,
                deadline,
                "during batched upper-error flatten",
            )?),
            None => None,
        };
        let input_lower = try_flatten_array(
            input_bounds.lower(),
            &mut admission,
            deadline,
            "during batched lower-input flatten",
        )?;
        let input_upper = try_flatten_array(
            input_bounds.upper(),
            &mut admission,
            deadline,
            "during batched upper-input flatten",
        )?;
        let output_shape = self.try_finite_output_shape(&plan, &mut admission, deadline)?;
        let mut concrete_lower = try_reserved_vec::<f32>(
            plan.output_elements,
            &mut admission,
            deadline,
            "during batched lower-output allocation",
        )?;
        let mut concrete_upper = try_reserved_vec::<f32>(
            plan.output_elements,
            &mut admission,
            deadline,
            "during batched upper-output allocation",
        )?;

        if !plan.broadcast_coefficients && plan.output_batches > plan.coefficient_batches {
            return Err(NyError::InternalError(
                "batched finite concretization coefficient-batch plan is inconsistent".into(),
            ));
        }
        for output_batch in 0..plan.output_batches {
            let coefficient_batch = if plan.broadcast_coefficients {
                0
            } else {
                output_batch
            };
            let input_batch = if plan.input_batches == 1 {
                0
            } else {
                output_batch
            };
            for output_index in 0..plan.m {
                let (lower, upper) = Self::concretize_finite_row(
                    lower_a.as_ref(),
                    upper_a.as_ref(),
                    lower_b.as_ref(),
                    upper_b.as_ref(),
                    lower_a_err.as_deref(),
                    upper_a_err.as_deref(),
                    input_lower.as_ref(),
                    input_upper.as_ref(),
                    coefficient_batch,
                    input_batch,
                    output_index,
                    plan.m,
                    plan.n,
                    deadline,
                )?;
                concrete_lower.push(lower);
                concrete_upper.push(upper);
                deadline.work(2, "during batched endpoint publication")?;
            }
        }
        deadline.checkpoint("before batched endpoint array wrapping")?;
        if concrete_lower.len() != plan.output_elements
            || concrete_upper.len() != plan.output_elements
        {
            return Err(NyError::InternalError(
                "batched finite concretization produced the wrong endpoint count".into(),
            ));
        }
        let lower = ArrayD::from_shape_vec(IxDyn(&output_shape), concrete_lower)
            .map_err(|error| NyError::InternalError(format!("batched lower reshape: {error}")))?;
        deadline.checkpoint("after batched lower endpoint array wrapping")?;
        let upper = ArrayD::from_shape_vec(IxDyn(&output_shape), concrete_upper)
            .map_err(|error| NyError::InternalError(format!("batched upper reshape: {error}")))?;
        deadline.checkpoint("after batched upper endpoint array wrapping")?;
        deadline.checkpoint("before batched bounded-tensor validation")?;
        let bounded = BoundedTensor::new_allow_infinite_with_poll(lower, upper, || {
            deadline.checkpoint("during batched bounded-tensor validation")
        })?;
        deadline.checkpoint("after batched bounded-tensor validation")?;
        Ok(bounded)
    }

    /// Concretize batched linear bounds given input bounds.
    ///
    /// For linear bounds A @ x + b, with x in [l, u]:
    /// - Lower bound: A_L_pos @ l + A_L_neg @ u + b_L (per position)
    /// - Upper bound: A_U_pos @ u + A_U_neg @ l + b_U (per position)
    ///
    /// Uses the positive/negative coefficient split for exact concretization,
    /// matching the reference (alpha-beta-CROWN `bound_general.py:1140-1160`).
    ///
    /// REQUIRES: `input_bounds.shape() == self.input_shape`, or the input/expected
    /// shape is vector-like (at most one non-1 dimension) with the same element
    /// count so it can be reshaped to `self.input_shape`.
    /// REQUIRES: `input_bounds.lower() <= input_bounds.upper()` element-wise (well-formed intervals).
    /// ENSURES: For all `x` such that `input_bounds.lower() <= x <= input_bounds.upper()`:
    ///   - `result.lower() <= lower_a @ x + lower_b` (sound lower bound),
    ///   - `result.upper() >= upper_a @ x + upper_b` (sound upper bound).
    ///     ENSURES: `result.shape() == self.output_shape`.
    ///
    /// # Errors
    /// - `NyError::ShapeMismatch` if input shape mismatches the expected bounds shape
    /// - `NyError::ShapeMismatch` if coefficients cannot broadcast to the input batch shape
    /// - `NyError::ShapeMismatch` if coefficient, input, or bias shapes are incompatible
    pub fn concretize(&self, input_bounds: &BoundedTensor) -> Result<BoundedTensor> {
        self.validate_internal_shapes()?;
        self.validate_no_nan()?;
        // input_bounds shape: [...batch, in_dim]
        // self.lower_a shape: [...batch, out_dim, in_dim]
        // self.lower_b shape: [...batch, out_dim]
        // output shape: [...batch, out_dim]

        let expected_shape = self.input_shape.as_slice();
        let got_shape = input_bounds.shape();
        let mut in_lower: Cow<ArrayD<f32>> = Cow::Borrowed(input_bounds.lower());
        let mut in_upper: Cow<ArrayD<f32>> = Cow::Borrowed(input_bounds.upper());

        if got_shape != expected_shape {
            let expected_elems = checked_shape_product(expected_shape).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "BatchedLinearBounds: expected shape product overflows: {:?}",
                    expected_shape
                ))
            })?;
            let got_elems = input_bounds.lower().len();
            let is_vector_like = |shape: &[usize]| shape.iter().filter(|&&d| d > 1).count() <= 1;
            let expected_vector_like = is_vector_like(expected_shape);
            let got_vector_like = is_vector_like(got_shape);

            // Allow reshape when either side is vector-like and element counts match.
            let can_reshape =
                expected_elems == got_elems && (expected_vector_like || got_vector_like);
            if can_reshape {
                let reshaped_lower = input_bounds
                    .lower()
                    .clone()
                    .into_shape_with_order(IxDyn(expected_shape))
                    .map_err(|_| {
                        NyError::shape_mismatch(self.input_shape.clone(), got_shape.to_vec())
                    })?;
                let reshaped_upper = input_bounds
                    .upper()
                    .clone()
                    .into_shape_with_order(IxDyn(expected_shape))
                    .map_err(|_| {
                        NyError::shape_mismatch(self.input_shape.clone(), got_shape.to_vec())
                    })?;
                in_lower = Cow::Owned(reshaped_lower);
                in_upper = Cow::Owned(reshaped_upper);
            } else {
                return Err(NyError::shape_mismatch(
                    self.input_shape.clone(),
                    got_shape.to_vec(),
                ));
            }
        }

        // Handle shape reconciliation between coefficient matrix and input.
        let a_shape = self.lower_a.shape();
        let x_shape = in_lower.shape();

        // Flat coefficients case: A is [out_dim, total_in] from attention graph
        // flatten_to_block_diagonal, and input is multi-dim [batch..., dim].
        // Flatten input to [total_in] for flat matvec instead of broadcasting A.
        let a_in_dim = *a_shape.last().unwrap_or(&0);
        let x_elems = in_lower.len();
        let x_last = *x_shape.last().unwrap_or(&0);
        let is_flat_attn =
            a_shape.len() == 2 && x_shape.len() > 1 && a_in_dim == x_elems && a_in_dim != x_last;

        if is_flat_attn {
            in_lower = Cow::Owned(
                in_lower
                    .as_ref()
                    .clone()
                    .into_shape_with_order(IxDyn(&[a_in_dim]))
                    .map_err(|_| {
                        NyError::InvalidSpec(format!(
                            "concretize flat: cannot reshape input lower to [{}]",
                            a_in_dim
                        ))
                    })?,
            );
            in_upper = Cow::Owned(
                in_upper
                    .as_ref()
                    .clone()
                    .into_shape_with_order(IxDyn(&[a_in_dim]))
                    .map_err(|_| {
                        NyError::InvalidSpec(format!(
                            "concretize flat: cannot reshape input upper to [{}]",
                            a_in_dim
                        ))
                    })?,
            );
        }

        let x_shape = in_lower.shape();
        let needs_broadcast = a_shape.len() == 2 && x_shape.len() > 1;

        // Justification: The 4-tuple represents linear bound coefficients (lower_A, upper_A,
        // lower_b, upper_b) that are either borrowed or owned depending on whether broadcasting
        // is needed. A named struct would add indirection for a local destructuring pattern.
        #[allow(clippy::type_complexity)]
        let (lower_a, upper_a, lower_b, upper_b): (
            Cow<ArrayD<f32>>,
            Cow<ArrayD<f32>>,
            Cow<ArrayD<f32>>,
            Cow<ArrayD<f32>>,
        ) = if needs_broadcast {
            // Coefficients are unbatched [out_dim, in_dim], input is batched [...batch, in_dim]
            // Broadcast by inserting leading batch dimensions
            let x_batch_dims = &x_shape[..x_shape.len() - 1];
            let mut new_a_shape: Vec<usize> = x_batch_dims.to_vec();
            new_a_shape.extend_from_slice(a_shape);
            let mut new_b_shape: Vec<usize> = x_batch_dims.to_vec();
            new_b_shape.push(a_shape[0]); // out_dim

            // Broadcast by repeating the unbatched matrices across batch dims
            let lower_a_bc = self
                .lower_a
                .broadcast(IxDyn(&new_a_shape))
                .ok_or_else(|| {
                    NyError::shape_mismatch(new_a_shape.clone(), self.lower_a.shape().to_vec())
                })?
                .to_owned();
            let upper_a_bc = self
                .upper_a
                .broadcast(IxDyn(&new_a_shape))
                .ok_or_else(|| {
                    NyError::shape_mismatch(new_a_shape.clone(), self.upper_a.shape().to_vec())
                })?
                .to_owned();
            let lower_b_bc = self
                .lower_b
                .broadcast(IxDyn(&new_b_shape))
                .ok_or_else(|| {
                    NyError::shape_mismatch(new_b_shape.clone(), self.lower_b.shape().to_vec())
                })?
                .to_owned();
            let upper_b_bc = self
                .upper_b
                .broadcast(IxDyn(&new_b_shape))
                .ok_or_else(|| {
                    NyError::shape_mismatch(new_b_shape.clone(), self.upper_b.shape().to_vec())
                })?
                .to_owned();
            (
                Cow::Owned(lower_a_bc),
                Cow::Owned(upper_a_bc),
                Cow::Owned(lower_b_bc),
                Cow::Owned(upper_b_bc),
            )
        } else {
            // No broadcasting needed - borrow original arrays (no clone!)
            (
                Cow::Borrowed(&self.lower_a),
                Cow::Borrowed(&self.upper_a),
                Cow::Borrowed(&self.lower_b),
                Cow::Borrowed(&self.upper_b),
            )
        };

        // Positive/negative coefficient split (#2220 Packet B).
        //
        // lower_a and upper_a are coefficients of SEPARATE linear bound functions,
        // not interval bounds on the same coefficient. The correct concretization is:
        //   lower = A_L_pos @ x_l + A_L_neg @ x_u + b_L
        //   upper = A_U_pos @ x_u + A_U_neg @ x_l + b_U
        //
        // This is both tighter and faster than the previous interval matvec approach.
        // Reference: alpha-beta-CROWN bound_general.py:1140-1160.
        let (concrete_lower, concrete_upper) = if Self::all_finite_for_blas(
            &lower_a, &upper_a, &lower_b, &upper_b, &in_lower, &in_upper,
        ) {
            Self::concretize_blas_posneg(
                &lower_a, &upper_a, &lower_b, &upper_b, &in_lower, &in_upper,
            )?
        } else {
            Self::concretize_scalar_posneg(
                &lower_a, &upper_a, &lower_b, &upper_b, &in_lower, &in_upper,
            )?
        };

        // Certified coefficient-error penalty (#vnncomp-aw-soundness). The batched
        // `A·W` is f64-accumulated, but that f64 accumulation still rounds; the
        // per-coefficient error `lower_a_err`/`upper_a_err` bounds `|stored - true|`.
        // Apply the SAME `max(|in_l|,|in_u|)`-scaled OUTWARD penalty the scalar
        // path uses at concretize, so the batched (β-CROWN/BaB) verdict path is
        // NOT 1-ULP optimistic. Skipped when there is no error (exact bounds).
        let (concrete_lower, concrete_upper) =
            self.apply_coeff_err_penalty(concrete_lower, concrete_upper, &in_lower, &in_upper)?;

        // Repair NaN/Inf at the type boundary (#3423). Widen strategy replaces NaN
        // with ±inf and fixes inversions.
        BoundedTensor::new_repaired(concrete_lower, concrete_upper, RepairStrategy::Widen)
    }

    /// Concretize batched linear bounds with directed rounding for soundness.
    ///
    /// Calls `concretize`, then applies a final 1-ULP directed rounding via
    /// `round_for_soundness_inplace`.
    ///
    /// Soundness of the underlying `concretize` is established inside each path,
    /// and rests on the SAME property in both: operands are cast f32→f64 so every
    /// product is exact, the dot product accumulates in f64, and the single
    /// f64→f32 cast is absorbed by a directed `next_down`/`next_up` — see
    /// `concretize_blas_posneg` (BLAS) and `concretize_scalar_posneg` (fallback).
    /// The 1 ULP of widening is measured at the RESULT magnitude, so it can only
    /// cover rounding that also happens at the result — which is why neither path
    /// may form its per-term products in f32, where round-to-nearest biases each
    /// term INWARD at the (possibly far larger) term magnitude. Both paths are
    /// therefore sound and equally tight. The extra 1-ULP applied here is strictly
    /// additive widening — it only makes bounds safer.
    ///
    /// NaN/Inf repair is centralized in `concretize` via `new_repaired(Widen)` (#3423).
    /// `round_for_soundness_inplace` (1-ULP widening) cannot introduce NaN or inversions.
    ///
    /// Reference: alpha-beta-CROWN `__double2float_rd`/`__double2float_ru`
    /// (`cuda_kernels.cu:8-21`).
    pub fn concretize_sound(&self, input_bounds: &BoundedTensor) -> Result<BoundedTensor> {
        let mut result = self.concretize(input_bounds)?;
        result.round_for_soundness_inplace();
        Ok(result)
    }

    /// Cooperatively pollable, memory-admitted sound concretization.
    ///
    /// `None` deliberately delegates to [`Self::concretize_sound`] so the
    /// historical no-deadline BLAS/scalar choice and every published bit remain
    /// unchanged. `Some(deadline)` selects a fallible scalar implementation:
    /// schema and numeric scans, every exact-product reduction, all copies and
    /// endpoint publication, and final [`BoundedTensor`] validation are polled
    /// after at most 4,096 touch-heavy units. The absolute deadline is never
    /// reset. All operation-owned numeric buffers are admitted against the CPU
    /// CROWN dense budget as one total-live-memory receipt.
    ///
    /// A deadline or allocation refusal publishes no tensor and cannot mutate
    /// either borrowed source object.
    pub(crate) fn concretize_sound_with_deadline(
        &self,
        input_bounds: &BoundedTensor,
        deadline: Option<Instant>,
    ) -> Result<BoundedTensor> {
        let Some(deadline) = deadline else {
            return self.concretize_sound(input_bounds);
        };
        let mut deadline = PatchesMaterializationDeadline::new(Some(deadline));
        self.concretize_sound_finite_with_budget(
            input_bounds,
            &mut deadline,
            crate::network::crown_memory::cpu_crown_dense_budget_bytes(),
        )
    }

    #[cfg(test)]
    pub(crate) fn finite_concretize_required_bytes_for_test(
        &self,
        input_bounds: &BoundedTensor,
    ) -> Result<usize> {
        let mut deadline = PatchesMaterializationDeadline::new(None);
        let plan = self.finite_concretize_plan(input_bounds, &mut deadline)?;
        Ok(self.finite_concretize_nominal_bytes(input_bounds, &plan))
    }

    #[cfg(test)]
    pub(crate) fn concretize_sound_with_budget_for_test(
        &self,
        input_bounds: &BoundedTensor,
        deadline: Instant,
        budget_bytes: usize,
    ) -> Result<BoundedTensor> {
        let mut deadline = PatchesMaterializationDeadline::new(Some(deadline));
        self.concretize_sound_finite_with_budget(input_bounds, &mut deadline, budget_bytes)
    }

    #[cfg(test)]
    pub(crate) fn concretize_sound_with_forced_deadline_for_test(
        &self,
        input_bounds: &BoundedTensor,
        stage: &'static str,
    ) -> Result<BoundedTensor> {
        let mut deadline = PatchesMaterializationDeadline::forced_at(stage);
        self.concretize_sound_finite_with_budget(
            input_bounds,
            &mut deadline,
            crate::network::crown_memory::cpu_crown_dense_budget_bytes(),
        )
    }
}
