// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched patches mode bounds for the CROWN graph engine.
//!
//! Phase 4 of #2613. Design: designs/2026-02-28-patches-mode-wrapper-enum-design.md

use ndarray::ArrayD;
#[cfg(test)]
use ndarray::IxDyn;
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};
use std::mem::size_of;
use std::time::Instant;

use super::patches::{
    CrownBounds, PatchesLinearBounds, PatchesMaterializationDeadline, PatchesMaterializationPurpose,
};
use super::{BatchedLinearBounds, BatchedLinearBounds64, LinearBounds};
use crate::network::crown_memory::BatchedDenseMaterializationEstimate;

/// One total-live receipt for finite batched sidecar staging. Every Vec
/// allocated by the transaction reconciles its actual capacity before fill.
struct BatchedSidecarAdmission {
    required_bytes: usize,
    capacity_overage_bytes: usize,
    budget_bytes: usize,
    site: &'static str,
}

impl BatchedSidecarAdmission {
    fn new(required_bytes: usize, site: &'static str) -> Result<Self> {
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        if required_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                site,
            });
        }
        Ok(Self {
            required_bytes,
            capacity_overage_bytes: 0,
            budget_bytes,
            site,
        })
    }

    fn empty_vec<T>(
        &mut self,
        len: usize,
        deadline: &mut PatchesMaterializationDeadline,
        stage: &'static str,
    ) -> Result<Vec<T>> {
        deadline.checkpoint(stage)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: self
                    .required_bytes
                    .saturating_add(self.capacity_overage_bytes),
                budget_bytes: self.budget_bytes,
                site: self.site,
            })?;
        deadline.checkpoint(stage)?;
        let requested_bytes = len.saturating_mul(size_of::<T>());
        let actual_bytes = values.capacity().saturating_mul(size_of::<T>());
        self.capacity_overage_bytes = self
            .capacity_overage_bytes
            .saturating_add(actual_bytes.saturating_sub(requested_bytes));
        let reconciled = self
            .required_bytes
            .saturating_add(self.capacity_overage_bytes);
        if reconciled > self.budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: reconciled,
                budget_bytes: self.budget_bytes,
                site: self.site,
            });
        }
        Ok(values)
    }
}

/// Cooperative f64 fan-in sidecar used only when an absolute deadline is
/// present. The historical opaque sidecar remains the zero-overhead `None`
/// path; finite requests never call its clone/mapv/accumulate/downcast loops.
#[derive(Debug, Clone)]
pub(crate) struct FiniteBatchedLinearBounds64 {
    lower_a: ArrayD<f64>,
    lower_b: ArrayD<f64>,
    upper_a: ArrayD<f64>,
    upper_b: ArrayD<f64>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl FiniteBatchedLinearBounds64 {
    fn memory_bytes(&self) -> usize {
        self.lower_a
            .len()
            .saturating_add(self.lower_b.len())
            .saturating_add(self.upper_a.len())
            .saturating_add(self.upper_b.len())
            .saturating_mul(size_of::<f64>())
            .saturating_add(
                self.input_shape
                    .len()
                    .saturating_add(self.output_shape.len())
                    .saturating_mul(size_of::<usize>()),
            )
    }

    fn copy_usize_slice(
        source: &[usize],
        admission: &mut BatchedSidecarAdmission,
        deadline: &mut PatchesMaterializationDeadline,
        stage: &'static str,
    ) -> Result<Vec<usize>> {
        let mut result = admission.empty_vec(source.len(), deadline, stage)?;
        for &value in source {
            result.push(value);
            deadline.work(1, stage)?;
        }
        deadline.checkpoint(stage)?;
        Ok(result)
    }

    fn coefficient_error_rows(
        error: Option<&ArrayD<f32>>,
        rows: usize,
        columns: usize,
        expected_len: usize,
        admission: &mut BatchedSidecarAdmission,
        deadline: &mut PatchesMaterializationDeadline,
        stage: &'static str,
    ) -> Result<Vec<u8>> {
        let mut affected = admission.empty_vec(rows, deadline, stage)?;
        for _ in 0..rows {
            affected.push(0);
            deadline.work(1, stage)?;
        }
        let Some(error) = error else {
            deadline.checkpoint(stage)?;
            return Ok(affected);
        };
        if error.len() != expected_len {
            for value in &mut affected {
                *value = 1;
                deadline.work(1, stage)?;
            }
            deadline.checkpoint(stage)?;
            return Ok(affected);
        }
        if columns == 0 {
            deadline.checkpoint(stage)?;
            return Ok(affected);
        }
        for (index, &value) in error.iter().enumerate() {
            if value != 0.0 {
                affected[index / columns] = 1;
            }
            deadline.work(1, stage)?;
        }
        deadline.checkpoint(stage)?;
        Ok(affected)
    }

    #[allow(clippy::too_many_arguments)]
    fn f64_array_from_f32(
        source: &ArrayD<f32>,
        affected: &[u8],
        columns: usize,
        coefficient: bool,
        conservative_value: f64,
        admission: &mut BatchedSidecarAdmission,
        deadline: &mut PatchesMaterializationDeadline,
        stage: &'static str,
    ) -> Result<ArrayD<f64>> {
        let mut values = admission.empty_vec(source.len(), deadline, stage)?;
        for (index, &value) in source.iter().enumerate() {
            let row = if coefficient {
                debug_assert!(columns != 0 || source.is_empty());
                index / columns
            } else {
                index
            };
            values.push(if affected.get(row).copied().unwrap_or(1) != 0 {
                conservative_value
            } else {
                f64::from(value)
            });
            deadline.work(1, stage)?;
        }
        deadline.checkpoint(stage)?;
        ArrayD::from_shape_vec(source.raw_dim(), values).map_err(|error| {
            NyError::InternalError(format!(
                "finite batched sidecar {stage} shape construction failed: {error}"
            ))
        })
    }

    fn from_f32(
        bounds: &BatchedLinearBounds,
        admission: &mut BatchedSidecarAdmission,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<Self> {
        deadline.checkpoint("before finite batched f64 staging")?;
        let columns = bounds.lower_a().shape().last().copied().unwrap_or(0);
        let rows = bounds.lower_b().len();
        let expected_coefficients = rows.checked_mul(columns).ok_or_else(|| {
            NyError::InvalidSpec(
                "finite batched sidecar coefficient dimensions overflow".to_string(),
            )
        })?;
        if bounds.lower_a().len() != expected_coefficients
            || bounds.upper_a().len() != expected_coefficients
            || bounds.upper_b().len() != rows
        {
            return Err(NyError::InvalidSpec(
                "finite batched sidecar received inconsistent coefficient/bias shapes".to_string(),
            ));
        }
        let lower_affected = Self::coefficient_error_rows(
            bounds.lower_a_err.as_ref(),
            rows,
            columns,
            expected_coefficients,
            admission,
            deadline,
            "during finite lower certificate scan",
        )?;
        let upper_affected = Self::coefficient_error_rows(
            bounds.upper_a_err.as_ref(),
            rows,
            columns,
            expected_coefficients,
            admission,
            deadline,
            "during finite upper certificate scan",
        )?;
        let lower_a = Self::f64_array_from_f32(
            bounds.lower_a(),
            &lower_affected,
            columns,
            true,
            0.0,
            admission,
            deadline,
            "during finite lower coefficient staging",
        )?;
        let lower_b = Self::f64_array_from_f32(
            bounds.lower_b(),
            &lower_affected,
            columns,
            false,
            f64::NEG_INFINITY,
            admission,
            deadline,
            "during finite lower bias staging",
        )?;
        let upper_a = Self::f64_array_from_f32(
            bounds.upper_a(),
            &upper_affected,
            columns,
            true,
            0.0,
            admission,
            deadline,
            "during finite upper coefficient staging",
        )?;
        let upper_b = Self::f64_array_from_f32(
            bounds.upper_b(),
            &upper_affected,
            columns,
            false,
            f64::INFINITY,
            admission,
            deadline,
            "during finite upper bias staging",
        )?;
        let input_shape = Self::copy_usize_slice(
            bounds.input_shape(),
            admission,
            deadline,
            "during finite input-shape staging",
        )?;
        let output_shape = Self::copy_usize_slice(
            bounds.output_shape(),
            admission,
            deadline,
            "during finite output-shape staging",
        )?;
        deadline.checkpoint("after finite batched f64 staging")?;
        Ok(Self {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            input_shape,
            output_shape,
        })
    }

    fn copy_f64_array(
        source: &ArrayD<f64>,
        admission: &mut BatchedSidecarAdmission,
        deadline: &mut PatchesMaterializationDeadline,
        stage: &'static str,
    ) -> Result<ArrayD<f64>> {
        let mut values = admission.empty_vec(source.len(), deadline, stage)?;
        for &value in source {
            values.push(value);
            deadline.work(1, stage)?;
        }
        deadline.checkpoint(stage)?;
        ArrayD::from_shape_vec(source.raw_dim(), values).map_err(|error| {
            NyError::InternalError(format!(
                "finite batched sidecar {stage} shape construction failed: {error}"
            ))
        })
    }

    fn staged_copy(
        &self,
        admission: &mut BatchedSidecarAdmission,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<Self> {
        let lower_a = Self::copy_f64_array(
            &self.lower_a,
            admission,
            deadline,
            "during finite lower sidecar copy",
        )?;
        let lower_b = Self::copy_f64_array(
            &self.lower_b,
            admission,
            deadline,
            "during finite lower-bias sidecar copy",
        )?;
        let upper_a = Self::copy_f64_array(
            &self.upper_a,
            admission,
            deadline,
            "during finite upper sidecar copy",
        )?;
        let upper_b = Self::copy_f64_array(
            &self.upper_b,
            admission,
            deadline,
            "during finite upper-bias sidecar copy",
        )?;
        let input_shape = Self::copy_usize_slice(
            &self.input_shape,
            admission,
            deadline,
            "during finite sidecar input-shape copy",
        )?;
        let output_shape = Self::copy_usize_slice(
            &self.output_shape,
            admission,
            deadline,
            "during finite sidecar output-shape copy",
        )?;
        Ok(Self {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            input_shape,
            output_shape,
        })
    }

    fn compatible_with(&self, incoming: &BatchedLinearBounds) -> bool {
        let same_shapes = self.lower_a.shape() == incoming.lower_a().shape()
            && self.lower_b.shape() == incoming.lower_b().shape()
            && self.upper_a.shape() == incoming.upper_a().shape()
            && self.upper_b.shape() == incoming.upper_b().shape()
            && self.input_shape == incoming.input_shape()
            && self.output_shape == incoming.output_shape();
        if same_shapes {
            return true;
        }
        self.lower_a.len() == incoming.lower_a().len()
            && self.lower_b.len() == incoming.lower_b().len()
            && self.upper_a.len() == incoming.upper_a().len()
            && self.upper_b.len() == incoming.upper_b().len()
            && checked_shape_product(&self.input_shape)
                .zip(checked_shape_product(incoming.input_shape()))
                .is_some_and(|(existing, new)| existing == new)
            && checked_shape_product(&self.output_shape)
                .zip(checked_shape_product(incoming.output_shape()))
                .is_some_and(|(existing, new)| existing == new)
            && incoming.lower_a().as_slice().is_some()
            && incoming.lower_b().as_slice().is_some()
            && incoming.upper_a().as_slice().is_some()
            && incoming.upper_b().as_slice().is_some()
    }

    fn widen_in_place(&mut self, deadline: &mut PatchesMaterializationDeadline) -> Result<()> {
        for value in &mut self.lower_a {
            *value = f64::NEG_INFINITY;
            deadline.work(1, "during finite lower sidecar widening")?;
        }
        for value in &mut self.lower_b {
            *value = f64::NEG_INFINITY;
            deadline.work(1, "during finite lower-bias sidecar widening")?;
        }
        for value in &mut self.upper_a {
            *value = f64::INFINITY;
            deadline.work(1, "during finite upper sidecar widening")?;
        }
        for value in &mut self.upper_b {
            *value = f64::INFINITY;
            deadline.work(1, "during finite upper-bias sidecar widening")?;
        }
        deadline.checkpoint("after finite sidecar widening")
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_array(
        existing: &mut ArrayD<f64>,
        incoming: &ArrayD<f32>,
        affected: &[u8],
        columns: usize,
        coefficient: bool,
        conservative_value: f64,
        nan_fallback: f64,
        deadline: &mut PatchesMaterializationDeadline,
        stage: &'static str,
    ) -> Result<()> {
        for (index, (existing_value, &incoming_value)) in
            existing.iter_mut().zip(incoming.iter()).enumerate()
        {
            let row = if coefficient { index / columns } else { index };
            let contribution = if affected.get(row).copied().unwrap_or(1) != 0 {
                conservative_value
            } else {
                f64::from(incoming_value)
            };
            let sum = *existing_value + contribution;
            *existing_value = if existing_value.is_nan() || contribution.is_nan() || sum.is_nan() {
                nan_fallback
            } else {
                sum
            };
            deadline.work(1, stage)?;
        }
        deadline.checkpoint(stage)
    }

    fn accumulate(
        &mut self,
        incoming: &BatchedLinearBounds,
        admission: &mut BatchedSidecarAdmission,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<()> {
        deadline.checkpoint("before finite batched f64 accumulation")?;
        if !self.compatible_with(incoming) {
            return self.widen_in_place(deadline);
        }
        let columns = incoming.lower_a().shape().last().copied().unwrap_or(0);
        let rows = incoming.lower_b().len();
        let expected_coefficients = rows.saturating_mul(columns);
        let lower_affected = Self::coefficient_error_rows(
            incoming.lower_a_err.as_ref(),
            rows,
            columns,
            expected_coefficients,
            admission,
            deadline,
            "during finite incoming lower certificate scan",
        )?;
        let upper_affected = Self::coefficient_error_rows(
            incoming.upper_a_err.as_ref(),
            rows,
            columns,
            expected_coefficients,
            admission,
            deadline,
            "during finite incoming upper certificate scan",
        )?;
        Self::accumulate_array(
            &mut self.lower_a,
            incoming.lower_a(),
            &lower_affected,
            columns,
            true,
            0.0,
            f64::NEG_INFINITY,
            deadline,
            "during finite lower coefficient accumulation",
        )?;
        Self::accumulate_array(
            &mut self.lower_b,
            incoming.lower_b(),
            &lower_affected,
            columns,
            false,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            deadline,
            "during finite lower bias accumulation",
        )?;
        Self::accumulate_array(
            &mut self.upper_a,
            incoming.upper_a(),
            &upper_affected,
            columns,
            true,
            0.0,
            f64::INFINITY,
            deadline,
            "during finite upper coefficient accumulation",
        )?;
        Self::accumulate_array(
            &mut self.upper_b,
            incoming.upper_b(),
            &upper_affected,
            columns,
            false,
            f64::INFINITY,
            f64::INFINITY,
            deadline,
            "during finite upper bias accumulation",
        )?;
        deadline.checkpoint("after finite batched f64 accumulation")
    }

    #[inline]
    fn downcast_coefficient(value: f64, lower: bool) -> Option<f32> {
        if !value.is_finite() {
            return None;
        }
        let cast = value as f32;
        if !cast.is_finite() {
            return None;
        }
        Some(if lower {
            next_down_f32(cast)
        } else {
            next_up_f32(cast)
        })
    }

    fn to_f32(
        &self,
        admission: &mut BatchedSidecarAdmission,
        deadline: &mut PatchesMaterializationDeadline,
    ) -> Result<BatchedLinearBounds> {
        deadline.checkpoint("before finite batched f64 downcast")?;
        let columns = self.lower_a.shape().last().copied().unwrap_or(0);
        let rows = self.lower_b.len();
        let expected_coefficients = rows.saturating_mul(columns);
        if self.lower_a.len() != expected_coefficients
            || self.upper_a.len() != expected_coefficients
            || self.upper_b.len() != rows
        {
            return Err(NyError::InvalidSpec(
                "finite batched f64 sidecar has inconsistent shapes".to_string(),
            ));
        }
        let lower_a_source = self.lower_a.as_slice().ok_or_else(|| {
            NyError::InvalidSpec("finite lower f64 sidecar is not contiguous".to_string())
        })?;
        let lower_b_source = self.lower_b.as_slice().ok_or_else(|| {
            NyError::InvalidSpec("finite lower-bias f64 sidecar is not contiguous".to_string())
        })?;
        let upper_a_source = self.upper_a.as_slice().ok_or_else(|| {
            NyError::InvalidSpec("finite upper f64 sidecar is not contiguous".to_string())
        })?;
        let upper_b_source = self.upper_b.as_slice().ok_or_else(|| {
            NyError::InvalidSpec("finite upper-bias f64 sidecar is not contiguous".to_string())
        })?;
        let mut lower_a = admission.empty_vec(
            expected_coefficients,
            deadline,
            "before finite lower downcast allocation",
        )?;
        let mut lower_b = admission.empty_vec(
            rows,
            deadline,
            "before finite lower-bias downcast allocation",
        )?;
        let mut upper_a = admission.empty_vec(
            expected_coefficients,
            deadline,
            "before finite upper downcast allocation",
        )?;
        let mut upper_b = admission.empty_vec(
            rows,
            deadline,
            "before finite upper-bias downcast allocation",
        )?;
        for row in 0..rows {
            let offset = row.saturating_mul(columns);
            let lower_bias = lower_b_source.get(row).copied();
            let upper_bias = upper_b_source.get(row).copied();
            let biases = lower_bias
                .and_then(|value| {
                    if value == f64::NEG_INFINITY {
                        Some(f32::NEG_INFINITY)
                    } else {
                        Self::downcast_coefficient(value, true)
                    }
                })
                .zip(upper_bias.and_then(|value| {
                    if value == f64::INFINITY {
                        Some(f32::INFINITY)
                    } else {
                        Self::downcast_coefficient(value, false)
                    }
                }));
            let mut row_valid = biases.is_some();
            if row_valid {
                for column in 0..columns {
                    let index = offset + column;
                    row_valid &= lower_a_source
                        .get(index)
                        .copied()
                        .and_then(|value| Self::downcast_coefficient(value, true))
                        .is_some();
                    row_valid &= upper_a_source
                        .get(index)
                        .copied()
                        .and_then(|value| Self::downcast_coefficient(value, false))
                        .is_some();
                    deadline.work(2, "during finite f64 row validation")?;
                }
            }
            if row_valid {
                let (lower_bias, upper_bias) = biases.expect("row validity checked biases");
                for column in 0..columns {
                    let index = offset + column;
                    lower_a.push(
                        Self::downcast_coefficient(
                            lower_a_source.get(index).copied().unwrap_or(f64::NAN),
                            true,
                        )
                        .expect("row validation checked lower coefficient"),
                    );
                    upper_a.push(
                        Self::downcast_coefficient(
                            upper_a_source.get(index).copied().unwrap_or(f64::NAN),
                            false,
                        )
                        .expect("row validation checked upper coefficient"),
                    );
                    deadline.work(2, "during finite f64 row publication")?;
                }
                lower_b.push(lower_bias);
                upper_b.push(upper_bias);
            } else {
                for _ in 0..columns {
                    lower_a.push(0.0);
                    upper_a.push(0.0);
                    deadline.work(2, "during finite conservative row publication")?;
                }
                lower_b.push(f32::NEG_INFINITY);
                upper_b.push(f32::INFINITY);
            }
            deadline.work(2, "during finite f64 bias publication")?;
        }
        deadline.checkpoint("after finite batched f64 downcast")?;
        let lower_a = ArrayD::from_shape_vec(self.lower_a.raw_dim(), lower_a).map_err(|error| {
            NyError::InternalError(format!("finite lower downcast shape failed: {error}"))
        })?;
        let lower_b = ArrayD::from_shape_vec(self.lower_b.raw_dim(), lower_b).map_err(|error| {
            NyError::InternalError(format!("finite lower-bias downcast shape failed: {error}"))
        })?;
        let upper_a = ArrayD::from_shape_vec(self.upper_a.raw_dim(), upper_a).map_err(|error| {
            NyError::InternalError(format!("finite upper downcast shape failed: {error}"))
        })?;
        let upper_b = ArrayD::from_shape_vec(self.upper_b.raw_dim(), upper_b).map_err(|error| {
            NyError::InternalError(format!("finite upper-bias downcast shape failed: {error}"))
        })?;
        let input_shape = Self::copy_usize_slice(
            &self.input_shape,
            admission,
            deadline,
            "during finite downcast input-shape copy",
        )?;
        let output_shape = Self::copy_usize_slice(
            &self.output_shape,
            admission,
            deadline,
            "during finite downcast output-shape copy",
        )?;
        deadline.checkpoint("before finite batched downcast publication")?;
        Ok(BatchedLinearBounds {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            input_shape,
            output_shape,
            lower_a_err: None,
            upper_a_err: None,
        })
    }
}

/// Batched wrapper enum for Patches mode in the batched CROWN graph engine.
///
/// Mirrors [`CrownBounds`] for the batched backward path. The batched graph
/// engine (`crown_batched.rs`) operates on `HashMap<String, BatchedCrownBounds>`
/// instead of `HashMap<String, BatchedLinearBounds>` to support Patches mode.
///
/// **MVP constraint:** The Patches variant stores unbatched `PatchesLinearBounds`.
/// This is correct because Conv2d backward is specification-independent — the same
/// kernel applies to all specs. At nonlinear layers (where per-spec slopes differ),
/// `ensure_batched_dense()` converts to `BatchedLinearBounds` before dispatch.
// BatchedLinearBounds is 496 bytes (hot path in backward loop for transformers);
// Patches is heap-allocated via Box. The size difference is acceptable — boxing
// Dense would add deref overhead on every backward step for non-CNN networks.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum BatchedCrownBounds {
    /// Standard dense batched A-matrix bounds (transformers, post-activation).
    Dense(BatchedLinearBounds),
    /// Dense merge-point accumulator kept in f64 until the node is consumed.
    Dense64(BatchedLinearBounds64),
    /// Pollable/fallible f64 merge-point accumulator for one finite authority.
    FiniteDense64(FiniteBatchedLinearBounds64),
    /// Sparse conv patches bounds, shared across specifications.
    /// Uses unbatched PatchesLinearBounds because conv backward is spec-independent.
    Patches(Box<PatchesLinearBounds>),
}

impl BatchedCrownBounds {
    #[inline]
    fn check_deadline(deadline: Option<Instant>, site: &'static str) -> Result<()> {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(format!(
                "{site}: deadline exceeded"
            )));
        }
        Ok(())
    }

    /// Logical live payload of a pre-existing batched f32 carrier. ndarray
    /// does not expose the backing Vec capacity through a shared array, so the
    /// process-envelope clamp remains the allocator-slack authority. When any
    /// coefficient certificate is present, charge both possible error arrays:
    /// only the presence bit is exposed at this boundary.
    fn batched_dense_retained_bytes(bounds: &BatchedLinearBounds) -> usize {
        let coefficient_bytes = bounds
            .lower_a()
            .len()
            .saturating_add(bounds.upper_a().len())
            .saturating_mul(size_of::<f32>());
        bounds
            .memory_bytes()
            .saturating_add(if bounds.has_coeff_err() {
                coefficient_bytes
            } else {
                0
            })
            .saturating_add(
                bounds
                    .input_shape()
                    .len()
                    .saturating_add(bounds.output_shape().len())
                    .saturating_mul(size_of::<usize>()),
            )
    }

    fn f64_sidecar_bytes_for_dense(bounds: &BatchedLinearBounds) -> usize {
        bounds
            .lower_a()
            .len()
            .saturating_add(bounds.lower_b().len())
            .saturating_add(bounds.upper_a().len())
            .saturating_add(bounds.upper_b().len())
            .saturating_mul(size_of::<f64>())
            .saturating_add(
                bounds
                    .input_shape()
                    .len()
                    .saturating_add(bounds.output_shape().len())
                    .saturating_mul(size_of::<usize>()),
            )
    }

    fn dense_storage_bytes(rows: usize, columns: usize, element_bytes: usize) -> usize {
        rows.saturating_mul(columns)
            .saturating_mul(2)
            .saturating_add(rows.saturating_mul(2))
            .saturating_mul(element_bytes)
    }

    /// Receipt for the Patches -> f32 Dense -> f64 sidecar transaction. The
    /// structured source remains live until publication, as do the incoming
    /// contribution and staged f32 relation. A shape-mismatch accumulation may
    /// allocate one replacement f64 array before dropping its predecessor, so
    /// charge the largest possible coefficient array as operation scratch.
    fn guard_patches_sidecar_staging(
        patches: &PatchesLinearBounds,
        incoming: &BatchedLinearBounds,
        rows: usize,
        columns: usize,
        preserve_certificate: bool,
    ) -> Result<BatchedSidecarAdmission> {
        let shape_bytes = 2usize.saturating_mul(size_of::<usize>());
        let staged_dense = Self::dense_storage_bytes(rows, columns, size_of::<f32>())
            // Full Patches materialization can carry one certified error
            // matrix per coefficient side in addition to A/b.
            .saturating_add(
                rows.saturating_mul(columns)
                    .saturating_mul(2)
                    .saturating_mul(size_of::<f32>()),
            )
            .saturating_add(shape_bytes);
        let staged_sidecar = Self::dense_storage_bytes(rows, columns, size_of::<f64>())
            .saturating_add(shape_bytes)
            .saturating_add(if preserve_certificate {
                rows.saturating_mul(columns)
                    .saturating_mul(2)
                    .saturating_mul(size_of::<f64>())
            } else {
                0
            });
        let certificate_rows = rows.saturating_mul(2).saturating_mul(size_of::<u8>());
        let required_bytes = patches
            .memory_bytes()
            .saturating_add(Self::batched_dense_retained_bytes(incoming))
            .saturating_add(staged_dense)
            .saturating_add(staged_sidecar)
            .saturating_add(certificate_rows);
        BatchedSidecarAdmission::new(
            required_bytes,
            "BatchedCrownBounds Patches f64 sidecar staging",
        )
    }

    fn guard_dense_sidecar_staging(
        existing: &BatchedLinearBounds,
        incoming: &BatchedLinearBounds,
        clone_existing: bool,
    ) -> Result<BatchedSidecarAdmission> {
        let existing_bytes = Self::batched_dense_retained_bytes(existing);
        let certificate_rows = existing
            .lower_b()
            .len()
            .saturating_mul(2)
            .saturating_mul(size_of::<u8>());
        let required_bytes = existing_bytes
            .saturating_add(Self::batched_dense_retained_bytes(incoming))
            .saturating_add(if clone_existing { existing_bytes } else { 0 })
            .saturating_add(Self::f64_sidecar_bytes_for_dense(existing))
            .saturating_add(certificate_rows);
        BatchedSidecarAdmission::new(
            required_bytes,
            "BatchedCrownBounds Dense f64 sidecar staging",
        )
    }

    fn guard_existing_finite_sidecar_staging(
        existing: &FiniteBatchedLinearBounds64,
        incoming: &BatchedLinearBounds,
    ) -> Result<BatchedSidecarAdmission> {
        let certificate_rows = incoming
            .lower_b()
            .len()
            .saturating_mul(2)
            .saturating_mul(size_of::<u8>());
        let required_bytes = existing
            .memory_bytes()
            .saturating_mul(2)
            .saturating_add(Self::batched_dense_retained_bytes(incoming))
            .saturating_add(certificate_rows);
        BatchedSidecarAdmission::new(
            required_bytes,
            "BatchedCrownBounds existing f64 sidecar staging",
        )
    }

    fn finite_sidecar_downcast_admission(
        existing: &FiniteBatchedLinearBounds64,
    ) -> Result<BatchedSidecarAdmission> {
        BatchedSidecarAdmission::new(
            existing.memory_bytes().saturating_mul(2),
            "BatchedCrownBounds finite f64 sidecar downcast",
        )
    }

    /// Convert to `BatchedLinearBounds`, materializing Dense if Patches. Consumes self.
    ///
    /// For the Patches variant, materializes the full dense A-matrix via
    /// `PatchesLinearBounds::to_dense()` then wraps as `BatchedLinearBounds`
    /// with no batch dimensions (single-spec equivalent).
    #[cfg(test)]
    pub(crate) fn into_batched_dense(self) -> Result<BatchedLinearBounds> {
        self.into_batched_dense_with_deadline(None)
    }

    /// Deadline-aware consuming conversion. Patches materialization is one
    /// total-live, move-based transaction; already-dense carriers retain the
    /// historical pass-through behavior.
    pub(crate) fn into_batched_dense_with_deadline(
        self,
        deadline: Option<Instant>,
    ) -> Result<BatchedLinearBounds> {
        match self {
            BatchedCrownBounds::Dense(blb) => Ok(blb),
            BatchedCrownBounds::Dense64(blb) => {
                if deadline.is_some() {
                    return Err(NyError::UnsupportedConfiguration(
                        "finite batched request encountered a legacy opaque f64 sidecar".into(),
                    ));
                }
                Ok(blb.into_f32())
            }
            BatchedCrownBounds::FiniteDense64(blb) => {
                let mut authority = PatchesMaterializationDeadline::new(deadline);
                let mut admission = Self::finite_sidecar_downcast_admission(&blb)?;
                blb.to_f32(&mut admission, &mut authority)
            }
            BatchedCrownBounds::Patches(plb) => Self::patches_to_batched_linear(&plb, deadline),
        }
    }

    /// Convert Patches to Dense in-place, returning `&mut BatchedLinearBounds`.
    ///
    /// If already Dense, returns the inner `BatchedLinearBounds` directly.
    /// If Patches, materializes to Dense first.
    #[cfg(test)]
    pub(crate) fn ensure_batched_dense(&mut self) -> Result<&mut BatchedLinearBounds> {
        self.ensure_batched_dense_with_deadline(None)
    }

    /// Deadline-aware transactional in-place conversion. A Patches carrier is
    /// borrowed through every fallible validation/allocation/fill checkpoint
    /// and replaced only after the final deadline check. Resource or deadline
    /// refusal therefore leaves the exact structured carrier installed.
    #[cfg(test)]
    pub(crate) fn ensure_batched_dense_with_deadline(
        &mut self,
        deadline: Option<Instant>,
    ) -> Result<&mut BatchedLinearBounds> {
        if let BatchedCrownBounds::Patches(plb) = self {
            let dense = Self::patches_to_batched_linear(plb, deadline)?;
            if deadline.is_some_and(|limit| Instant::now() >= limit) {
                return Err(NyError::DeadlineExceeded(
                    "batched patches materialization: deadline exceeded before publication".into(),
                ));
            }
            *self = BatchedCrownBounds::Dense(dense);
        } else if matches!(
            self,
            BatchedCrownBounds::Dense64(_) | BatchedCrownBounds::FiniteDense64(_)
        ) {
            if let BatchedCrownBounds::FiniteDense64(existing) = self {
                let mut authority = PatchesMaterializationDeadline::new(deadline);
                let mut admission = Self::finite_sidecar_downcast_admission(existing)?;
                let dense = existing.to_f32(&mut admission, &mut authority)?;
                authority.checkpoint("before finite batched in-place publication")?;
                *self = BatchedCrownBounds::Dense(dense);
            } else if deadline.is_some() {
                return Err(NyError::UnsupportedConfiguration(
                    "finite batched request encountered a legacy opaque f64 sidecar".into(),
                ));
            } else {
                // The no-authority compatibility API retains its historical
                // move-based downcast. Patches never uses this placeholder.
                let previous =
                    std::mem::replace(self, BatchedCrownBounds::Dense(Self::placeholder_dense()));
                let BatchedCrownBounds::Dense64(blb) = previous else {
                    unreachable!()
                };
                *self = BatchedCrownBounds::Dense(blb.into_f32());
            }
        }
        match self {
            BatchedCrownBounds::Dense(blb) => Ok(blb),
            BatchedCrownBounds::Dense64(_)
            | BatchedCrownBounds::FiniteDense64(_)
            | BatchedCrownBounds::Patches(_) => unreachable!(),
        }
    }

    /// Budget-checked conversion to Dense (#3550). Returns `CpuMemoryExceeded` if
    /// the Patches-to-Dense materialization would exceed the CPU dense budget.
    ///
    /// Dense variant passes through without a budget check (already materialized).
    #[cfg(test)]
    pub(crate) fn into_batched_dense_checked(
        self,
        site: &'static str,
    ) -> Result<BatchedLinearBounds> {
        self.into_batched_dense_checked_with_deadline(site, None)
    }

    /// Budget- and deadline-checked consuming conversion.
    pub(crate) fn into_batched_dense_checked_with_deadline(
        self,
        site: &'static str,
        deadline: Option<Instant>,
    ) -> Result<BatchedLinearBounds> {
        Self::check_deadline(deadline, site)?;
        match &self {
            BatchedCrownBounds::Patches(plb) => {
                let (out_dim, in_dim) = plb.dense_pair_shape()?;
                Self::check_deadline(deadline, site)?;
                BatchedDenseMaterializationEstimate::new(site, 1, out_dim, in_dim)
                    .check_budget()?;
            }
            BatchedCrownBounds::Dense64(_) => {
                if deadline.is_some() {
                    return Err(NyError::UnsupportedConfiguration(
                        "finite batched request encountered a legacy opaque f64 sidecar".into(),
                    ));
                }
            }
            BatchedCrownBounds::FiniteDense64(_) => {}
            BatchedCrownBounds::Dense(_) => {}
        }
        self.into_batched_dense_with_deadline(deadline)
    }

    /// Budget-checked in-place conversion to Dense (#3550).
    ///
    /// If already Dense, returns the inner reference directly.
    /// If Patches, checks budget before materializing.
    #[cfg(test)]
    pub(crate) fn ensure_batched_dense_checked(
        &mut self,
        site: &'static str,
    ) -> Result<&mut BatchedLinearBounds> {
        self.ensure_batched_dense_checked_with_deadline(site, None)
    }

    /// Budget- and deadline-checked transactional in-place conversion.
    #[cfg(test)]
    pub(crate) fn ensure_batched_dense_checked_with_deadline(
        &mut self,
        site: &'static str,
        deadline: Option<Instant>,
    ) -> Result<&mut BatchedLinearBounds> {
        Self::check_deadline(deadline, site)?;
        match self {
            BatchedCrownBounds::Patches(plb) => {
                let (out_dim, in_dim) = plb.dense_pair_shape()?;
                Self::check_deadline(deadline, site)?;
                BatchedDenseMaterializationEstimate::new(site, 1, out_dim, in_dim)
                    .check_budget()?;
            }
            BatchedCrownBounds::Dense64(_) => {
                if deadline.is_some() {
                    return Err(NyError::UnsupportedConfiguration(
                        "finite batched request encountered a legacy opaque f64 sidecar".into(),
                    ));
                }
            }
            BatchedCrownBounds::FiniteDense64(_) => {}
            BatchedCrownBounds::Dense(_) => {}
        }
        self.ensure_batched_dense_with_deadline(deadline)
    }

    /// Helper: convert `PatchesLinearBounds` to `BatchedLinearBounds`.
    ///
    /// Materializes the dense A-matrices and wraps with flattened 1D shapes.
    /// The resulting `BatchedLinearBounds` has no batch dimensions — the
    /// A-matrices are `[out_dim, in_dim]` (2D) and biases are `[out_dim]` (1D).
    fn patches_to_batched_linear(
        plb: &PatchesLinearBounds,
        deadline: Option<Instant>,
    ) -> Result<BatchedLinearBounds> {
        Self::patches_to_batched_linear_with_resident(plb, deadline, 0)
    }

    fn patches_to_batched_linear_with_resident(
        plb: &PatchesLinearBounds,
        deadline: Option<Instant>,
        resident_base_bytes: usize,
    ) -> Result<BatchedLinearBounds> {
        let dense_lb = plb.to_dense_with_deadline_and_resident_for_purpose(
            deadline,
            resident_base_bytes,
            PatchesMaterializationPurpose::Other,
        )?;
        let batched = if deadline.is_some() {
            Self::linear_into_batched_preserving_certificate(dense_lb, deadline)?
        } else {
            Self::linear_into_batched(dense_lb)
        };
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(
                "batched patches materialization: deadline exceeded after move-based wrapping"
                    .into(),
            ));
        }
        Ok(batched)
    }

    /// Move all six scalar buffers into batched storage. Finite paths retain
    /// the coefficient certificate so the cooperative sidecar/concretizer can
    /// consume it without an opaque discharge scan.
    fn singleton_shape_with_deadline(
        value: usize,
        deadline: Option<Instant>,
        site: &'static str,
    ) -> Result<Vec<usize>> {
        Self::check_deadline(deadline, site)?;
        let budget_bytes = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
        if size_of::<usize>() > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: size_of::<usize>(),
                budget_bytes,
                site,
            });
        }
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(1)
            .map_err(|_| NyError::CpuMemoryExceeded {
                required_bytes: size_of::<usize>(),
                budget_bytes,
                site,
            })?;
        Self::check_deadline(deadline, site)?;
        let actual_bytes = shape.capacity().saturating_mul(size_of::<usize>());
        if actual_bytes > budget_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: actual_bytes,
                budget_bytes,
                site,
            });
        }
        shape.push(value);
        Self::check_deadline(deadline, site)?;
        Ok(shape)
    }

    fn linear_into_batched_preserving_certificate(
        lb: LinearBounds,
        deadline: Option<Instant>,
    ) -> Result<BatchedLinearBounds> {
        let out_dim = lb.num_outputs();
        let in_dim = lb.num_inputs();
        let LinearBounds {
            lower_a,
            lower_b,
            upper_a,
            upper_b,
            lower_a_err,
            upper_a_err,
        } = lb;
        Ok(BatchedLinearBounds {
            lower_a: lower_a.into_dyn(),
            lower_b: lower_b.into_dyn(),
            upper_a: upper_a.into_dyn(),
            upper_b: upper_b.into_dyn(),
            input_shape: Self::singleton_shape_with_deadline(
                in_dim,
                deadline,
                "finite batched input-shape allocation",
            )?,
            output_shape: Self::singleton_shape_with_deadline(
                out_dim,
                deadline,
                "finite batched output-shape allocation",
            )?,
            lower_a_err: lower_a_err.map(|array| array.into_dyn()),
            upper_a_err: upper_a_err.map(|array| array.into_dyn()),
        })
    }

    /// Move a `LinearBounds` into batched storage without duplicating any of
    /// its four admitted dense arrays.
    fn linear_into_batched(mut lb: LinearBounds) -> BatchedLinearBounds {
        // The move-only unchecked batched constructor cannot receive the
        // scalar certificate. Never silently drop it: conservatively discharge
        // affected rows in-place before destructuring, with no full-size clone.
        lb.discharge_coeff_err_to_conservative();
        let out_dim = lb.num_outputs();
        let in_dim = lb.num_inputs();
        let (lower_a, lower_b, upper_a, upper_b) = lb.into_parts();
        // KEEP unchecked: LinearBounds already validated these arrays; into_dyn()
        // only changes the view rank for batched storage.
        BatchedLinearBounds::from_parts_unchecked(
            lower_a.into_dyn(),
            lower_b.into_dyn(),
            upper_a.into_dyn(),
            upper_b.into_dyn(),
            vec![in_dim],
            vec![out_dim],
        )
    }

    #[cfg(test)]
    fn placeholder_dense() -> BatchedLinearBounds {
        // The placeholder is confined to already-dense, infallible conversion
        // and merge branches. Patches never replaces its source before every
        // fallible materialization/admission step has succeeded.
        BatchedLinearBounds::from_parts_unchecked(
            ArrayD::zeros(IxDyn(&[0, 0])),
            ArrayD::zeros(IxDyn(&[0])),
            ArrayD::zeros(IxDyn(&[0, 0])),
            ArrayD::zeros(IxDyn(&[0])),
            vec![0],
            vec![0],
        )
    }

    /// Convert an unbatched `CrownBounds` to `BatchedCrownBounds`.
    ///
    /// Used when the Patches backward path produces a result that needs to
    /// be stored in the batched bounds map. Preserves the variant:
    /// - `CrownBounds::Dense` → `BatchedCrownBounds::Dense`
    /// - `CrownBounds::Patches` → `BatchedCrownBounds::Patches`
    pub(crate) fn from_crown_bounds(cb: CrownBounds) -> Result<BatchedCrownBounds> {
        Self::from_crown_bounds_with_deadline(cb, None)
    }

    /// Deadline-aware move from the scalar carrier into batched storage.
    pub(crate) fn from_crown_bounds_with_deadline(
        cb: CrownBounds,
        deadline: Option<Instant>,
    ) -> Result<BatchedCrownBounds> {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Err(NyError::DeadlineExceeded(
                "batched CROWN carrier conversion: deadline exceeded before move".into(),
            ));
        }
        match cb {
            CrownBounds::Dense(lb) => {
                let dense = if deadline.is_some() {
                    Self::linear_into_batched_preserving_certificate(lb, deadline)?
                } else {
                    Self::linear_into_batched(lb)
                };
                let batched = BatchedCrownBounds::Dense(dense);
                if deadline.is_some_and(|limit| Instant::now() >= limit) {
                    return Err(NyError::DeadlineExceeded(
                        "batched CROWN carrier conversion: deadline exceeded before publication"
                            .into(),
                    ));
                }
                Ok(batched)
            }
            CrownBounds::Patches(pb) => Ok(BatchedCrownBounds::Patches(pb)),
        }
    }

    /// Total heap memory used by the current bounds representation, in bytes.
    ///
    /// Dispatches to the appropriate variant's memory tracking.
    pub(crate) fn memory_bytes(&self) -> usize {
        match self {
            BatchedCrownBounds::Dense(blb) => blb.memory_bytes(),
            BatchedCrownBounds::Dense64(blb) => blb.memory_bytes(),
            BatchedCrownBounds::FiniteDense64(blb) => blb.memory_bytes(),
            BatchedCrownBounds::Patches(plb) => plb.memory_bytes(),
        }
    }

    /// Whether this is currently in Patches mode.
    pub(crate) fn is_patches(&self) -> bool {
        matches!(self, BatchedCrownBounds::Patches(_))
    }

    /// Merge another dense contribution into this bounds entry, promoting the
    /// dense payload to an f64 accumulator on the first merge.
    #[cfg(test)]
    pub(crate) fn merge_dense_checked(
        &mut self,
        new_bounds: BatchedLinearBounds,
        site: &'static str,
    ) -> Result<()> {
        self.merge_dense_checked_with_deadline(new_bounds, site, None)
    }

    /// Deadline-aware merge. Any required Patches materialization completes
    /// transactionally before the existing entry is promoted and accumulated.
    pub(crate) fn merge_dense_checked_with_deadline(
        &mut self,
        new_bounds: BatchedLinearBounds,
        site: &'static str,
        deadline: Option<Instant>,
    ) -> Result<()> {
        Self::check_deadline(deadline, site)?;

        let merged = if deadline.is_some() {
            let mut authority = PatchesMaterializationDeadline::new(deadline);
            match self {
                BatchedCrownBounds::Patches(patches) => {
                    let (rows, columns) = patches.dense_pair_shape()?;
                    authority.checkpoint("before finite batched Patches merge admission")?;
                    let mut admission = Self::guard_patches_sidecar_staging(
                        patches,
                        &new_bounds,
                        rows,
                        columns,
                        false,
                    )?;
                    // Keep the exact structured source installed until both the
                    // dense materialization and pollable f64 fan-in are complete.
                    let dense = Self::patches_to_batched_linear_with_resident(
                        patches,
                        deadline,
                        Self::batched_dense_retained_bytes(&new_bounds),
                    )?;
                    authority.checkpoint("before finite batched Patches sidecar staging")?;
                    let mut accumulator = FiniteBatchedLinearBounds64::from_f32(
                        &dense,
                        &mut admission,
                        &mut authority,
                    )?;
                    accumulator.accumulate(&new_bounds, &mut admission, &mut authority)?;
                    Some(BatchedCrownBounds::FiniteDense64(accumulator))
                }
                BatchedCrownBounds::Dense(existing) => {
                    let mut admission =
                        Self::guard_dense_sidecar_staging(existing, &new_bounds, false)?;
                    authority.checkpoint("before finite batched Dense sidecar staging")?;
                    let mut accumulator = FiniteBatchedLinearBounds64::from_f32(
                        existing,
                        &mut admission,
                        &mut authority,
                    )?;
                    accumulator.accumulate(&new_bounds, &mut admission, &mut authority)?;
                    Some(BatchedCrownBounds::FiniteDense64(accumulator))
                }
                BatchedCrownBounds::FiniteDense64(existing) => {
                    let mut admission =
                        Self::guard_existing_finite_sidecar_staging(existing, &new_bounds)?;
                    authority.checkpoint("before finite batched sidecar transaction")?;
                    let mut staged = existing.staged_copy(&mut admission, &mut authority)?;
                    staged.accumulate(&new_bounds, &mut admission, &mut authority)?;
                    Some(BatchedCrownBounds::FiniteDense64(staged))
                }
                BatchedCrownBounds::Dense64(_) => {
                    return Err(NyError::UnsupportedConfiguration(
                        "finite batched merge encountered a legacy opaque f64 sidecar".into(),
                    ));
                }
            }
        } else {
            match self {
                BatchedCrownBounds::Patches(patches) => {
                    let (rows, columns) = patches.dense_pair_shape()?;
                    let _admission = Self::guard_patches_sidecar_staging(
                        patches,
                        &new_bounds,
                        rows,
                        columns,
                        true,
                    )?;
                    let dense = Self::patches_to_batched_linear(patches, None)?;
                    let mut accumulator = BatchedLinearBounds64::from_f32(&dense);
                    accumulator.accumulate(&new_bounds);
                    Some(BatchedCrownBounds::Dense64(accumulator))
                }
                BatchedCrownBounds::Dense(existing) => {
                    let mut accumulator = BatchedLinearBounds64::from_f32(existing);
                    accumulator.accumulate(&new_bounds);
                    Some(BatchedCrownBounds::Dense64(accumulator))
                }
                BatchedCrownBounds::Dense64(existing) => {
                    existing.accumulate(&new_bounds);
                    None
                }
                BatchedCrownBounds::FiniteDense64(existing) => {
                    let mut admission =
                        Self::guard_existing_finite_sidecar_staging(existing, &new_bounds)?;
                    let mut authority = PatchesMaterializationDeadline::new(None);
                    let mut staged = existing.staged_copy(&mut admission, &mut authority)?;
                    staged.accumulate(&new_bounds, &mut admission, &mut authority)?;
                    Some(BatchedCrownBounds::FiniteDense64(staged))
                }
            }
        };

        Self::check_deadline(deadline, site)?;
        if let Some(merged) = merged {
            *self = merged;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2, ArrayD, IxDyn};
    use ny_tensor::{next_down_f32, next_up_f32};
    use std::time::Duration;

    fn assert_same_patches(actual: &PatchesLinearBounds, expected: &PatchesLinearBounds) {
        assert_eq!(actual.row_count, expected.row_count);
        assert_eq!(actual.lower_a.coeff_err, expected.lower_a.coeff_err);
        assert_eq!(actual.lower_a.patches, expected.lower_a.patches);
        assert_eq!(actual.lower_a.geometry, expected.lower_a.geometry);
        assert_eq!(actual.lower_a.identity, expected.lower_a.identity);
        assert_eq!(actual.lower_a.output_shape, expected.lower_a.output_shape);
        assert_eq!(actual.lower_a.input_shape, expected.lower_a.input_shape);
        assert_eq!(actual.lower_a.unstable_idx, expected.lower_a.unstable_idx);
        assert_eq!(actual.lower_b, expected.lower_b);
        assert_eq!(actual.upper_a.coeff_err, expected.upper_a.coeff_err);
        assert_eq!(actual.upper_a.patches, expected.upper_a.patches);
        assert_eq!(actual.upper_a.geometry, expected.upper_a.geometry);
        assert_eq!(actual.upper_a.identity, expected.upper_a.identity);
        assert_eq!(actual.upper_a.output_shape, expected.upper_a.output_shape);
        assert_eq!(actual.upper_a.input_shape, expected.upper_a.input_shape);
        assert_eq!(actual.upper_a.unstable_idx, expected.upper_a.unstable_idx);
        assert_eq!(actual.upper_b, expected.upper_b);
    }

    fn scalar_batched_linear_bounds(value: f32) -> BatchedLinearBounds {
        BatchedLinearBounds::from_parts_unchecked(
            ArrayD::from_elem(IxDyn(&[1, 1, 1]), value),
            ArrayD::from_elem(IxDyn(&[1, 1]), value),
            ArrayD::from_elem(IxDyn(&[1, 1, 1]), value),
            ArrayD::from_elem(IxDyn(&[1, 1]), value),
            vec![1, 1],
            vec![1, 1],
        )
    }

    #[test]
    fn test_batched_crown_bounds_into_dense_passthrough() -> Result<()> {
        let blb = BatchedLinearBounds::from_parts_unchecked(
            ArrayD::zeros(IxDyn(&[4, 4])),
            ArrayD::zeros(IxDyn(&[4])),
            ArrayD::zeros(IxDyn(&[4, 4])),
            ArrayD::zeros(IxDyn(&[4])),
            vec![4],
            vec![4],
        );
        let bcb = BatchedCrownBounds::Dense(blb);
        let result = bcb.into_batched_dense()?;
        assert_eq!(result.lower_a().shape(), &[4, 4]);
        Ok(())
    }

    #[test]
    fn test_batched_crown_bounds_patches_to_dense() -> Result<()> {
        let shape = (1, 2, 2); // 1 channel, 2x2
        let dim = 4;
        let plb = PatchesLinearBounds::identity(shape, shape);
        let bcb = BatchedCrownBounds::Patches(Box::new(plb));
        let result = bcb.into_batched_dense()?;
        // Should produce [4, 4] identity in BatchedLinearBounds
        assert_eq!(result.lower_a().shape(), &[dim, dim]);
        assert_eq!(result.upper_a().shape(), &[dim, dim]);
        assert_eq!(result.lower_b().shape(), &[dim]);
        // Verify identity: diagonal entries = 1.0
        for i in 0..dim {
            assert_eq!(result.lower_a()[[i, i]], 1.0);
            assert_eq!(result.upper_a()[[i, i]], 1.0);
        }
        Ok(())
    }

    #[test]
    fn test_batched_crown_bounds_ensure_dense() -> Result<()> {
        let shape = (1, 2, 2);
        let plb = PatchesLinearBounds::identity(shape, shape);
        let mut bcb = BatchedCrownBounds::Patches(Box::new(plb));
        let blb = bcb.ensure_batched_dense()?;
        assert_eq!(blb.lower_a().shape(), &[4, 4]);
        // Should be Dense variant after ensure
        assert!(matches!(bcb, BatchedCrownBounds::Dense(_)));
        Ok(())
    }

    #[test]
    fn linear_to_batched_conversion_moves_all_dense_allocations() -> Result<()> {
        let lb = LinearBounds::new(
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .expect("lower fixture shape"),
            Array1::from_vec(vec![7.0, 8.0]),
            Array2::from_shape_vec((2, 3), vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
                .expect("upper fixture shape"),
            Array1::from_vec(vec![15.0, 16.0]),
        )?;
        let pointers = (
            lb.lower_a().as_ptr(),
            lb.lower_b().as_ptr(),
            lb.upper_a().as_ptr(),
            lb.upper_b().as_ptr(),
        );

        let batched = BatchedCrownBounds::linear_into_batched(lb);
        assert_eq!(batched.lower_a().as_ptr(), pointers.0);
        assert_eq!(batched.lower_b().as_ptr(), pointers.1);
        assert_eq!(batched.upper_a().as_ptr(), pointers.2);
        assert_eq!(batched.upper_b().as_ptr(), pointers.3);
        Ok(())
    }

    #[test]
    fn linear_to_batched_conversion_never_drops_scalar_coefficient_error() {
        let mut lb = LinearBounds::identity(2);
        lb.set_coeff_err(
            array![[0.25, 0.0], [0.0, 0.0]],
            array![[0.5, 0.0], [0.0, 0.0]],
        );

        let batched = BatchedCrownBounds::linear_into_batched(lb);

        assert_eq!(batched.lower_a()[[0, 0]], 0.0);
        assert_eq!(batched.lower_a()[[0, 1]], 0.0);
        assert_eq!(batched.lower_b()[[0]], f32::NEG_INFINITY);
        assert_eq!(batched.upper_a()[[0, 0]], 0.0);
        assert_eq!(batched.upper_a()[[0, 1]], 0.0);
        assert_eq!(batched.upper_b()[[0]], f32::INFINITY);
        assert_eq!(batched.lower_a()[[1, 1]], 1.0);
        assert_eq!(batched.upper_a()[[1, 1]], 1.0);
        assert!(!batched.has_coeff_err());
    }

    #[test]
    fn expired_batched_patches_ensure_is_transactional() {
        let patches = PatchesLinearBounds::identity((1, 2, 2), (1, 2, 2));
        let expected = patches.clone();
        let mut bounds = BatchedCrownBounds::Patches(Box::new(patches));
        let expired = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("one-second deadline subtraction");

        let error = bounds
            .ensure_batched_dense_checked_with_deadline(
                "test:expired_batched_patches_ensure_is_transactional",
                Some(expired),
            )
            .expect_err("expired materialization must be terminal");
        assert!(error.is_deadline_exceeded());
        let BatchedCrownBounds::Patches(actual) = &bounds else {
            panic!("deadline refusal replaced the Patches carrier")
        };
        assert_same_patches(actual, &expected);
    }

    #[test]
    fn live_deadline_batched_patches_materialization_matches_legacy() -> Result<()> {
        let shape = (1, 2, 2);
        let legacy =
            BatchedCrownBounds::Patches(Box::new(PatchesLinearBounds::identity(shape, shape)))
                .into_batched_dense()?;
        let bounded =
            BatchedCrownBounds::Patches(Box::new(PatchesLinearBounds::identity(shape, shape)))
                .into_batched_dense_checked_with_deadline(
                    "test:live_deadline_batched_patches_materialization_matches_legacy",
                    Some(Instant::now() + Duration::from_secs(30)),
                )?;

        assert_eq!(bounded.lower_a(), legacy.lower_a());
        assert_eq!(bounded.lower_b(), legacy.lower_b());
        assert_eq!(bounded.upper_a(), legacy.upper_a());
        assert_eq!(bounded.upper_b(), legacy.upper_b());
        assert_eq!(bounded.input_shape(), legacy.input_shape());
        assert_eq!(bounded.output_shape(), legacy.output_shape());
        Ok(())
    }

    #[test]
    fn batched_patches_merge_memory_refusal_is_atomic() {
        crate::tests::with_env_edits(|env| {
            env.set("NY_DENSE_BUDGET_MB", "0");
            let patches = PatchesLinearBounds::identity((1, 1, 1), (1, 1, 1));
            let expected = patches.clone();
            let mut carrier = BatchedCrownBounds::Patches(Box::new(patches));

            let error = carrier
                .merge_dense_checked_with_deadline(
                    scalar_batched_linear_bounds(1.0),
                    "test:batched_patches_merge_memory_refusal_is_atomic",
                    Some(Instant::now() + Duration::from_secs(30)),
                )
                .expect_err("zero total-live budget must refuse the staged sidecar");

            assert!(matches!(error, NyError::CpuMemoryExceeded { .. }));
            let BatchedCrownBounds::Patches(actual) = &carrier else {
                panic!("memory refusal replaced the pending Patches carrier")
            };
            assert_same_patches(actual, &expected);
        });
    }

    #[test]
    fn expired_batched_patches_merge_is_atomic() {
        let patches = PatchesLinearBounds::identity((1, 1, 1), (1, 1, 1));
        let expected = patches.clone();
        let mut carrier = BatchedCrownBounds::Patches(Box::new(patches));
        let expired = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("one-second deadline subtraction");

        let error = carrier
            .merge_dense_checked_with_deadline(
                scalar_batched_linear_bounds(1.0),
                "test:expired_batched_patches_merge_is_atomic",
                Some(expired),
            )
            .expect_err("expired sidecar staging must be terminal");

        assert!(error.is_deadline_exceeded());
        let BatchedCrownBounds::Patches(actual) = &carrier else {
            panic!("deadline refusal replaced the pending Patches carrier")
        };
        assert_same_patches(actual, &expected);
    }

    #[test]
    fn legacy_opaque_sidecar_is_typed_closed_under_finite_authority() {
        let original = scalar_batched_linear_bounds(2.0);
        let mut carrier = BatchedCrownBounds::Dense64(BatchedLinearBounds64::from_f32(&original));

        let error = carrier
            .ensure_batched_dense_checked_with_deadline(
                "test:legacy_opaque_sidecar_is_typed_closed_under_finite_authority",
                Some(Instant::now() + Duration::from_secs(30)),
            )
            .expect_err("finite authority must not enter the legacy opaque downcast");

        assert!(matches!(error, NyError::UnsupportedConfiguration(_)));
        assert!(matches!(&carrier, BatchedCrownBounds::Dense64(_)));
        let recovered = carrier
            .into_batched_dense()
            .expect("typed refusal must retain a recoverable legacy sidecar");
        assert_eq!(recovered.lower_a().shape(), original.lower_a().shape());
        assert_eq!(recovered.upper_a().shape(), original.upper_a().shape());
        assert!(recovered.lower_a()[[0, 0, 0]] <= original.lower_a()[[0, 0, 0]]);
        assert!(recovered.upper_a()[[0, 0, 0]] >= original.upper_a()[[0, 0, 0]]);
    }

    #[test]
    fn finite_sidecar_merge_and_downcast_match_legacy() -> Result<()> {
        crate::tests::with_env_edits(|_env| {
            let contributions = [1.25_f32, -0.5_f32, 3.0_f32];
            let mut legacy =
                BatchedCrownBounds::Dense(scalar_batched_linear_bounds(contributions[0]));
            let mut finite =
                BatchedCrownBounds::Dense(scalar_batched_linear_bounds(contributions[0]));
            let deadline = Some(Instant::now() + Duration::from_secs(30));

            for &value in &contributions[1..] {
                legacy.merge_dense_checked(
                    scalar_batched_linear_bounds(value),
                    "test:finite_sidecar_merge_and_downcast_match_legacy",
                )?;
                finite.merge_dense_checked_with_deadline(
                    scalar_batched_linear_bounds(value),
                    "test:finite_sidecar_merge_and_downcast_match_legacy",
                    deadline,
                )?;
            }
            assert!(matches!(&finite, BatchedCrownBounds::FiniteDense64(_)));
            let legacy = legacy.into_batched_dense()?;
            let finite = finite.into_batched_dense_checked_with_deadline(
                "test:finite_sidecar_merge_and_downcast_match_legacy",
                deadline,
            )?;

            assert_eq!(finite.lower_a(), legacy.lower_a());
            assert_eq!(finite.lower_b(), legacy.lower_b());
            assert_eq!(finite.upper_a(), legacy.upper_a());
            assert_eq!(finite.upper_b(), legacy.upper_b());
            assert_eq!(finite.input_shape(), legacy.input_shape());
            assert_eq!(finite.output_shape(), legacy.output_shape());
            Ok(())
        })
    }

    #[test]
    fn test_batched_crown_bounds_merge_promotes_dense64_until_materialization_3904() -> Result<()> {
        let contributions = [1_099_511_627_776.0_f32, 1.0_f32, -1_099_511_627_776.0_f32];
        let mut bcb = BatchedCrownBounds::Dense(scalar_batched_linear_bounds(contributions[0]));

        bcb.merge_dense_checked(
            scalar_batched_linear_bounds(contributions[1]),
            "test_batched_crown_bounds_merge_promotes_dense64_until_materialization_3904:first",
        )?;
        assert!(
            matches!(bcb, BatchedCrownBounds::Dense64(_)),
            "first merge should promote the dense payload to the f64 accumulator"
        );

        bcb.merge_dense_checked(
            scalar_batched_linear_bounds(contributions[2]),
            "test_batched_crown_bounds_merge_promotes_dense64_until_materialization_3904:second",
        )?;
        assert!(
            matches!(bcb, BatchedCrownBounds::Dense64(_)),
            "subsequent merges should keep the f64 accumulator alive until materialization"
        );

        let merged = bcb.into_batched_dense()?;
        assert_eq!(merged.lower_a()[[0, 0, 0]], next_down_f32(1.0));
        assert_eq!(merged.lower_b()[[0, 0]], next_down_f32(1.0));
        assert_eq!(merged.upper_a()[[0, 0, 0]], next_up_f32(1.0));
        assert_eq!(merged.upper_b()[[0, 0]], next_up_f32(1.0));
        Ok(())
    }

    #[test]
    fn batched_f64_merge_never_drops_coefficient_error() -> Result<()> {
        let mut existing = scalar_batched_linear_bounds(1.0);
        existing.set_coeff_err(
            ArrayD::from_elem(IxDyn(&[1, 1, 1]), 0.25),
            ArrayD::from_elem(IxDyn(&[1, 1, 1]), 0.5),
        );
        let mut carrier = BatchedCrownBounds::Dense(existing);

        carrier.merge_dense_checked(
            scalar_batched_linear_bounds(0.0),
            "test:batched_f64_merge_never_drops_coefficient_error",
        )?;
        let merged = carrier.into_batched_dense()?;

        assert!(merged.lower_b()[[0, 0]].is_finite());
        assert!(merged.upper_b()[[0, 0]].is_finite());
        assert!(merged.has_coeff_err());
        assert!(merged.lower_a_err.as_ref().expect("lower certificate")[[0, 0, 0]] >= 0.25);
        assert!(merged.upper_a_err.as_ref().expect("upper certificate")[[0, 0, 0]] >= 0.5);
        Ok(())
    }

    #[test]
    fn finite_f64_merge_never_drops_coefficient_error() -> Result<()> {
        crate::tests::with_env_edits(|_env| {
            let mut existing = scalar_batched_linear_bounds(1.0);
            existing.set_coeff_err(
                ArrayD::from_elem(IxDyn(&[1, 1, 1]), 0.25),
                ArrayD::from_elem(IxDyn(&[1, 1, 1]), 0.5),
            );
            let mut carrier = BatchedCrownBounds::Dense(existing);
            let deadline = Some(Instant::now() + Duration::from_secs(30));

            carrier.merge_dense_checked_with_deadline(
                scalar_batched_linear_bounds(0.0),
                "test:finite_f64_merge_never_drops_coefficient_error",
                deadline,
            )?;
            let merged = carrier.into_batched_dense_checked_with_deadline(
                "test:finite_f64_merge_never_drops_coefficient_error",
                deadline,
            )?;

            assert_eq!(merged.lower_b()[[0, 0]], f32::NEG_INFINITY);
            assert_eq!(merged.upper_b()[[0, 0]], f32::INFINITY);
            assert!(!merged.has_coeff_err());
            Ok(())
        })
    }
}
