// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! N-ary concatenation propagation (3+ inputs).
//!
//! Split from `concat/mod.rs` for file size compliance.

use std::borrow::Cow;

use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::LinearBounds;

use super::ConcatLayer;

impl ConcatLayer {
    fn prepare_nary_ibp_inputs<'a>(
        &self,
        inputs: &'a [&'a BoundedTensor],
    ) -> Result<(Vec<Cow<'a, BoundedTensor>>, usize, bool)> {
        let first_shape = inputs[0].shape();
        let mut max_ndim = first_shape.len();
        let mut min_ndim = first_shape.len();

        for input in inputs.iter().skip(1) {
            let ndim = input.shape().len();
            max_ndim = max_ndim.max(ndim);
            min_ndim = min_ndim.min(ndim);
        }

        if max_ndim - min_ndim > 1 {
            let mismatch_shape = inputs
                .iter()
                .find(|input| input.shape().len() != first_shape.len())
                .map(|input| input.shape().to_vec())
                .unwrap_or_else(|| first_shape.to_vec());
            return Err(NyError::ShapeMismatch {
                expected: first_shape.to_vec(),
                got: mismatch_shape,
            });
        }

        let mut batch_size = None;
        if max_ndim != min_ndim {
            for input in inputs {
                if input.shape().len() == max_ndim {
                    batch_size = Some(input.shape()[0]);
                    break;
                }
            }
        }

        // Match alpha-beta-CROWN's batch-first concat semantics
        // (`auto_LiRPA/operators/slice_concat.py`, `BoundConcat.interval_propagate`):
        // if ny drops a leading size-1 batch dimension on some branches, reinsert
        // it before validating shape compatibility.
        let prepared = inputs
            .iter()
            .map(|input| match batch_size {
                Some(batch_size) if input.shape().len() < max_ndim => {
                    let lower = Self::broadcast_to_batch(input.lower(), batch_size)?;
                    let upper = Self::broadcast_to_batch(input.upper(), batch_size)?;
                    // Pure layout op (batch broadcast): value-preserving, so
                    // infinite bounds pass through soundly. NaN is still rejected.
                    Ok(Cow::Owned(BoundedTensor::new_allow_infinite(lower, upper)?))
                }
                _ => Ok(Cow::Borrowed(*input)),
            })
            .collect::<Result<Vec<_>>>()?;

        Ok((prepared, max_ndim, batch_size.is_some()))
    }

    fn validate_nary_ibp_shapes(
        &self,
        inputs: &[Cow<'_, BoundedTensor>],
        ndim: usize,
        restored_batch: bool,
    ) -> Result<usize> {
        let axis = self.normalize_axis_with_restored_batch(ndim, restored_batch)?;
        let first_shape = inputs[0].shape();

        for input in inputs.iter().skip(1) {
            let shape = input.shape();
            for (d, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if d != axis && s1 != s2 {
                    return Err(NyError::ShapeMismatch {
                        expected: first_shape.to_vec(),
                        got: shape.to_vec(),
                    });
                }
            }
        }

        Ok(axis)
    }

    /// Propagate IBP bounds through N-ary concatenation.
    ///
    /// For Y = concat(A, B, C, ...) along axis:
    /// Y_lower = concat(A_lower, B_lower, C_lower, ...)
    /// Y_upper = concat(A_upper, B_upper, C_upper, ...)
    pub fn propagate_ibp_nary(&self, inputs: &[&BoundedTensor]) -> Result<BoundedTensor> {
        if inputs.is_empty() {
            return Err(NyError::InvalidSpec(
                "Concat requires at least one input".to_string(),
            ));
        }
        if inputs.len() == 1 {
            return Ok(inputs[0].clone());
        }

        let (prepared_inputs, max_ndim, restored_batch) = self.prepare_nary_ibp_inputs(inputs)?;
        let axis = self.validate_nary_ibp_shapes(&prepared_inputs, max_ndim, restored_batch)?;

        // Collect all lower and upper bound views for concatenation
        let lower_views: Vec<_> = prepared_inputs.iter().map(|b| b.lower().view()).collect();
        let upper_views: Vec<_> = prepared_inputs.iter().map(|b| b.upper().view()).collect();

        // Concatenate
        let out_lower = ndarray::concatenate(ndarray::Axis(axis), &lower_views)
            .map_err(|e| NyError::InvalidSpec(format!("Concat lower bounds failed: {}", e)))?;
        let out_upper = ndarray::concatenate(ndarray::Axis(axis), &upper_views)
            .map_err(|e| NyError::InvalidSpec(format!("Concat upper bounds failed: {}", e)))?;

        // Pure layout op (concatenation): value-preserving, so infinite bounds
        // pass through soundly. Allow `±inf` to flow without tripping the NaN
        // firewall; NaN is still rejected.
        BoundedTensor::new_allow_infinite(out_lower, out_upper)
    }

    /// CROWN backward propagation for N-ary Concat (Y = concat(A, B, C, ...)).
    ///
    /// Splits the coefficient matrix into N parts based on input sizes.
    /// Returns a vector of LinearBounds, one for each input.
    pub fn propagate_linear_nary(
        &self,
        bounds: &LinearBounds,
        input_shapes: &[Vec<usize>],
    ) -> Result<Vec<LinearBounds>> {
        if input_shapes.is_empty() {
            return Err(NyError::InvalidSpec(
                "Concat requires at least one input".to_string(),
            ));
        }

        let sizes: Vec<usize> = input_shapes
            .iter()
            .enumerate()
            .map(|(i, s)| {
                checked_shape_product(s).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "Concat N-ary: input[{}] shape product overflows usize: {:?}",
                        i, s,
                    ))
                })
            })
            .collect::<Result<Vec<usize>>>()?;
        let total_size: usize = sizes
            .iter()
            .try_fold(0usize, |acc, &s| acc.checked_add(s))
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Concat N-ary: sum of {} input sizes overflows usize",
                    sizes.len(),
                ))
            })?;

        if bounds.num_inputs() != total_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![total_size],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_inputs = input_shapes.len();
        let mut result = Vec::with_capacity(num_inputs);

        // Split bias evenly among all inputs.
        // Use directed rounding so re-accumulated bias remains conservative.
        let bias_divisor = num_inputs as f32;
        let lower_b_part = bounds.lower_b().mapv(|v| next_down_f32(v / bias_divisor));
        let upper_b_part = bounds.upper_b().mapv(|v| next_up_f32(v / bias_divisor));

        // Split coefficient matrices based on cumulative sizes
        let mut offset = 0;
        for &size in &sizes {
            let lower_a_part = bounds
                .lower_a()
                .slice(ndarray::s![.., offset..offset + size])
                .to_owned();
            let upper_a_part = bounds
                .upper_a()
                .slice(ndarray::s![.., offset..offset + size])
                .to_owned();

            result.push(LinearBounds::new_or_conservative(
                lower_a_part,
                lower_b_part.clone(),
                upper_a_part,
                upper_b_part.clone(),
            )?);

            offset += size;
        }

        Ok(result)
    }
}
