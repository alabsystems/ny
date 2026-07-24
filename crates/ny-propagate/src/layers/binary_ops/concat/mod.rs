// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::{BatchedLinearBounds, LinearBounds};

/// Concatenation layer: concatenates two tensors along a specified axis.
///
/// This is used for operations like concatenating CLS token with patch embeddings in ViT,
/// or combining tensors in attention mechanisms. For IBP, this is straightforward:
/// concat(lower_a, lower_b) and concat(upper_a, upper_b).
///
/// For CROWN backward propagation, we split the coefficient matrix back to each input.
#[derive(Debug, Clone)]
pub struct ConcatLayer {
    /// The axis along which to concatenate (negative indices supported).
    pub axis: i64,
    /// Whether `axis` was derived from a positive ONNX axis after squeezing a
    /// leading batch dimension. When IBP temporarily restores that batch
    /// dimension, the effective concat axis must shift by +1.
    restored_batch_axis_shift: bool,
    /// Optional stored input shapes for CROWN backward when inputs are constant tensors.
    /// This is used when one or more inputs come from ConstantOfShape and aren't in node_bounds.
    pub input_shapes: Option<Vec<Vec<usize>>>,
    /// Optional constant tensors for inputs that are known at graph construction time.
    /// Each element is Some(tensor) for constant inputs, None for dynamic inputs.
    /// Used during IBP forward when the constant isn't in node_bounds cache.
    pub constant_inputs: Option<Vec<Option<BoundedTensor>>>,
}

impl ConcatLayer {
    /// Create a new concatenation layer.
    pub fn new(axis: i64) -> Self {
        Self {
            axis,
            restored_batch_axis_shift: false,
            input_shapes: None,
            constant_inputs: None,
        }
    }

    /// Create a new concatenation layer with known input shapes.
    pub fn with_input_shapes(axis: i64, input_shapes: Vec<Vec<usize>>) -> Self {
        Self {
            axis,
            restored_batch_axis_shift: false,
            input_shapes: Some(input_shapes),
            constant_inputs: None,
        }
    }

    /// Create a new concatenation layer with constant input tensors.
    pub fn with_constants(
        axis: i64,
        input_shapes: Vec<Vec<usize>>,
        constant_inputs: Vec<Option<BoundedTensor>>,
    ) -> Self {
        Self {
            axis,
            restored_batch_axis_shift: false,
            input_shapes: Some(input_shapes),
            constant_inputs: Some(constant_inputs),
        }
    }

    /// Mark that the stored axis came from a positive ONNX axis after
    /// squeezing a leading batch dimension.
    #[must_use]
    pub fn with_restored_batch_axis_shift(mut self, restored_batch_axis_shift: bool) -> Self {
        self.restored_batch_axis_shift = restored_batch_axis_shift;
        self
    }

    /// The stored shape for input at given index, if available.
    pub fn input_shape(&self, index: usize) -> Option<&[usize]> {
        self.input_shapes
            .as_ref()
            .and_then(|shapes| shapes.get(index))
            .map(|v| v.as_slice())
    }

    /// The constant BoundedTensor for input at given index, if available.
    pub fn constant_input(&self, index: usize) -> Option<&BoundedTensor> {
        self.constant_inputs
            .as_ref()
            .and_then(|inputs| inputs.get(index))
            .and_then(|opt| opt.as_ref())
    }

    /// Normalize axis to positive index given the number of dimensions.
    pub(crate) fn normalize_axis(&self, ndim: usize) -> Result<usize> {
        crate::layers::common::resolve_axis(self.axis, ndim, "Concat")
    }

    /// Normalize axis when a leading batch dimension was reintroduced for
    /// broadcasted inputs. Only axes known to have been shifted during ONNX
    /// conversion should move by +1 here.
    pub(super) fn normalize_axis_with_restored_batch(
        &self,
        ndim: usize,
        restored_batch: bool,
    ) -> Result<usize> {
        if restored_batch && self.restored_batch_axis_shift {
            let squeezed_ndim = ndim.checked_sub(1).ok_or_else(|| {
                NyError::InvalidSpec(
                    "Concat restored batch dimension requires ndim >= 1".to_string(),
                )
            })?;
            Ok(self.normalize_axis(squeezed_ndim)? + 1)
        } else {
            self.normalize_axis(ndim)
        }
    }

    /// Propagate IBP bounds through concatenation.
    ///
    /// For Y = concat(A, B) along axis:
    /// Y_lower = concat(A_lower, B_lower)
    /// Y_upper = concat(A_upper, B_upper)
    ///
    /// When one input has fewer dimensions (e.g., constant without batch),
    /// it will be broadcast to match the batch dimension of the other input.
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let ndim_a = input_a.shape().len();
        let ndim_b = input_b.shape().len();

        // Handle broadcasting when one input has fewer dimensions (e.g., constant without batch)
        let (a_lower, a_upper, b_lower, b_upper, effective_ndim, restored_batch) =
            if ndim_a != ndim_b {
                // One input needs broadcasting to add batch dimension
                if ndim_a + 1 == ndim_b {
                    // Input A is missing batch dimension - broadcast it
                    let batch_size = input_b.shape()[0];
                    let a_lower_expanded = Self::broadcast_to_batch(input_a.lower(), batch_size)?;
                    let a_upper_expanded = Self::broadcast_to_batch(input_a.upper(), batch_size)?;
                    (
                        std::borrow::Cow::Owned(a_lower_expanded),
                        std::borrow::Cow::Owned(a_upper_expanded),
                        std::borrow::Cow::Borrowed(input_b.lower()),
                        std::borrow::Cow::Borrowed(input_b.upper()),
                        ndim_b,
                        true,
                    )
                } else if ndim_b + 1 == ndim_a {
                    // Input B is missing batch dimension - broadcast it
                    let batch_size = input_a.shape()[0];
                    let b_lower_expanded = Self::broadcast_to_batch(input_b.lower(), batch_size)?;
                    let b_upper_expanded = Self::broadcast_to_batch(input_b.upper(), batch_size)?;
                    (
                        std::borrow::Cow::Borrowed(input_a.lower()),
                        std::borrow::Cow::Borrowed(input_a.upper()),
                        std::borrow::Cow::Owned(b_lower_expanded),
                        std::borrow::Cow::Owned(b_upper_expanded),
                        ndim_a,
                        true,
                    )
                } else {
                    return Err(NyError::ShapeMismatch {
                        expected: input_a.shape().to_vec(),
                        got: input_b.shape().to_vec(),
                    });
                }
            } else {
                (
                    std::borrow::Cow::Borrowed(input_a.lower()),
                    std::borrow::Cow::Borrowed(input_a.upper()),
                    std::borrow::Cow::Borrowed(input_b.lower()),
                    std::borrow::Cow::Borrowed(input_b.upper()),
                    ndim_a,
                    false,
                )
            };

        let axis = self.normalize_axis_with_restored_batch(effective_ndim, restored_batch)?;

        // Check that all dimensions except axis match
        for (i, (&da, &db)) in a_lower
            .shape()
            .iter()
            .zip(b_lower.shape().iter())
            .enumerate()
        {
            if i != axis && da != db {
                return Err(NyError::ShapeMismatch {
                    expected: a_lower.shape().to_vec(),
                    got: b_lower.shape().to_vec(),
                });
            }
        }

        // Concatenate lower and upper bounds
        let out_lower =
            ndarray::concatenate(ndarray::Axis(axis), &[a_lower.view(), b_lower.view()])
                .map_err(|e| NyError::InvalidSpec(format!("Concat lower bounds failed: {}", e)))?;

        let out_upper =
            ndarray::concatenate(ndarray::Axis(axis), &[a_upper.view(), b_upper.view()])
                .map_err(|e| NyError::InvalidSpec(format!("Concat upper bounds failed: {}", e)))?;

        // Pure layout op (concatenation): value-preserving, so infinite bounds
        // pass through soundly. Allow `±inf` to flow without tripping the NaN
        // firewall; NaN is still rejected.
        BoundedTensor::new_allow_infinite(out_lower, out_upper)
    }

    /// Broadcast a tensor by adding a batch dimension at the front and repeating.
    /// Input shape [d1, d2, ...] -> output shape [batch_size, d1, d2, ...]
    pub(super) fn broadcast_to_batch(
        tensor: &ArrayD<f32>,
        batch_size: usize,
    ) -> Result<ArrayD<f32>> {
        let old_shape = tensor.shape();
        let mut new_shape = vec![batch_size];
        new_shape.extend_from_slice(old_shape);

        // Expand dimensions and broadcast
        let expanded = tensor.clone().insert_axis(ndarray::Axis(0));
        expanded
            .broadcast(IxDyn(&new_shape))
            .map(|v| v.to_owned())
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: new_shape,
                got: expanded.shape().to_vec(),
            })
    }

    /// CROWN backward propagation for Concat (Y = concat(A, B)).
    ///
    /// For concatenation, the Jacobian has a block structure:
    /// - ∂Y/∂A = [I, 0] (identity for A portion, zeros for B portion)
    /// - ∂Y/∂B = [0, I] (zeros for A portion, identity for B portion)
    ///
    /// When propagating backwards, we split the coefficient matrix:
    /// - Coefficients for first size_a elements go to input A
    /// - Coefficients for remaining size_b elements go to input B
    ///
    /// Returns (bounds_for_a, bounds_for_b).
    pub fn propagate_linear_binary(
        &self,
        bounds: &LinearBounds,
        input_a_shape: &[usize],
        input_b_shape: &[usize],
    ) -> Result<(LinearBounds, LinearBounds)> {
        let size_a: usize = checked_shape_product(input_a_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Concat: input_a shape product overflows usize: {:?}",
                input_a_shape,
            ))
        })?;
        let size_b: usize = checked_shape_product(input_b_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Concat: input_b shape product overflows usize: {:?}",
                input_b_shape,
            ))
        })?;
        let total_size = size_a.checked_add(size_b).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Concat: size_a ({}) + size_b ({}) overflows usize",
                size_a, size_b,
            ))
        })?;

        if bounds.num_inputs() != total_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![total_size],
                got: vec![bounds.num_inputs()],
            });
        }

        // Split coefficient matrices at size_a
        let lower_a_a = bounds.lower_a().slice(ndarray::s![.., ..size_a]).to_owned();
        let lower_a_b = bounds.lower_a().slice(ndarray::s![.., size_a..]).to_owned();
        let upper_a_a = bounds.upper_a().slice(ndarray::s![.., ..size_a]).to_owned();
        let upper_a_b = bounds.upper_a().slice(ndarray::s![.., size_a..]).to_owned();

        // Split bias evenly between the two inputs
        // Directed rounding on f32 halving to preserve soundness (#2173).
        let lower_b_half = bounds.lower_b().mapv(|v| next_down_f32(v * 0.5));
        let upper_b_half = bounds.upper_b().mapv(|v| next_up_f32(v * 0.5));

        let bounds_a = LinearBounds::new_or_conservative(
            lower_a_a,
            lower_b_half.clone(),
            upper_a_a,
            upper_b_half.clone(),
        )?;

        let bounds_b =
            LinearBounds::new_or_conservative(lower_a_b, lower_b_half, upper_a_b, upper_b_half)?;

        Ok((bounds_a, bounds_b))
    }

    /// Batched CROWN backward propagation for Concat.
    pub fn propagate_linear_batched_binary(
        &self,
        bounds: &BatchedLinearBounds,
        input_a_shape: &[usize],
        input_b_shape: &[usize],
    ) -> Result<(BatchedLinearBounds, BatchedLinearBounds)> {
        let size_a: usize = checked_shape_product(input_a_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Concat batched: input_a shape product overflows usize: {:?}",
                input_a_shape,
            ))
        })?;
        let size_b: usize = checked_shape_product(input_b_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Concat batched: input_b shape product overflows usize: {:?}",
                input_b_shape,
            ))
        })?;

        // The coefficient matrices are shaped [batch..., out_dim, in_dim]
        // We need to split in_dim at size_a
        let a_shape = bounds.lower_a.shape();
        let ndim = a_shape.len();
        if ndim < 2 {
            return Err(NyError::InvalidSpec(
                "Batched linear bounds must have at least 2 dimensions".to_string(),
            ));
        }

        let in_dim = a_shape[ndim - 1];
        let total_size = size_a.checked_add(size_b).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Concat batched: size_a ({}) + size_b ({}) overflows usize",
                size_a, size_b,
            ))
        })?;
        if in_dim != total_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![total_size],
                got: vec![in_dim],
            });
        }

        // Use ndarray slicing to split along the last dimension
        use ndarray::SliceInfoElem;
        let mut slice_a: Vec<SliceInfoElem> = (0..ndim - 1)
            .map(|_| SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1,
            })
            .collect();
        slice_a.push(SliceInfoElem::Slice {
            start: 0,
            end: Some(size_a as isize),
            step: 1,
        });

        let mut slice_b: Vec<SliceInfoElem> = (0..ndim - 1)
            .map(|_| SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1,
            })
            .collect();
        slice_b.push(SliceInfoElem::Slice {
            start: size_a as isize,
            end: None,
            step: 1,
        });

        let lower_a_a = bounds
            .lower_a
            .slice(slice_a.as_slice())
            .to_owned()
            .into_dyn();
        let lower_a_b = bounds
            .lower_a
            .slice(slice_b.as_slice())
            .to_owned()
            .into_dyn();
        let upper_a_a = bounds
            .upper_a
            .slice(slice_a.as_slice())
            .to_owned()
            .into_dyn();
        let upper_a_b = bounds
            .upper_a
            .slice(slice_b.as_slice())
            .to_owned()
            .into_dyn();

        // Split bias evenly
        // Directed rounding on f32 halving to preserve soundness (#2173).
        let lower_b_half = bounds.lower_b.mapv(|v| next_down_f32(v * 0.5));
        let upper_b_half = bounds.upper_b.mapv(|v| next_up_f32(v * 0.5));

        // Phase 4 audit: per-layer slicing + bias halving.
        let bounds_a = BatchedLinearBounds::new_or_conservative(
            lower_a_a,
            lower_b_half.clone(),
            upper_a_a,
            upper_b_half.clone(),
            input_a_shape.to_vec(),
            bounds.output_shape.clone(),
        )?;

        let bounds_b = BatchedLinearBounds::new_or_conservative(
            lower_a_b,
            lower_b_half,
            upper_a_b,
            upper_b_half,
            input_b_shape.to_vec(),
            bounds.output_shape.clone(),
        )?;

        Ok((bounds_a, bounds_b))
    }
}

mod nary;

#[cfg(test)]
mod tests;
