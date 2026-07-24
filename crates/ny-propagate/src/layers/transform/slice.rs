// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Slice layer: extracts a contiguous slice of a tensor along a specified axis.

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;
use tracing::debug;

use super::super::common::BoundPropagation;
use crate::{contiguous_flat_slice, contiguous_flat_slice_mut, BatchedLinearBounds, LinearBounds};

/// Slice layer: extracts a contiguous slice of a tensor along a specified axis.
///
/// Used to implement Split op (which produces multiple outputs, each a slice).
/// For input of shape [..., N, ...] along axis, extracts indices [start:end).
///
/// For IBP: slice(lower), slice(upper)
/// For CROWN backward: expand coefficients back to original size (pad with zeros)
#[derive(Debug, Clone)]
pub struct SliceLayer {
    /// Axis along which to slice (supports negative indexing).
    pub axis: i32,
    /// Start index (inclusive).
    pub start: usize,
    /// End index (exclusive).
    pub end: usize,
    /// Optional ONNX-style end offset from the runtime axis length.
    end_offset_from_axis_end: Option<usize>,
    /// Input shape (required for CROWN backward propagation).
    input_shape: Option<Vec<usize>>,
}

impl SliceLayer {
    /// Create a new slice layer.
    pub fn new(axis: i32, start: usize, end: usize) -> Self {
        Self {
            axis,
            start,
            end,
            end_offset_from_axis_end: None,
            input_shape: None,
        }
    }

    /// Create a slice whose end is resolved as `axis_len - end_offset`.
    pub fn new_with_end_offset(axis: i32, start: usize, end_offset: usize) -> Self {
        Self {
            axis,
            start,
            end: usize::MAX,
            end_offset_from_axis_end: Some(end_offset),
            input_shape: None,
        }
    }

    /// Set the input shape for CROWN backward propagation.
    pub fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.input_shape = Some(shape);
    }

    /// Compute the positive axis index given the input dimension count.
    fn resolve_axis(&self, ndim: usize) -> Result<usize> {
        super::super::common::resolve_axis_i32(self.axis, ndim, "Slice")
    }

    /// Resolved `(axis, start, end)` for a given input shape, with ONNX
    /// clamping. In-crate helper for the f64 cell evaluator.
    pub(crate) fn resolved_range(&self, input_shape: &[usize]) -> Result<(usize, usize, usize)> {
        let (axis, end) = self.validate_range(input_shape)?;
        Ok((axis, self.start.min(input_shape[axis]), end))
    }

    /// Validate and clamp `start`/`end` to axis size, returning the resolved axis (#2759, #3206).
    ///
    /// ONNX semantics: both `start` and `end` are clamped to `[0, axis_len]`.
    /// ONNX uses INT64_MAX as a sentinel for "slice to end of dimension".
    /// After clamping, `start >= end` produces an empty slice (size 0), which is
    /// an error for bound propagation (likely a const-folding gap — this Slice
    /// should have been eliminated during ONNX loading).
    fn validate_range(&self, input_shape: &[usize]) -> Result<(usize, usize)> {
        let axis = self.resolve_axis(input_shape.len())?;
        let axis_len = input_shape[axis];
        // Clamp both start and end to axis size (ONNX spec #3206)
        let start = self.start.min(axis_len);
        let end = self
            .end_offset_from_axis_end
            .map(|offset| axis_len.saturating_sub(offset))
            .unwrap_or_else(|| self.end.min(axis_len));
        if start >= end {
            return Err(NyError::InvalidSpec(format!(
                "Slice range [{}:{}) empty after clamping to axis {} size {} \
                 (original [{}:{})); this Slice should have been const-folded \
                 during ONNX loading — see #3206",
                start, end, axis, axis_len, self.start, self.end
            )));
        }
        Ok((axis, end))
    }

    /// Compute output shape given input shape.
    pub fn compute_output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        let (axis, end) = self.validate_range(input_shape)?;
        let mut output_shape = input_shape.to_vec();
        // Safe: validate_range ensures start < end.
        output_shape[axis] = end - self.start;
        Ok(output_shape)
    }

    /// Compute checked input and output sizes for CROWN backward.
    /// Returns `(input_shape, input_size, output_shape, output_size)`.
    fn checked_linear_sizes(&self) -> Result<(&[usize], usize, Vec<usize>, usize)> {
        let input_shape = self.input_shape.as_ref().ok_or_else(|| {
            NyError::InvalidSpec("SliceLayer requires input_shape for CROWN backward".to_string())
        })?;
        let input_size = checked_shape_product(input_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Slice: input shape product overflows usize: {:?}",
                input_shape,
            ))
        })?;
        let output_shape = self.compute_output_shape(input_shape)?;
        let output_size = checked_shape_product(&output_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Slice: output shape product overflows usize: {:?}",
                output_shape,
            ))
        })?;
        Ok((input_shape, input_size, output_shape, output_size))
    }

    /// CROWN backward propagation with bounds (uses pre_activation shape).
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        // Create a copy with the input shape set
        let mut slice_with_shape = self.clone();
        slice_with_shape.set_input_shape(pre_activation.shape().to_vec());

        // Delegate to propagate_linear
        match slice_with_shape.propagate_linear(bounds)? {
            Cow::Owned(lb) => Ok(lb),
            Cow::Borrowed(_) => {
                // This shouldn't happen since propagate_linear returns Owned for SliceLayer
                Ok(bounds.clone())
            }
        }
    }

    /// Batched CROWN backward propagation.
    ///
    /// Expands the last dimension (columns) of the A coefficient matrices from
    /// `output_size` back to `input_size`, mapping each output column to the
    /// corresponding input column with the slice offset applied. Columns outside
    /// the slice range are zero (no contribution to the bound).
    ///
    /// A matrices: shape [...batch, out_dim, output_size] -> [...batch, out_dim, input_size]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        let input_shape = pre_activation.shape().to_vec();
        let input_size = checked_shape_product(&input_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Slice batched CROWN: input shape product overflows: {:?}",
                input_shape
            ))
        })?;
        let output_shape = self.compute_output_shape(&input_shape)?;
        let output_size = checked_shape_product(&output_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Slice batched CROWN: output shape product overflows: {:?}",
                output_shape
            ))
        })?;

        // Validate that the last dimension of A matches output_size
        let a_shape = bounds.lower_a.shape();
        let a_ndim = a_shape.len();
        if a_ndim < 2 {
            return Err(NyError::InvalidSpec(
                "Slice batched CROWN: A matrices must have at least 2 dimensions".to_string(),
            ));
        }
        let in_dim = a_shape[a_ndim - 1];
        if in_dim != output_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![output_size],
                got: vec![in_dim],
            });
        }

        // Build the output-flat -> input-flat index mapping (same logic as scalar propagate_linear)
        let ndim = input_shape.len();
        let axis = self.resolve_axis(ndim)?;
        // validate_range checks start < end and clamps end
        let _ = self.validate_range(&input_shape)?;

        let mut output_strides = vec![1usize; ndim];
        let mut input_strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        }

        // mapping[out_flat] = in_flat
        let mapping: Vec<usize> = (0..output_size)
            .map(|out_flat| {
                let mut multi_idx = vec![0usize; ndim];
                let mut remaining = out_flat;
                for i in 0..ndim {
                    multi_idx[i] = remaining / output_strides[i];
                    remaining %= output_strides[i];
                }
                multi_idx[axis] += self.start;
                multi_idx
                    .iter()
                    .zip(input_strides.iter())
                    .map(|(idx, stride)| idx * stride)
                    .sum()
            })
            .collect();

        // Build new A matrices with expanded last dimension
        let mut new_a_shape = a_shape.to_vec();
        new_a_shape[a_ndim - 1] = input_size;
        let mut new_lower_a = ArrayD::<f32>::zeros(IxDyn(&new_a_shape));
        let mut new_upper_a = ArrayD::<f32>::zeros(IxDyn(&new_a_shape));

        // Outer size = product of all dims except the last
        let outer_size: usize = checked_shape_product(&a_shape[..a_ndim - 1]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Slice batched CROWN: outer shape product overflows: {:?}",
                &a_shape[..a_ndim - 1],
            ))
        })?;

        let flat_lower = contiguous_flat_slice(&bounds.lower_a);
        let flat_upper = contiguous_flat_slice(&bounds.upper_a);
        let new_lower_flat = contiguous_flat_slice_mut(&mut new_lower_a)?;
        let new_upper_flat = contiguous_flat_slice_mut(&mut new_upper_a)?;

        // For each row in outer dimensions, scatter output columns to input columns
        for row in 0..outer_size {
            let old_base = row * output_size;
            let new_base = row * input_size;
            for (out_col, &in_col) in mapping.iter().enumerate() {
                new_lower_flat[new_base + in_col] = flat_lower[old_base + out_col];
                new_upper_flat[new_base + in_col] = flat_upper[old_base + out_col];
            }
        }

        debug!(
            "Slice batched CROWN: expanded {} -> {} columns across {} positions (axis={}, [{},{}))",
            output_size, input_size, outer_size, axis, self.start, self.end
        );

        BatchedLinearBounds::new_or_conservative(
            new_lower_a,
            bounds.lower_b.clone(),
            new_upper_a,
            bounds.upper_b.clone(),
            input_shape,
            bounds.output_shape.clone(),
        )
    }
}

impl BoundPropagation for SliceLayer {
    fn requires_pre_activation_bounds(&self) -> bool {
        // Slice needs pre-activation bounds to derive input_shape for CROWN backward.
        // Without this, the generic `propagate_crown_backward` trait dispatch would
        // call `propagate_linear()` directly, which errors because input_shape is None.
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        // Delegate to the inherent method which clones self, sets input_shape
        // from pre_activation, then calls propagate_linear.
        SliceLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }

    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let input_shape = input.shape();
        let ndim = input_shape.len();
        let (axis, end) = self.validate_range(input_shape)?;

        // Slice both lower and upper bounds
        let slice_info: Vec<ndarray::SliceInfoElem> = (0..ndim)
            .map(|i| {
                if i == axis {
                    ndarray::SliceInfoElem::Slice {
                        start: self.start as isize,
                        end: Some(end as isize),
                        step: 1,
                    }
                } else {
                    ndarray::SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    }
                }
            })
            .collect();

        let lower_slice_info = ndarray::SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_info.clone())
            .map_err(|e| NyError::InvalidSpec(format!("Slice: failed to build slice info: {e}")))?;
        let upper_slice_info = ndarray::SliceInfo::<_, IxDyn, IxDyn>::try_from(slice_info)
            .map_err(|e| NyError::InvalidSpec(format!("Slice: failed to build slice info: {e}")))?;
        let out_lower = input.lower().slice(lower_slice_info).to_owned();
        let out_upper = input.upper().slice(upper_slice_info).to_owned();

        // Pure layout op (sub-range selection): value-preserving, so infinite
        // bounds pass through soundly. Allow `±inf` to flow without tripping the
        // NaN firewall; NaN is still rejected.
        BoundedTensor::new_allow_infinite(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        let (input_shape, input_size, output_shape, output_size) = self.checked_linear_sizes()?;
        let ndim = input_shape.len();
        let axis = self.resolve_axis(ndim)?;

        let num_outputs = bounds.num_outputs();
        let num_inputs_current = bounds.num_inputs();

        if num_inputs_current != output_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![output_size],
                got: vec![num_inputs_current],
            });
        }

        // For slice backward: expand coefficients from sliced positions back to original positions
        // The coefficients for positions outside the slice are zero.
        let mut new_lower_a = Array2::<f32>::zeros((num_outputs, input_size));
        let mut new_upper_a = Array2::<f32>::zeros((num_outputs, input_size));

        // Compute strides for index mapping
        let mut output_strides = vec![1usize; ndim];
        let mut input_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        }

        // Map each output index back to its input index
        for out_flat in 0..output_size {
            // Convert flat output index to multi-dimensional index
            let mut multi_idx = vec![0usize; ndim];
            let mut remaining = out_flat;
            for i in 0..ndim {
                multi_idx[i] = remaining / output_strides[i];
                remaining %= output_strides[i];
            }

            // Adjust the slice axis index
            multi_idx[axis] += self.start;

            // Convert back to flat input index
            let in_flat: usize = multi_idx
                .iter()
                .zip(input_strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum();

            // Copy coefficients from output position to input position
            for row in 0..num_outputs {
                new_lower_a[[row, in_flat]] = bounds.lower_a()[[row, out_flat]];
                new_upper_a[[row, in_flat]] = bounds.upper_a()[[row, out_flat]];
            }
        }

        Ok(Cow::Owned(LinearBounds::new_or_conservative(
            new_lower_a,
            bounds.lower_b().clone(),
            new_upper_a,
            bounds.upper_b().clone(),
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, ArrayD, IxDyn};

    fn make_bounded(lower: ArrayD<f32>, upper: ArrayD<f32>) -> BoundedTensor {
        BoundedTensor::new(lower, upper).unwrap()
    }

    // -- compute_output_shape --

    #[ntest::timeout(5000)]
    #[test]
    fn test_output_shape_axis0() {
        let layer = SliceLayer::new(0, 1, 3);
        let result = layer.compute_output_shape(&[5, 4]).unwrap();
        assert_eq!(result, vec![2, 4]);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_output_shape_axis1() {
        let layer = SliceLayer::new(1, 0, 2);
        let result = layer.compute_output_shape(&[3, 5]).unwrap();
        assert_eq!(result, vec![3, 2]);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_output_shape_negative_axis() {
        let layer = SliceLayer::new(-1, 1, 4);
        let result = layer.compute_output_shape(&[2, 5]).unwrap();
        // -1 resolves to axis 1
        assert_eq!(result, vec![2, 3]);
    }

    // -- IBP propagation --

    #[ntest::timeout(5000)]
    #[test]
    fn test_ibp_slice_axis0() {
        // Input [4, 2], slice axis=0, start=1, end=3 -> output [2, 2]
        let lower =
            ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                .unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[4, 2]),
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        )
        .unwrap();
        let input = make_bounded(lower, upper);

        let layer = SliceLayer::new(0, 1, 3);
        let result = layer.propagate_ibp(&input).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        // Rows 1 and 2 of original
        assert_eq!(result.lower()[[0, 0]], 3.0);
        assert_eq!(result.lower()[[0, 1]], 4.0);
        assert_eq!(result.lower()[[1, 0]], 5.0);
        assert_eq!(result.lower()[[1, 1]], 6.0);
        assert_eq!(result.upper()[[0, 0]], 30.0);
        assert_eq!(result.upper()[[1, 1]], 60.0);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_ibp_slice_axis1() {
        // Input [2, 4], slice axis=1, start=0, end=2 -> output [2, 2]
        let lower =
            ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
                .unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 4]),
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        )
        .unwrap();
        let input = make_bounded(lower, upper);

        let layer = SliceLayer::new(1, 0, 2);
        let result = layer.propagate_ibp(&input).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.lower()[[0, 0]], 1.0);
        assert_eq!(result.lower()[[0, 1]], 2.0);
        assert_eq!(result.lower()[[1, 0]], 5.0);
        assert_eq!(result.lower()[[1, 1]], 6.0);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_ibp_slice_full_axis() {
        // Slicing the entire axis should be a no-op on values.
        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap();
        let input = make_bounded(lower.clone(), upper.clone());

        let layer = SliceLayer::new(0, 0, 3);
        let result = layer.propagate_ibp(&input).unwrap();

        assert_eq!(result.shape(), &[3]);
        assert_eq!(
            result.lower().as_slice().unwrap(),
            lower.as_slice().unwrap()
        );
        assert_eq!(
            result.upper().as_slice().unwrap(),
            upper.as_slice().unwrap()
        );
    }

    // -- resolve_axis error cases (#1952) --

    #[ntest::timeout(5000)]
    #[test]
    fn test_negative_axis_out_of_range_returns_error_1952() {
        // axis = -3 on a 2D tensor: |axis| > ndim → should return Err, not wrap.
        let layer = SliceLayer::new(-3, 0, 1);
        assert!(
            layer.compute_output_shape(&[4, 5]).is_err(),
            "negative axis exceeding ndim must return Err, not wrap to large usize"
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_positive_axis_out_of_range_returns_error_1952() {
        // axis = 3 on a 2D tensor → should return Err.
        let layer = SliceLayer::new(3, 0, 1);
        assert!(
            layer.compute_output_shape(&[4, 5]).is_err(),
            "axis >= ndim must return Err"
        );
    }

    // -- IBP error cases --

    #[ntest::timeout(5000)]
    #[test]
    fn test_ibp_slice_out_of_bounds_clamps_to_axis_size() {
        // ONNX uses INT64_MAX as "to end" sentinel. end > axis_size is clamped, not rejected.
        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap();
        let input = make_bounded(lower.clone(), upper);

        let layer = SliceLayer::new(0, 0, 5); // end > axis size → clamped to 3
        let result = layer.propagate_ibp(&input).unwrap();
        assert_eq!(result.shape(), &[3]); // whole tensor
        assert_eq!(
            result.lower().as_slice().unwrap(),
            lower.as_slice().unwrap()
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_ibp_slice_usize_max_sentinel_clamps_to_axis_size_3193() {
        // Regression: ONNX uses INT64_MAX → usize::MAX as "slice to end" sentinel.
        // validate_range must clamp this to axis_len, not reject it.
        // Source: #3193 — linearizenn models use end = MAX sentinel.
        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let input = make_bounded(lower.clone(), upper.clone());

        let layer = SliceLayer::new(0, 0, usize::MAX); // MAX sentinel → clamp to 4
        let result = layer.propagate_ibp(&input).unwrap();
        assert_eq!(result.shape(), &[4]); // whole tensor
        assert_eq!(
            result.lower().as_slice().unwrap(),
            lower.as_slice().unwrap()
        );
        assert_eq!(
            result.upper().as_slice().unwrap(),
            upper.as_slice().unwrap()
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_compute_output_shape_usize_max_sentinel_3193() {
        // Regression: compute_output_shape with MAX sentinel must clamp end to axis size.
        let layer = SliceLayer::new(1, 0, usize::MAX);
        let result = layer.compute_output_shape(&[3, 5]).unwrap();
        assert_eq!(result, vec![3, 5]); // end clamped to 5, so full axis
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_crown_backward_usize_max_sentinel_3193() {
        // Regression: CROWN backward with MAX sentinel must expand correctly.
        let mut layer = SliceLayer::new(0, 1, usize::MAX);
        layer.set_input_shape(vec![4]);

        // Sentinel end → clamp to 4, so slice [1:4] → 3 output elements
        let bounds = LinearBounds::new(
            Array2::eye(3),
            Array1::zeros(3),
            Array2::eye(3),
            Array1::zeros(3),
        )
        .unwrap();

        let result = layer.propagate_linear(&bounds).unwrap().into_owned();
        assert_eq!(result.lower_a.shape(), &[3, 4]);
        // Row 0 maps output[0] to input[1] (start=1)
        assert_eq!(result.lower_a[[0, 1]], 1.0);
        // Row 1 maps output[1] to input[2]
        assert_eq!(result.lower_a[[1, 2]], 1.0);
        // Row 2 maps output[2] to input[3]
        assert_eq!(result.lower_a[[2, 3]], 1.0);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_ibp_slice_start_ge_end() {
        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap();
        let input = make_bounded(lower, upper);

        let layer = SliceLayer::new(0, 2, 2); // start == end
        assert!(layer.propagate_ibp(&input).is_err());
    }

    // -- CROWN backward propagation --

    #[ntest::timeout(5000)]
    #[test]
    fn test_crown_backward_1d_slice() {
        // Input: [4], Slice: [1:3] -> output [2]
        // CROWN backward should expand [2] coefficients to [4] with zeros outside slice.
        let mut layer = SliceLayer::new(0, 1, 3);
        layer.set_input_shape(vec![4]);

        // Identity bounds for 2-element output
        let bounds = LinearBounds::new(
            Array2::eye(2),
            Array1::zeros(2),
            Array2::eye(2),
            Array1::zeros(2),
        )
        .unwrap();

        let result = layer.propagate_linear(&bounds).unwrap();
        let result = result.into_owned();

        // Output should be [2, 4] matrices
        assert_eq!(result.lower_a.shape(), &[2, 4]);
        // Row 0 should map to input position 1 (start=1)
        assert_eq!(result.lower_a[[0, 0]], 0.0);
        assert_eq!(result.lower_a[[0, 1]], 1.0); // identity for position 1
        assert_eq!(result.lower_a[[0, 2]], 0.0);
        assert_eq!(result.lower_a[[0, 3]], 0.0);
        // Row 1 should map to input position 2
        assert_eq!(result.lower_a[[1, 0]], 0.0);
        assert_eq!(result.lower_a[[1, 1]], 0.0);
        assert_eq!(result.lower_a[[1, 2]], 1.0); // identity for position 2
        assert_eq!(result.lower_a[[1, 3]], 0.0);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_crown_backward_2d_slice_axis1() {
        // Input: [2, 4], Slice axis=1, [1:3] -> output [2, 2]
        // Backward should map 4 output positions to 8 input positions.
        let mut layer = SliceLayer::new(1, 1, 3);
        layer.set_input_shape(vec![2, 4]);

        // Identity for 4 output elements (2*2)
        let bounds = LinearBounds::new(
            Array2::eye(4),
            Array1::zeros(4),
            Array2::eye(4),
            Array1::zeros(4),
        )
        .unwrap();

        let result = layer.propagate_linear(&bounds).unwrap();
        let result = result.into_owned();

        assert_eq!(result.lower_a.shape(), &[4, 8]);

        // Output flat index 0 -> multi [0,0] -> input multi [0, 0+1] = [0,1] -> input flat 1
        assert_eq!(result.lower_a[[0, 1]], 1.0);
        // Output flat index 1 -> multi [0,1] -> input multi [0, 1+1] = [0,2] -> input flat 2
        assert_eq!(result.lower_a[[1, 2]], 1.0);
        // Output flat index 2 -> multi [1,0] -> input multi [1, 0+1] = [1,1] -> input flat 5
        assert_eq!(result.lower_a[[2, 5]], 1.0);
        // Output flat index 3 -> multi [1,1] -> input multi [1, 1+1] = [1,2] -> input flat 6
        assert_eq!(result.lower_a[[3, 6]], 1.0);

        // All other positions should be zero
        let nonzero_count: usize = result.lower_a.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(
            nonzero_count, 4,
            "should have exactly 4 nonzero entries in lower_a"
        );
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_crown_backward_preserves_bias() {
        let mut layer = SliceLayer::new(0, 0, 2);
        layer.set_input_shape(vec![4]);

        let bounds = LinearBounds::new(
            Array2::eye(2),
            Array1::from_vec(vec![1.0, 2.0]),
            Array2::eye(2),
            Array1::from_vec(vec![3.0, 4.0]),
        )
        .unwrap();

        let result = layer.propagate_linear(&bounds).unwrap();
        let result = result.into_owned();

        assert_eq!(result.lower_b.as_slice().unwrap(), &[1.0, 2.0]);
        assert_eq!(result.upper_b.as_slice().unwrap(), &[3.0, 4.0]);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_crown_backward_non_identity_asymmetric_coefficients() {
        // Input: [4], Slice: [1:3] -> output [2].
        // Verify backward mapping preserves arbitrary coefficients and
        // keeps lower/upper paths distinct.
        let mut layer = SliceLayer::new(0, 1, 3);
        layer.set_input_shape(vec![4]);

        let bounds = LinearBounds::new(
            Array2::from_shape_vec((2, 2), vec![1.5, -2.0, 0.25, 3.0]).unwrap(),
            Array1::from_vec(vec![0.1, -0.2]),
            Array2::from_shape_vec((2, 2), vec![-0.5, 4.0, 2.5, -1.25]).unwrap(),
            Array1::from_vec(vec![1.1, -1.2]),
        )
        .unwrap();

        let result = layer.propagate_linear(&bounds).unwrap().into_owned();

        assert_eq!(result.lower_a.shape(), &[2, 4]);
        assert_eq!(result.upper_a.shape(), &[2, 4]);

        // row 0 maps sliced coords [0,1] -> input coords [1,2]
        assert_eq!(result.lower_a[[0, 0]], 0.0);
        assert_eq!(result.lower_a[[0, 1]], 1.5);
        assert_eq!(result.lower_a[[0, 2]], -2.0);
        assert_eq!(result.lower_a[[0, 3]], 0.0);
        assert_eq!(result.upper_a[[0, 0]], 0.0);
        assert_eq!(result.upper_a[[0, 1]], -0.5);
        assert_eq!(result.upper_a[[0, 2]], 4.0);
        assert_eq!(result.upper_a[[0, 3]], 0.0);

        // row 1 maps sliced coords [0,1] -> input coords [1,2]
        assert_eq!(result.lower_a[[1, 0]], 0.0);
        assert_eq!(result.lower_a[[1, 1]], 0.25);
        assert_eq!(result.lower_a[[1, 2]], 3.0);
        assert_eq!(result.lower_a[[1, 3]], 0.0);
        assert_eq!(result.upper_a[[1, 0]], 0.0);
        assert_eq!(result.upper_a[[1, 1]], 2.5);
        assert_eq!(result.upper_a[[1, 2]], -1.25);
        assert_eq!(result.upper_a[[1, 3]], 0.0);

        // Bias vectors are unchanged.
        assert_eq!(result.lower_b.as_slice().unwrap(), &[0.1, -0.2]);
        assert_eq!(result.upper_b.as_slice().unwrap(), &[1.1, -1.2]);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_crown_backward_requires_input_shape() {
        let layer = SliceLayer::new(0, 0, 2);
        let bounds = LinearBounds::new(
            Array2::eye(2),
            Array1::zeros(2),
            Array2::eye(2),
            Array1::zeros(2),
        )
        .unwrap();
        // Should error because input_shape is not set
        assert!(layer.propagate_linear(&bounds).is_err());
    }

    // -- propagate_linear_with_bounds --

    #[ntest::timeout(5000)]
    #[test]
    fn test_propagate_linear_with_bounds_sets_shape() {
        let layer = SliceLayer::new(0, 1, 3);
        let pre_act = make_bounded(
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0; 4]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0; 4]).unwrap(),
        );

        let bounds = LinearBounds::new(
            Array2::eye(2),
            Array1::zeros(2),
            Array2::eye(2),
            Array1::zeros(2),
        )
        .unwrap();

        let result = layer
            .propagate_linear_with_bounds(&bounds, &pre_act)
            .unwrap();
        assert_eq!(result.lower_a.shape(), &[2, 4]);
    }

    // -- Batched CROWN backward propagation (#3188) --

    #[ntest::timeout(5000)]
    #[test]
    fn test_batched_crown_backward_1d_slice() {
        // Input: [4], Slice: [1:3] -> output [2]
        // Batched identity bounds: A shape [2, 2], expand to [2, 4]
        use crate::BatchedLinearBounds;

        let layer = SliceLayer::new(0, 1, 3);
        let pre_act = make_bounded(
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0; 4]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0; 4]).unwrap(),
        );

        let bounds = BatchedLinearBounds::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
            vec![2],
            vec![2],
        )
        .unwrap();

        let result = layer.propagate_linear_batched(&bounds, &pre_act).unwrap();
        assert_eq!(result.lower_a.shape(), &[2, 4]);
        // Row 0 maps output[0] to input[1] (start=1)
        assert_eq!(result.lower_a[[0, 0]], 0.0);
        assert_eq!(result.lower_a[[0, 1]], 1.0);
        assert_eq!(result.lower_a[[0, 2]], 0.0);
        assert_eq!(result.lower_a[[0, 3]], 0.0);
        // Row 1 maps output[1] to input[2]
        assert_eq!(result.lower_a[[1, 0]], 0.0);
        assert_eq!(result.lower_a[[1, 1]], 0.0);
        assert_eq!(result.lower_a[[1, 2]], 1.0);
        assert_eq!(result.lower_a[[1, 3]], 0.0);
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_batched_crown_backward_2d_slice_axis1() {
        // Input: [2, 4], Slice axis=1, [1:3] -> output [2, 2]
        // Batched: A shape [4, 4], expand to [4, 8]
        use crate::BatchedLinearBounds;

        let layer = SliceLayer::new(1, 1, 3);
        let pre_act = make_bounded(
            ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![0.0; 8]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0; 8]).unwrap(),
        );

        // Identity for 4 output elements
        let eye_4: Vec<f32> = (0..16)
            .map(|i| if i / 4 == i % 4 { 1.0 } else { 0.0 })
            .collect();
        let bounds = BatchedLinearBounds::new(
            ArrayD::from_shape_vec(IxDyn(&[4, 4]), eye_4.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0; 4]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[4, 4]), eye_4).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0; 4]).unwrap(),
            vec![4],
            vec![4],
        )
        .unwrap();

        let result = layer.propagate_linear_batched(&bounds, &pre_act).unwrap();
        assert_eq!(result.lower_a.shape(), &[4, 8]);

        // Output flat 0 -> multi [0,0] -> input [0, 1] -> flat 1
        assert_eq!(result.lower_a[[0, 1]], 1.0);
        // Output flat 1 -> multi [0,1] -> input [0, 2] -> flat 2
        assert_eq!(result.lower_a[[1, 2]], 1.0);
        // Output flat 2 -> multi [1,0] -> input [1, 1] -> flat 5
        assert_eq!(result.lower_a[[2, 5]], 1.0);
        // Output flat 3 -> multi [1,1] -> input [1, 2] -> flat 6
        assert_eq!(result.lower_a[[3, 6]], 1.0);

        // Exactly 4 nonzero entries
        let nonzero: usize = result.lower_a.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nonzero, 4);
    }
}
