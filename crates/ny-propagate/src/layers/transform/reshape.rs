// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Reshape-family transform layers.

use ndarray::IxDyn;
use ny_core::{checked_shape_product, reshape_copy_axis_from_sentinel, NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;

use crate::{layers::common::BoundPropagation, LinearBounds};

/// Reshape layer: changes tensor shape while preserving total elements.
///
/// Used in attention to reshape [seq, hidden] -> [seq, heads, head_dim].
#[derive(Debug, Clone)]
pub struct ReshapeLayer {
    /// Target shape. -1 means infer that dimension.
    pub target_shape: Vec<i64>,
}

impl ReshapeLayer {
    /// Create a new reshape layer with target shape.
    pub fn new(target_shape: Vec<i64>) -> Self {
        Self { target_shape }
    }

    /// Validate target dimensions and compute the product of known (non-inferred) dimensions.
    /// Returns `(infer_idx, known_product)` where `infer_idx` is the index of the -1 dim.
    fn validate_target_dims(&self, input_shape: &[usize]) -> Result<(Option<usize>, usize)> {
        let mut infer_idx = None;
        let mut known_product: usize = 1;

        for (i, &dim) in self.target_shape.iter().enumerate() {
            if dim == -1 {
                if infer_idx.is_some() {
                    return Err(NyError::InvalidSpec(
                        "Reshape can only have one inferred dimension (-1)".to_string(),
                    ));
                }
                infer_idx = Some(i);
            } else if dim == 0 {
                if i < input_shape.len() {
                    known_product *= input_shape[i];
                } else {
                    return Err(NyError::InvalidSpec(format!(
                        "Reshape dimension 0 at index {} but input only has {} dims",
                        i,
                        input_shape.len()
                    )));
                }
            } else if let Some(axis) = reshape_copy_axis_from_sentinel(dim) {
                if axis >= input_shape.len() {
                    return Err(NyError::InvalidSpec(format!(
                        "Reshape copy-axis sentinel references input axis {} but input only has {} dims",
                        axis,
                        input_shape.len()
                    )));
                }
                known_product *= input_shape[axis];
            } else {
                // Guard: negative dims (other than -1, 0) are invalid per ONNX spec (#2911)
                if dim < 0 {
                    return Err(NyError::InvalidSpec(format!(
                        "Reshape dimension {} at index {} is invalid (only -1 and 0 are \
                         special; other negative values are not supported)",
                        dim, i,
                    )));
                }
                known_product *= dim as usize;
            }
        }
        Ok((infer_idx, known_product))
    }

    /// Compute the actual output shape given an input shape.
    pub fn compute_output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        let total_elements: usize = checked_shape_product(input_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Reshape: input shape product overflows usize: {:?}",
                input_shape,
            ))
        })?;

        let (_infer_idx, known_product) = self.validate_target_dims(input_shape)?;

        // Build output shape
        let mut output_shape: Vec<usize> = Vec::with_capacity(self.target_shape.len());
        for (i, &dim) in self.target_shape.iter().enumerate() {
            if dim == -1 {
                if known_product == 0 {
                    return Err(NyError::InvalidSpec(
                        "Cannot infer reshape dimension when other dimensions are zero".to_string(),
                    ));
                }
                output_shape.push(total_elements / known_product);
            } else if dim == 0 {
                output_shape.push(input_shape[i]);
            } else if let Some(axis) = reshape_copy_axis_from_sentinel(dim) {
                output_shape.push(input_shape[axis]);
            } else {
                output_shape.push(dim as usize);
            }
        }

        // Verify total elements match
        let output_total: usize = checked_shape_product(&output_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Reshape: output shape product overflows usize: {:?}",
                output_shape,
            ))
        })?;
        if output_total != total_elements {
            return Err(NyError::ShapeMismatch {
                expected: vec![total_elements],
                got: vec![output_total],
            });
        }

        Ok(output_shape)
    }
}

impl BoundPropagation for ReshapeLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let output_shape = self.compute_output_shape(input.shape())?;

        // Ensure contiguous memory layout before reshape (required after transpose).
        // as_standard_layout() returns a CowArray, into_owned() gives us owned contiguous data.
        let lower_contiguous = input.lower().as_standard_layout().into_owned();
        let upper_contiguous = input.upper().as_standard_layout().into_owned();

        // Reshape lower and upper bounds
        let lower = lower_contiguous
            .into_shape_with_order(IxDyn(&output_shape))
            .map_err(|e| NyError::InvalidSpec(format!("Reshape failed: {}", e)))?;
        let upper = upper_contiguous
            .into_shape_with_order(IxDyn(&output_shape))
            .map_err(|e| NyError::InvalidSpec(format!("Reshape failed: {}", e)))?;

        // Pure layout op: a reshape only permutes/relabels indices, so infinite
        // bounds pass through soundly. Allow `±inf` (sound conservative bounds from
        // skipped/opaque upstream ops) to flow without tripping the NaN firewall;
        // NaN is still rejected.
        BoundedTensor::new_allow_infinite(lower, upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Reshape is a linear operation (permutation of indices).
        // For CROWN backward propagation, we keep the same coefficients
        // since reshape doesn't change the values, just their arrangement.
        // The coefficient matrices still map to the same flat input.
        Ok(Cow::Borrowed(bounds))
    }
}

/// Flatten layer: reshapes tensor by flattening dimensions according to ONNX semantics.
///
/// ONNX Flatten uses an `axis` parameter:
/// - Input shape: (d_0, d_1, ..., d_n)
/// - Output shape: (d_0 * d_1 * ... * d_{axis-1}, d_axis * ... * d_n)
///
/// Special cases:
/// - axis=0: Output is (1, total_elements)
/// - axis=n: Output is (total_elements, 1)
///
/// This is commonly used in CNNs to flatten spatial dimensions before a Linear layer.
#[derive(Debug, Clone)]
pub struct FlattenLayer {
    /// The axis from which to flatten. Negative values count from the end.
    /// Default is 1 (flatten all dimensions except batch).
    pub axis: i32,
}

impl FlattenLayer {
    /// Create a new flatten layer with the specified axis.
    pub fn new(axis: i32) -> Self {
        Self { axis }
    }

    /// Create a flatten layer that flattens all dimensions (axis=0).
    /// Output shape: (1, total_elements)
    pub fn flatten_all() -> Self {
        Self { axis: 0 }
    }

    /// Compute the output shape given input shape.
    pub fn compute_output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        let ndim = input_shape.len();
        if ndim == 0 {
            return Err(NyError::InvalidSpec(
                "Flatten requires at least 1D input".to_string(),
            ));
        }

        // Handle negative axis with bounds checking
        let axis = if self.axis < 0 {
            let raw = ndim as i32 + self.axis;
            if raw < 0 {
                return Err(NyError::InvalidSpec(format!(
                    "Negative axis {} resolves to {} which is < 0 for ndim={}",
                    self.axis, raw, ndim
                )));
            }
            // SAFETY(as usize): raw is i32, guard above ensures raw >= 0.
            raw as usize
        } else {
            // SAFETY(as usize): self.axis is i32, else-branch means axis >= 0.
            let raw = self.axis as usize;
            if raw > ndim {
                return Err(NyError::InvalidSpec(format!(
                    "Axis {} is > ndim {}",
                    raw, ndim
                )));
            }
            raw
        };

        // Compute dimensions before and after axis
        let dim_before: usize = if axis == 0 {
            1
        } else {
            checked_shape_product(&input_shape[..axis]).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Flatten: shape product overflows usize for dims before axis {}: {:?}",
                    axis,
                    &input_shape[..axis],
                ))
            })?
        };

        let dim_after: usize = if axis >= ndim {
            1
        } else {
            checked_shape_product(&input_shape[axis..]).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Flatten: shape product overflows usize for dims from axis {}: {:?}",
                    axis,
                    &input_shape[axis..],
                ))
            })?
        };

        Ok(vec![dim_before, dim_after])
    }
}

impl BoundPropagation for FlattenLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let output_shape = self.compute_output_shape(input.shape())?;

        // Ensure contiguous memory layout before reshape
        let lower_contiguous = input.lower().as_standard_layout().into_owned();
        let upper_contiguous = input.upper().as_standard_layout().into_owned();

        // Reshape lower and upper bounds
        let lower = lower_contiguous
            .into_shape_with_order(IxDyn(&output_shape))
            .map_err(|e| NyError::InvalidSpec(format!("Flatten failed: {}", e)))?;
        let upper = upper_contiguous
            .into_shape_with_order(IxDyn(&output_shape))
            .map_err(|e| NyError::InvalidSpec(format!("Flatten failed: {}", e)))?;

        // Pure layout op (see Reshape note): infinite bounds pass through soundly.
        BoundedTensor::new_allow_infinite(lower, upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // Flatten is a linear operation (index rearrangement).
        // For CROWN backward propagation, we keep the same coefficients
        // since flatten doesn't change the values, just their arrangement.
        Ok(Cow::Borrowed(bounds))
    }
}
