// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shape manipulation operations on BoundedTensor: reshape, flatten, slice, expand, concat, stack.

use ndarray::{Axis, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};

use super::BoundedTensor;

impl BoundedTensor {
    /// Reshape the tensor.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        let current_shape = self.lower.shape();
        let current_size = checked_shape_product(current_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "BoundedTensor::reshape: current shape product overflows: {:?}",
                current_shape
            ))
        })?;
        let target_size = checked_shape_product(shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "BoundedTensor::reshape: target shape product overflows: {:?}",
                shape
            ))
        })?;

        // Check if shapes are identical - no reshape needed
        if current_shape == shape {
            return Ok(Self {
                lower: self.lower.clone(),
                upper: self.upper.clone(),
                // Drop the L2 annotation: reshape produces a fresh tensor whose
                // sphere is not re-derived. Sound (only forgoes tightening).
                l2: None,
            });
        }

        // Check element count before attempting reshape
        if current_size != target_size {
            return Err(NyError::shape_mismatch(
                shape.to_vec(),
                current_shape.to_vec(),
            ));
        }

        // Make arrays contiguous before reshaping to avoid layout issues
        let lower_contiguous = if self.lower.is_standard_layout() {
            self.lower.clone()
        } else {
            self.lower.as_standard_layout().to_owned()
        };
        let upper_contiguous = if self.upper.is_standard_layout() {
            self.upper.clone()
        } else {
            self.upper.as_standard_layout().to_owned()
        };

        let lower = lower_contiguous
            .into_shape_with_order(IxDyn(shape))
            .map_err(|_| NyError::shape_mismatch(shape.to_vec(), current_shape.to_vec()))?;
        let upper = upper_contiguous
            .into_shape_with_order(IxDyn(shape))
            .map_err(|_| NyError::shape_mismatch(shape.to_vec(), current_shape.to_vec()))?;
        Ok(Self {
            lower,
            upper,
            l2: None,
        })
    }

    /// Flatten the tensor to 1D.
    ///
    /// Collects elements in standard (row-major) order into a 1D array.
    /// This is infallible: `iter()` yields exactly `self.len()` elements
    /// regardless of memory layout, and `Array1::from_vec` + `into_dyn()`
    /// cannot fail.
    pub fn flatten(&self) -> Self {
        let lower_vec: Vec<f32> = self.lower.iter().copied().collect();
        let upper_vec: Vec<f32> = self.upper.iter().copied().collect();
        Self {
            lower: ndarray::Array1::from_vec(lower_vec).into_dyn(),
            upper: ndarray::Array1::from_vec(upper_vec).into_dyn(),
            l2: None,
        }
    }

    /// Flatten and extract lower/upper bounds as `Array1<f32>`.
    ///
    /// This encodes the invariant that `flatten()` always produces 1D arrays,
    /// so `into_dimensionality::<Ix1>()` cannot fail. Returns
    /// `NyError::InternalError` if the invariant is violated (indicates a
    /// bug in `flatten()`, not user error).
    ///
    /// Use this instead of manual `flatten() → .lower().clone() →
    /// into_dimensionality::<Ix1>()` chains to avoid silent zero substitution
    /// patterns (#1926, #1931).
    pub fn flatten_to_ix1(
        &self,
        context: &str,
    ) -> Result<(ndarray::Array1<f32>, ndarray::Array1<f32>)> {
        let flat = self.flatten();
        let lower = flat
            .lower
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|e| {
                NyError::InternalError(format!(
                    "flatten_to_ix1 lower at '{}' not convertible to Ix1 \
                     (shape {:?}): {} — this indicates a bug in BoundedTensor::flatten()",
                    context,
                    self.lower.shape(),
                    e
                ))
            })?;
        let upper = flat
            .upper
            .into_dimensionality::<ndarray::Ix1>()
            .map_err(|e| {
                NyError::InternalError(format!(
                    "flatten_to_ix1 upper at '{}' not convertible to Ix1 \
                     (shape {:?}): {} — this indicates a bug in BoundedTensor::flatten()",
                    context,
                    self.upper.shape(),
                    e
                ))
            })?;
        Ok((lower, upper))
    }

    /// Extract a single slice along the specified axis.
    ///
    /// Returns a tensor with the specified axis removed (not kept as size-1).
    ///
    /// # Arguments
    /// * `axis` - The axis to slice along
    /// * `index` - The index to select
    ///
    /// # Example
    /// ```text
    /// // Extract single position from tensor:
    /// let pos_0 = tensor.slice_axis(1, 0)?;  // [1, 768] from [1, 512, 768]
    /// ```
    pub fn slice_axis(&self, axis: usize, index: usize) -> Result<BoundedTensor> {
        let shape = self.shape();
        if axis >= shape.len() {
            return Err(NyError::InvalidSpec(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                shape.len()
            )));
        }
        if index >= shape[axis] {
            return Err(NyError::InvalidSpec(format!(
                "Index {} out of bounds for axis {} with size {}",
                index, axis, shape[axis]
            )));
        }

        let lower = self.lower.index_axis(Axis(axis), index).to_owned();
        let upper = self.upper.index_axis(Axis(axis), index).to_owned();

        BoundedTensor::new(lower, upper)
    }

    /// Extract a range of slices along the specified axis.
    ///
    /// Returns a tensor with reduced size along the specified axis.
    ///
    /// # Arguments
    /// * `axis` - The axis to slice along
    /// * `start` - Starting index (inclusive)
    /// * `end` - Ending index (exclusive)
    ///
    /// # Example
    /// ```text
    /// // Extract range along axis:
    /// let first_half = tensor.slice_axis_range(1, 0, 256)?;  // [1, 256, 768]
    /// ```
    pub fn slice_axis_range(&self, axis: usize, start: usize, end: usize) -> Result<BoundedTensor> {
        use ndarray::{Axis, Slice};

        let shape = self.shape();
        if axis >= shape.len() {
            return Err(NyError::InvalidSpec(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                shape.len()
            )));
        }
        if end > shape[axis] {
            return Err(NyError::InvalidSpec(format!(
                "End index {} out of bounds for axis {} with size {}",
                end, axis, shape[axis]
            )));
        }
        if start >= end {
            return Err(NyError::InvalidSpec(format!(
                "Invalid range: start {} >= end {}",
                start, end
            )));
        }

        // Use slice_axis which is cleaner than building SliceInfo manually
        let slice_spec = Slice::from(start..end);
        let lower = self
            .lower
            .slice_axis(Axis(axis), slice_spec)
            .as_standard_layout()
            .into_owned();
        let upper = self
            .upper
            .slice_axis(Axis(axis), slice_spec)
            .as_standard_layout()
            .into_owned();

        BoundedTensor::new(lower, upper)
    }

    /// Insert a size-1 axis at the specified position.
    ///
    /// # Arguments
    /// * `axis` - Where to insert the new axis
    ///
    /// # Example
    /// ```text
    /// // Insert size-1 axis:
    /// let expanded = tensor.expand_axis(1)?;  // [1, 768] -> [1, 1, 768]
    /// ```
    pub fn expand_axis(&self, axis: usize) -> Result<BoundedTensor> {
        let shape = self.shape();
        if axis > shape.len() {
            return Err(NyError::InvalidSpec(format!(
                "Axis {} out of bounds for inserting into tensor with {} dimensions",
                axis,
                shape.len()
            )));
        }

        let lower = self.lower.clone().insert_axis(Axis(axis));
        let upper = self.upper.clone().insert_axis(Axis(axis));

        BoundedTensor::new(lower, upper)
    }

    /// Concatenate multiple bounded tensors along an axis.
    ///
    /// # Arguments
    /// * `tensors` - Slice of tensors to concatenate
    /// * `axis` - Axis along which to concatenate
    ///
    /// # Example
    /// ```text
    /// // Concatenate position embeddings:
    /// let pos_0 = tensor.slice_axis(1, 0)?;  // [1, 768]
    /// let pos_1 = tensor.slice_axis(1, 1)?;  // [1, 768]
    /// let combined = BoundedTensor::concat(&[pos_0.expand_axis(1)?, pos_1.expand_axis(1)?], 1)?;
    /// // Result: [1, 2, 768]
    /// ```
    pub fn concat(tensors: &[BoundedTensor], axis: usize) -> Result<BoundedTensor> {
        if tensors.is_empty() {
            return Err(NyError::InvalidSpec(
                "Cannot concatenate empty tensor list".to_string(),
            ));
        }

        let first_shape = tensors[0].shape();
        if axis >= first_shape.len() {
            return Err(NyError::InvalidSpec(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                first_shape.len()
            )));
        }

        // Validate shapes (all must match except along concat axis)
        for (i, t) in tensors.iter().enumerate().skip(1) {
            let shape = t.shape();
            if shape.len() != first_shape.len() {
                return Err(NyError::shape_mismatch(
                    first_shape.to_vec(),
                    shape.to_vec(),
                ));
            }
            for (d, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if d != axis && s1 != s2 {
                    return Err(NyError::InvalidSpec(format!(
                        "Shape mismatch at tensor {}: dimension {} is {} but expected {}",
                        i, d, s2, s1
                    )));
                }
            }
        }

        let lower_views: Vec<_> = tensors.iter().map(|t| t.lower.view()).collect();
        let upper_views: Vec<_> = tensors.iter().map(|t| t.upper.view()).collect();

        let lower = ndarray::concatenate(Axis(axis), &lower_views)
            .map_err(|e| NyError::InvalidSpec(format!("Concatenation failed: {}", e)))?;
        let upper = ndarray::concatenate(Axis(axis), &upper_views)
            .map_err(|e| NyError::InvalidSpec(format!("Concatenation failed: {}", e)))?;

        BoundedTensor::new(lower, upper)
    }

    /// Stack multiple bounded tensors along a new axis.
    ///
    /// Creates a new axis and stacks tensors along it.
    ///
    /// # Arguments
    /// * `tensors` - Slice of tensors to stack (all must have same shape)
    /// * `axis` - Where to insert the new stacking axis
    ///
    /// # Example
    /// ```text
    /// // Stack position embeddings along new sequence axis:
    /// let stacked = BoundedTensor::stack(&[pos_0, pos_1, pos_2], 1)?;  // [1, 3, 768]
    /// ```
    pub fn stack(tensors: &[BoundedTensor], axis: usize) -> Result<BoundedTensor> {
        if tensors.is_empty() {
            return Err(NyError::InvalidSpec(
                "Cannot stack empty tensor list".to_string(),
            ));
        }

        let first_shape = tensors[0].shape();
        if axis > first_shape.len() {
            return Err(NyError::InvalidSpec(format!(
                "Axis {} out of bounds for stacking into tensor with {} dimensions",
                axis,
                first_shape.len()
            )));
        }

        // Validate all shapes match exactly
        for t in tensors.iter().skip(1) {
            if t.shape() != first_shape {
                return Err(NyError::shape_mismatch(
                    first_shape.to_vec(),
                    t.shape().to_vec(),
                ));
            }
        }

        let lower_views: Vec<_> = tensors.iter().map(|t| t.lower.view()).collect();
        let upper_views: Vec<_> = tensors.iter().map(|t| t.upper.view()).collect();

        let lower = ndarray::stack(Axis(axis), &lower_views)
            .map_err(|e| NyError::InvalidSpec(format!("Stacking failed: {}", e)))?;
        let upper = ndarray::stack(Axis(axis), &upper_views)
            .map_err(|e| NyError::InvalidSpec(format!("Stacking failed: {}", e)))?;

        BoundedTensor::new(lower, upper)
    }
}
