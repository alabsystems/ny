// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, Axis, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};

use super::ZonotopeTensor;

impl ZonotopeTensor {
    /// Convert to a zonotope with compatible structure for another zonotope.
    ///
    /// If zonotopes have different numbers of error terms, this expands
    /// the smaller one with zeros to match.
    pub fn expand_to_match(&self, other: &Self) -> Result<(Self, Self)> {
        if self.n_error_terms == other.n_error_terms && self.element_shape == other.element_shape {
            return Ok((self.clone(), other.clone()));
        }

        // For now, only support same shape
        if self.element_shape != other.element_shape {
            return Err(NyError::shape_mismatch(
                self.element_shape.clone(),
                other.element_shape.clone(),
            ));
        }

        let max_errors = self.n_error_terms.max(other.n_error_terms);
        let shape = &self.element_shape;

        // Expand self if needed
        let expanded_self = if self.n_error_terms < max_errors {
            let mut new_shape = vec![1 + max_errors];
            new_shape.extend_from_slice(shape);
            let mut new_coeffs = ArrayD::zeros(IxDyn(&new_shape));
            for i in 0..=self.n_error_terms {
                new_coeffs
                    .index_axis_mut(Axis(0), i)
                    .assign(&self.coeffs.index_axis(Axis(0), i));
            }
            Self {
                coeffs: new_coeffs,
                n_error_terms: max_errors,
                element_shape: shape.clone(),
            }
        } else {
            self.clone()
        };

        // Expand other if needed
        let expanded_other = if other.n_error_terms < max_errors {
            let mut new_shape = vec![1 + max_errors];
            new_shape.extend_from_slice(shape);
            let mut new_coeffs = ArrayD::zeros(IxDyn(&new_shape));
            for i in 0..=other.n_error_terms {
                new_coeffs
                    .index_axis_mut(Axis(0), i)
                    .assign(&other.coeffs.index_axis(Axis(0), i));
            }
            Self {
                coeffs: new_coeffs,
                n_error_terms: max_errors,
                element_shape: shape.clone(),
            }
        } else {
            other.clone()
        };

        Ok((expanded_self, expanded_other))
    }

    /// Reshape the zonotope to a new element shape.
    ///
    /// This preserves all correlations because we're just rearranging elements
    /// within each error term slice. The total number of elements must be preserved.
    ///
    /// # Arguments
    /// * `target_shape` - The new shape for the element tensor
    ///
    /// # Example
    /// ```text
    /// // Reshape zonotope preserving total element count:
    /// let z = z.reshape(&[2, 16])?;  // (4, 8) -> (2, 16)
    /// ```
    pub fn reshape(&self, target_shape: &[usize]) -> Result<Self> {
        let old_size = checked_shape_product(&self.element_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ZonotopeTensor::reshape: source shape product overflows: {:?}",
                self.element_shape
            ))
        })?;
        let new_size = checked_shape_product(target_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "ZonotopeTensor::reshape: target shape product overflows: {:?}",
                target_shape
            ))
        })?;

        if old_size != new_size {
            return Err(NyError::shape_mismatch(
                self.element_shape.clone(),
                target_shape.to_vec(),
            ));
        }

        // coeffs shape: (1 + n_error_terms, ...old_element_shape)
        // new shape: (1 + n_error_terms, ...target_shape)
        let mut new_coeffs_shape = vec![1 + self.n_error_terms];
        new_coeffs_shape.extend_from_slice(target_shape);

        // Ensure contiguous memory layout before reshaping
        // This is needed when the array comes from operations like tile that
        // may produce non-standard layouts
        let contiguous = if self.coeffs.is_standard_layout() {
            self.coeffs.clone()
        } else {
            self.coeffs.as_standard_layout().to_owned()
        };

        let new_coeffs = contiguous
            .into_shape_with_order(IxDyn(&new_coeffs_shape))
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "Failed to reshape zonotope coeffs from {:?} to {:?}: {}",
                    self.coeffs.shape(),
                    new_coeffs_shape,
                    e
                ))
            })?;

        Ok(Self {
            coeffs: new_coeffs,
            n_error_terms: self.n_error_terms,
            element_shape: target_shape.to_vec(),
        })
    }

    /// Tile (repeat) the zonotope along a specified axis.
    ///
    /// This preserves correlations because duplicated elements share the same
    /// error symbols as the original. This is essential for tracking GQA attention
    /// where K/V heads are repeated to match Q heads.
    ///
    /// # Arguments
    /// * `axis` - The axis in element_shape to tile (0-indexed)
    /// * `reps` - Number of times to repeat along that axis
    ///
    /// # Example
    /// ```text
    /// // Repeat along axis for GQA head expansion:
    /// let z = z.tile(0, 4)?;  // (4, 128) -> (16, 128)
    /// ```
    pub fn tile(&self, axis: usize, reps: usize) -> Result<Self> {
        if axis >= self.element_shape.len() {
            return Err(NyError::InvalidSpec(format!(
                "Tile axis {} out of bounds for shape {:?}",
                axis, self.element_shape
            )));
        }

        if reps == 0 {
            return Err(NyError::InvalidSpec("Tile reps must be > 0".to_string()));
        }

        if reps == 1 {
            return Ok(self.clone());
        }

        // In coeffs, axis 0 is error terms, so element axis i corresponds to coeffs axis i+1
        let coeffs_axis = axis + 1;

        // Use ndarray concatenate to repeat
        let views: Vec<_> = std::iter::repeat_n(self.coeffs.view(), reps).collect();

        let new_coeffs = ndarray::concatenate(Axis(coeffs_axis), &views).map_err(|e| {
            NyError::InvalidSpec(format!(
                "Failed to tile zonotope along axis {}: {}",
                axis, e
            ))
        })?;

        // Update element shape
        let mut new_element_shape = self.element_shape.clone();
        new_element_shape[axis] *= reps;

        Ok(Self {
            coeffs: new_coeffs,
            n_error_terms: self.n_error_terms,
            element_shape: new_element_shape,
        })
    }

    /// Transpose the last two dimensions of the zonotope.
    ///
    /// For a zonotope with element_shape (..., M, N), produces one with shape (..., N, M).
    /// This is needed for matmul operations where we want A @ B instead of A @ B^T.
    pub fn transpose_last_two(&self) -> Result<Self> {
        if self.element_shape.len() < 2 {
            return Err(NyError::InvalidSpec(format!(
                "transpose_last_two requires at least 2 dimensions, got {:?}",
                self.element_shape
            )));
        }

        let ndim = self.coeffs.ndim();
        let mut axes: Vec<usize> = (0..ndim).collect();
        // Swap the last two axes in coeffs (which are last two element dims)
        axes.swap(ndim - 2, ndim - 1);

        let transposed = self.coeffs.clone().permuted_axes(axes);

        let mut new_element_shape = self.element_shape.clone();
        let n = new_element_shape.len();
        new_element_shape.swap(n - 2, n - 1);

        Ok(Self {
            coeffs: transposed,
            n_error_terms: self.n_error_terms,
            element_shape: new_element_shape,
        })
    }
}
