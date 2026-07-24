// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2, ArrayD, Axis, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use std::borrow::Cow;

use super::ZonotopeTensor;

/// Linear operations on zonotopes (preserve zonotope form exactly).
impl ZonotopeTensor {
    /// Scalar addition: z + c
    pub fn shift(&self, scalar: f32) -> Self {
        let mut result = self.clone();
        result
            .coeffs
            .index_axis_mut(Axis(0), 0)
            .mapv_inplace(|v| v + scalar);
        result
    }

    /// Scalar multiplication: c * z
    pub fn scale(&self, scalar: f32) -> Self {
        let mut result = self.clone();
        result.coeffs.mapv_inplace(|v| v * scalar);
        result
    }

    /// Element-wise addition of two zonotopes with same error symbols.
    ///
    /// (a₀ + Σᵢ aᵢeᵢ) + (b₀ + Σᵢ bᵢeᵢ) = (a₀+b₀) + Σᵢ (aᵢ+bᵢ)eᵢ
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.n_error_terms != other.n_error_terms {
            return Err(NyError::InvalidSpec(format!(
                "Cannot add zonotopes with different error term counts: {} vs {}",
                self.n_error_terms, other.n_error_terms
            )));
        }

        if self.element_shape != other.element_shape {
            return Err(NyError::shape_mismatch(
                self.element_shape.clone(),
                other.element_shape.clone(),
            ));
        }

        let coeffs = &self.coeffs + &other.coeffs;
        Self::new(coeffs)
    }

    /// Linear transformation: W·z + b
    ///
    /// Applies weight matrix to center and all error coefficients.
    /// The zonotope form is preserved exactly.
    ///
    /// # Arguments
    /// * `weight` - Weight matrix of shape (out_features, in_features)
    /// * `bias` - Optional bias vector of shape (out_features,)
    ///
    /// # Input/Output
    /// * Input zonotope shape: (..., in_features)
    /// * Output zonotope shape: (..., out_features)
    pub fn linear(&self, weight: &Array2<f32>, bias: Option<&Array1<f32>>) -> Result<Self> {
        let in_features = weight.ncols();
        let out_features = weight.nrows();

        // Check that last dimension matches weight's in_features
        if self.element_shape.is_empty() || self.element_shape.last() != Some(&in_features) {
            return Err(NyError::shape_mismatch(
                vec![in_features],
                self.element_shape.clone(),
            ));
        }

        let coeffs: Cow<'_, ArrayD<f32>> = if self.coeffs.is_standard_layout() {
            Cow::Borrowed(&self.coeffs)
        } else {
            Cow::Owned(self.coeffs.as_standard_layout().to_owned())
        };
        let coeffs = coeffs.as_ref();

        let prefix_shape = &self.element_shape[..self.element_shape.len() - 1];
        let prefix_size = checked_shape_product(prefix_shape)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "zonotope linear: prefix shape product overflows: {:?}",
                    prefix_shape
                ))
            })?
            .max(1);
        let n_rows = 1 + self.n_error_terms;

        let mut result_shape = vec![n_rows];
        result_shape.extend_from_slice(prefix_shape);
        result_shape.push(out_features);
        let mut result_coeffs = ArrayD::<f32>::zeros(IxDyn(&result_shape));

        let weight_t = weight.t();
        for row in 0..n_rows {
            let input_view = coeffs
                .index_axis(Axis(0), row)
                .into_shape_with_order(IxDyn(&[prefix_size, in_features]))
                .map_err(|_| NyError::InvalidSpec("Cannot reshape linear input to 2D".to_string()))?
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| NyError::InvalidSpec("Cannot view linear input as 2D".to_string()))?;

            let output_2d = input_view.dot(&weight_t);
            let output = output_2d
                .into_dyn()
                .into_shape_with_order(IxDyn(&result_shape[1..]))
                .map_err(|_| NyError::InvalidSpec("Cannot reshape linear output".to_string()))?;

            result_coeffs.index_axis_mut(Axis(0), row).assign(&output);
        }

        if let Some(b) = bias {
            let mut center = result_coeffs.index_axis_mut(Axis(0), 0);
            let last_axis = center.ndim().saturating_sub(1);
            for mut lane in center.lanes_mut(Axis(last_axis)) {
                lane += &b.view();
            }
        }

        let mut new_element_shape = prefix_shape.to_vec();
        new_element_shape.push(out_features);

        Ok(Self {
            coeffs: result_coeffs,
            n_error_terms: self.n_error_terms,
            element_shape: new_element_shape,
        })
    }
}
