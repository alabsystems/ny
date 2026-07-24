// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, Axis, IxDyn};
use ny_core::{checked_shape_product, nan_propagating_max, NyError, Result};
use serde::{Deserialize, Serialize};

use crate::BoundedTensor;

/// A zonotope tensor: center + Σᵢ (coeffᵢ · eᵢ) where eᵢ ∈ [-1, 1]
///
/// Memory layout: coeffs has shape `(1 + n_error_terms, ...element_shape)`
/// - `coeffs[0]` = center
/// - `coeffs[1..]` = error term coefficients
///
/// # Example
///
/// For input x ∈ [0.9, 1.1] (center=1.0, epsilon=0.1):
/// ```text
/// Zonotope: x = 1.0 + 0.1·e₁   where e₁ ∈ [-1, 1]
/// coeffs = [[1.0], [0.1]]  (shape: 2×1)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZonotopeTensor {
    /// Combined center and error coefficients.
    /// Shape: (1 + n_error_terms, ...element_shape)
    pub(crate) coeffs: ArrayD<f32>,

    /// Number of error terms (not including center).
    pub(crate) n_error_terms: usize,

    /// Shape of each element tensor (excludes the error term dimension).
    pub(crate) element_shape: Vec<usize>,
}

impl ZonotopeTensor {
    /// Create a zonotope from combined coefficients array.
    ///
    /// # Arguments
    /// * `coeffs` - Array of shape (1 + n_error_terms, ...element_shape)
    pub fn new(coeffs: ArrayD<f32>) -> Result<Self> {
        let shape = coeffs.shape();
        if shape.is_empty() {
            return Err(NyError::InvalidSpec(
                "Zonotope coeffs must have at least 1 dimension".to_string(),
            ));
        }

        let n_error_terms = shape[0].saturating_sub(1);
        let element_shape = shape[1..].to_vec();

        Ok(Self {
            coeffs,
            n_error_terms,
            element_shape,
        })
    }

    /// Create a zonotope representing a concrete value (no uncertainty).
    pub fn concrete(values: ArrayD<f32>) -> Self {
        let mut coeffs_shape = vec![1];
        coeffs_shape.extend_from_slice(values.shape());

        let mut coeffs = ArrayD::zeros(IxDyn(&coeffs_shape));
        coeffs.index_axis_mut(Axis(0), 0).assign(&values);

        Self {
            coeffs,
            n_error_terms: 0,
            element_shape: values.shape().to_vec(),
        }
    }

    /// Get the combined center and error coefficient array.
    pub fn coeffs(&self) -> &ArrayD<f32> {
        &self.coeffs
    }

    /// Get the number of error terms (not including center).
    pub fn n_error_terms(&self) -> usize {
        self.n_error_terms
    }

    /// Get the center tensor (point estimate).
    pub fn center(&self) -> ArrayD<f32> {
        self.coeffs.index_axis(Axis(0), 0).to_owned()
    }

    /// Compute the radius at each element (max deviation from center).
    ///
    /// radius = Σᵢ |coeffᵢ| (sum of absolute error coefficients)
    pub fn radius(&self) -> ArrayD<f32> {
        let mut radius = ArrayD::zeros(IxDyn(&self.element_shape));

        for i in 1..=self.n_error_terms {
            radius = radius + self.coeffs.index_axis(Axis(0), i).mapv(f32::abs);
        }

        radius
    }

    /// Convert zonotope to interval bounds [center - radius, center + radius].
    ///
    /// This is a lossy conversion - interval bounds don't track correlations.
    /// Returns `Err` if the resulting bounds contain NaN (e.g. from NaN coefficients).
    /// Infinite bounds are permitted (legitimately uncertain zonotopes).
    pub fn to_bounded_tensor(&self) -> Result<BoundedTensor> {
        let center = self.center();
        let radius = self.radius();

        BoundedTensor::new_allow_infinite(&center - &radius, &center + &radius)
    }

    /// Get the shape of each element (excluding error term dimension).
    pub fn shape(&self) -> &[usize] {
        &self.element_shape
    }

    /// Total number of elements per error term.
    pub fn len(&self) -> usize {
        checked_shape_product(&self.element_shape)
            .expect("ZonotopeTensor::len: element shape product overflows")
    }

    /// Check if the zonotope has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum width (upper - lower) across all elements.
    ///
    /// Uses `nan_propagating_max` so NaN coefficients surface as NaN width
    /// instead of being silently absorbed by `f32::max(NaN, x) == x`.
    pub fn max_width(&self) -> f32 {
        let radius = self.radius();
        let width = radius.mapv(|r| 2.0 * r);
        width.iter().cloned().fold(0.0_f32, nan_propagating_max)
    }

    /// Check if bounds have exploded to infinity.
    pub fn has_unbounded(&self) -> bool {
        self.coeffs.iter().any(|v| v.is_infinite())
    }
}
