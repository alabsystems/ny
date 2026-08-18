// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, Axis, IxDyn};
use ny_core::{checked_shape_product, nan_propagating_max, NyError, Result};
use serde::{de::Error as _, Deserialize, Deserializer, Serialize};

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
#[derive(Debug, Clone, Serialize)]
pub struct ZonotopeTensor {
    /// Combined center and error coefficients.
    /// Shape: (1 + n_error_terms, ...element_shape)
    pub(crate) coeffs: ArrayD<f32>,

    /// Number of error terms (not including center).
    pub(crate) n_error_terms: usize,

    /// Shape of each element tensor (excludes the error term dimension).
    pub(crate) element_shape: Vec<usize>,
}

#[derive(Deserialize)]
struct SerializedZonotopeTensor {
    coeffs: ArrayD<f32>,
    n_error_terms: usize,
    element_shape: Vec<usize>,
}

impl<'de> Deserialize<'de> for ZonotopeTensor {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serialized = SerializedZonotopeTensor::deserialize(deserializer)?;
        let zonotope = Self::new(serialized.coeffs).map_err(D::Error::custom)?;
        if serialized.n_error_terms != zonotope.n_error_terms
            || serialized.element_shape != zonotope.element_shape
        {
            return Err(D::Error::custom(format!(
                "ZonotopeTensor serialized metadata does not match coefficient shape: \
                 expected n_error_terms={} and element_shape={:?}, got \
                 n_error_terms={} and element_shape={:?}",
                zonotope.n_error_terms,
                zonotope.element_shape,
                serialized.n_error_terms,
                serialized.element_shape
            )));
        }
        Ok(zonotope)
    }
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

        if shape[0] == 0 {
            return Err(NyError::InvalidSpec(
                "Zonotope coeffs must contain a center row".to_string(),
            ));
        }

        let n_error_terms = shape[0] - 1;
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

#[cfg(test)]
mod serde_tests {
    use super::*;
    use ndarray::arr1;

    #[derive(Serialize)]
    struct RawZonotopeTensor {
        coeffs: ArrayD<f32>,
        n_error_terms: usize,
        element_shape: Vec<usize>,
    }

    #[test]
    fn serde_round_trip_preserves_valid_zonotope() {
        let zonotope = ZonotopeTensor::from_input_shared(&arr1(&[1.0, 2.0]).into_dyn(), 0.25);

        let encoded = serde_json::to_string(&zonotope).expect("serialize");
        let decoded: ZonotopeTensor = serde_json::from_str(&encoded).expect("deserialize");

        assert_eq!(decoded.coeffs(), zonotope.coeffs());
        assert_eq!(decoded.n_error_terms(), zonotope.n_error_terms());
        assert_eq!(decoded.shape(), zonotope.shape());
    }

    #[test]
    fn new_and_serde_reject_missing_center_row() {
        let coeffs = ArrayD::zeros(IxDyn(&[0, 2]));
        assert!(ZonotopeTensor::new(coeffs.clone()).is_err());

        let raw = RawZonotopeTensor {
            coeffs,
            n_error_terms: 0,
            element_shape: vec![2],
        };
        let encoded = serde_json::to_string(&raw).expect("serialize malformed fixture");
        assert!(serde_json::from_str::<ZonotopeTensor>(&encoded).is_err());
    }

    #[test]
    fn serde_rejects_stale_shape_metadata() {
        let raw = RawZonotopeTensor {
            coeffs: ArrayD::zeros(IxDyn(&[2, 3])),
            n_error_terms: 7,
            element_shape: vec![99],
        };
        let encoded = serde_json::to_string(&raw).expect("serialize malformed fixture");

        assert!(serde_json::from_str::<ZonotopeTensor>(&encoded).is_err());
    }
}
