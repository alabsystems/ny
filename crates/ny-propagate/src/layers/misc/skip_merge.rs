// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Skip and placeholder layers for graph dependency preservation.

use ny_core::Result;
use ny_tensor::BoundedTensor;
use std::borrow::Cow;

use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

/// Placeholder layer for skipped ops that preserves graph dependencies.
///
/// The layer behaves like an identity on its first input during bound
/// propagation while allowing the graph node to reference multiple inputs.
#[derive(Debug, Clone, Default)]
pub struct SkipMergeLayer;

impl SkipMergeLayer {
    /// Create a new skip-merge layer.
    pub fn new() -> Self {
        Self
    }
}

impl BoundPropagation for SkipMergeLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        Ok(input.clone())
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Ok(Cow::Borrowed(bounds))
    }
}

/// Opaque layer for skipped ops with multiple activation inputs.
///
/// Returns conservative unbounded bounds to avoid implying identity behavior.
///
/// When the skipped op's output shape is known from load-time shape inference
/// (`output_shape`), the unbounded bounds are emitted in that DECLARED shape
/// instead of echoing the first input's shape. This keeps downstream
/// shape-sensitive ops (Concat, Reshape, ScatterND) consistent so they
/// propagate `[-inf, +inf]` instead of hard-erroring on a shape mismatch
/// (cctsdb_yolo_2023). Soundness: bounds are `[-inf, +inf]` either way — the
/// shape only determines the layout of the conservative substitution.
#[derive(Debug, Clone, Default)]
pub struct OpaqueSkipLayer {
    /// Declared output shape from load-time shape inference (internal,
    /// unbatched convention). `None` falls back to the first input's shape.
    output_shape: Option<Vec<usize>>,
}

impl OpaqueSkipLayer {
    /// Create a new opaque skip layer (output shape follows the first input).
    pub fn new() -> Self {
        Self { output_shape: None }
    }

    /// Create an opaque skip layer with a declared output shape.
    pub fn with_output_shape(output_shape: Vec<usize>) -> Self {
        Self {
            output_shape: Some(output_shape),
        }
    }

    /// Declared output shape, if known.
    pub fn output_shape(&self) -> Option<&[usize]> {
        self.output_shape.as_deref()
    }

    fn unbounded_like(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let dim = match &self.output_shape {
            Some(shape) => ndarray::IxDyn(shape),
            None => input.lower().raw_dim(),
        };
        let lower = ndarray::ArrayD::from_elem(dim.clone(), f32::NEG_INFINITY);
        let upper = ndarray::ArrayD::from_elem(dim, f32::INFINITY);
        BoundedTensor::new_allow_infinite(lower, upper)
    }

    fn unbounded_linear(bounds: &LinearBounds) -> LinearBounds {
        LinearBounds::conservative(bounds.num_outputs(), bounds.num_inputs())
    }
}

impl BoundPropagation for OpaqueSkipLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.unbounded_like(input)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Ok(Cow::Owned(Self::unbounded_linear(bounds)))
    }
}

#[cfg(test)]
mod tests {
    use super::OpaqueSkipLayer;
    use crate::layers::common::BoundPropagation;
    use crate::LinearBounds;
    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    #[test]
    fn opaque_skip_ibp_returns_unbounded_bounds() {
        let lower = ArrayD::from_elem(IxDyn(&[2, 2]), -1.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0_f32);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let output = OpaqueSkipLayer::new().propagate_ibp(&input).unwrap();

        assert_eq!(output.shape(), input.shape());
        assert!(output
            .lower()
            .iter()
            .all(|v| v.is_infinite() && v.is_sign_negative()));
        assert!(output
            .upper()
            .iter()
            .all(|v| v.is_infinite() && v.is_sign_positive()));
    }

    /// Declared output shape overrides the input shape for the conservative
    /// substitution (cctsdb_yolo_2023 Concat_120 shape-mismatch fix).
    #[test]
    fn opaque_skip_ibp_uses_declared_output_shape() {
        let lower = ArrayD::from_elem(IxDyn(&[1, 1, 1]), -1.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&[1, 1, 1]), 1.0_f32);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = OpaqueSkipLayer::with_output_shape(vec![1, 12296, 1]);
        let output = layer.propagate_ibp(&input).unwrap();

        assert_eq!(output.shape(), &[1, 12296, 1]);
        assert!(output
            .lower()
            .iter()
            .all(|v| v.is_infinite() && v.is_sign_negative()));
        assert!(output
            .upper()
            .iter()
            .all(|v| v.is_infinite() && v.is_sign_positive()));
    }

    #[test]
    fn opaque_skip_linear_returns_unbounded_bias() {
        let bounds = LinearBounds::identity(3);
        let output = OpaqueSkipLayer::new()
            .propagate_linear(&bounds)
            .unwrap()
            .into_owned();

        assert!(output.lower_a.iter().all(|v| *v == 0.0_f32));
        assert!(output.upper_a.iter().all(|v| *v == 0.0_f32));
        assert!(output
            .lower_b
            .iter()
            .all(|v| v.is_infinite() && v.is_sign_negative()));
        assert!(output
            .upper_b
            .iter()
            .all(|v| v.is_infinite() && v.is_sign_positive()));
    }
}
