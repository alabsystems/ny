// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Squeeze and Unsqueeze layers for bound propagation.

use ndarray::Axis;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;

use super::super::common::BoundPropagation;
use crate::LinearBounds;

/// Squeeze layer: removes dimensions of size 1 at specified axis.
#[derive(Debug, Clone)]
pub struct SqueezeLayer {
    pub axis: i32,
    input_shape: Option<Vec<usize>>,
}

impl SqueezeLayer {
    /// Create a new Squeeze layer that removes the dimension at `axis`.
    pub fn new(axis: i32) -> Self {
        Self {
            axis,
            input_shape: None,
        }
    }

    /// Set the expected input shape for axis normalization.
    pub fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.input_shape = Some(shape);
    }

    fn normalize_axis(&self, ndim: usize) -> Result<usize> {
        let axis = super::super::common::resolve_axis_i32(self.axis, ndim, "Squeeze")?;
        // Axis 0 normally targets the (stripped) batch dimension and is forbidden.
        // Exception: a rank-1 unbatched tensor (`ndim == 1`) never carried a batch
        // dimension — its sole axis is a genuine size-1 data axis (the converter only
        // emits `Squeeze(axis=0)` here when the recorded ONNX shape was rank ≤ 1, the
        // `data_had_batch_axis == Some(false)` convention). Removing it is a pure
        // shape op (`remove_axis`) with no bound math, so it is sound. For `ndim > 1`,
        // axis 0 is the batch dimension and stays forbidden.
        if axis == 0 && ndim > 1 {
            return Err(NyError::InvalidSpec("Squeeze axis 0 forbidden".to_string()));
        }
        Ok(axis)
    }
}

impl BoundPropagation for SqueezeLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let axis = self.normalize_axis(input.shape().len())?;
        if input.shape()[axis] != 1 {
            return Err(NyError::InvalidSpec(format!(
                "Squeeze axis {} has size {}, expected 1",
                axis,
                input.shape()[axis]
            )));
        }
        let lower = input.lower().clone().remove_axis(Axis(axis));
        let upper = input.upper().clone().remove_axis(Axis(axis));
        // Pure layout op (axis removal): output values are exactly the input
        // values reindexed, so infinite bounds pass through soundly. Using
        // `new_allow_infinite` keeps the firewall from rejecting a sound `±inf`
        // that flowed in from a skipped/opaque upstream op (e.g. ScatterND index
        // subgraphs in the cctsdb_yolo detection head). NaN is still rejected.
        BoundedTensor::new_allow_infinite(lower, upper)
    }

    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Ok(Cow::Borrowed(bounds))
    }
}

/// Unsqueeze layer: inserts dimension of size 1 at specified axis.
#[derive(Debug, Clone)]
pub struct UnsqueezeLayer {
    pub axis: i32,
    input_shape: Option<Vec<usize>>,
}

impl UnsqueezeLayer {
    /// Create a new Unsqueeze layer that inserts a dimension at `axis`.
    pub fn new(axis: i32) -> Self {
        Self {
            axis,
            input_shape: None,
        }
    }

    /// Set the expected input shape for axis normalization.
    pub fn set_input_shape(&mut self, shape: Vec<usize>) {
        self.input_shape = Some(shape);
    }

    fn normalize_axis(&self, input_ndim: usize) -> Result<usize> {
        // Unsqueeze resolves against output_ndim = input_ndim + 1 per ONNX spec
        let output_ndim = input_ndim + 1;
        let axis = super::super::common::resolve_axis_i32(self.axis, output_ndim, "Unsqueeze")?;
        // axis=0 is allowed — used in shape-computation subgraphs (lsnc quadrotor2d_output)
        // where Unsqueeze inserts a leading dimension on scalar/1D data, not a batch dim
        Ok(axis)
    }
}

impl BoundPropagation for UnsqueezeLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let axis = self.normalize_axis(input.shape().len())?;
        let lower = input.lower().clone().insert_axis(Axis(axis));
        let upper = input.upper().clone().insert_axis(Axis(axis));
        // Pure layout op (axis insertion): value-preserving, so infinite bounds
        // pass through soundly (see Squeeze note). NaN is still rejected.
        BoundedTensor::new_allow_infinite(lower, upper)
    }

    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Ok(Cow::Borrowed(bounds))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    /// Pure layout ops (Unsqueeze/Squeeze) must pass `±inf` bounds through
    /// unchanged rather than tripping the NaN/Inf firewall. This is the cctsdb_yolo
    /// regression: a skipped/opaque upstream op (ScatterND index subgraph) produced
    /// a sound `[-inf, +inf]`, and `Unsqueeze` then aborted the whole verification
    /// with "lower bounds contain NaN or Inf". Infinite bounds are a sound
    /// conservative over-approximation; only NaN is genuinely corrupt.
    #[test]
    fn unsqueeze_passes_infinite_bounds_through() {
        let lower = ArrayD::from_elem(IxDyn(&[3]), f32::NEG_INFINITY);
        let upper = ArrayD::from_elem(IxDyn(&[3]), f32::INFINITY);
        let input = BoundedTensor::new_allow_infinite(lower, upper).unwrap();
        let out = UnsqueezeLayer::new(0).propagate_ibp(&input).unwrap();
        assert_eq!(out.shape(), &[1, 3]);
        assert!(out
            .lower()
            .iter()
            .all(|v| v.is_infinite() && v.is_sign_negative()));
        assert!(out
            .upper()
            .iter()
            .all(|v| v.is_infinite() && v.is_sign_positive()));
    }

    /// Mixed finite/infinite bounds also pass through (only the opaque element is
    /// unbounded), preserving the finite information.
    #[test]
    fn unsqueeze_passes_mixed_finite_and_infinite() {
        let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, f32::NEG_INFINITY]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, f32::INFINITY]).unwrap();
        let input = BoundedTensor::new_allow_infinite(lower, upper).unwrap();
        let out = UnsqueezeLayer::new(1).propagate_ibp(&input).unwrap();
        assert_eq!(out.shape(), &[2, 1]);
        assert_eq!(out.lower().as_slice().unwrap()[0], -1.0);
        assert!(out.lower().as_slice().unwrap()[1].is_infinite());
    }

    /// Squeeze likewise passes `±inf` through a size-1 axis removal soundly.
    /// (Squeeze a non-batch trailing axis on a `[4, 1]` tensor.)
    #[test]
    fn squeeze_passes_infinite_bounds_through() {
        let lower = ArrayD::from_elem(IxDyn(&[4, 1]), f32::NEG_INFINITY);
        let upper = ArrayD::from_elem(IxDyn(&[4, 1]), f32::INFINITY);
        let input = BoundedTensor::new_allow_infinite(lower, upper).unwrap();
        let out = SqueezeLayer::new(1).propagate_ibp(&input).unwrap();
        assert_eq!(out.shape(), &[4]);
        assert!(out.lower().iter().all(|v| v.is_infinite()));
    }
}
