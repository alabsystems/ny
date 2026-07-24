// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::bounds::LinearBounds;

/// Shared flat-dimension and tensor-shape contract for graph CROWN targets.
#[derive(Debug, Clone)]
pub(crate) struct GraphTargetShapeContract {
    node_name: String,
    tensor_shape: Vec<usize>,
    flat_dim: usize,
}

impl GraphTargetShapeContract {
    pub(crate) fn from_bounds(node_name: &str, bounds: &BoundedTensor) -> Self {
        Self {
            node_name: node_name.to_string(),
            tensor_shape: bounds.shape().to_vec(),
            flat_dim: bounds.len(),
        }
    }

    pub(crate) fn flat_dim(&self) -> usize {
        self.flat_dim
    }

    pub(crate) fn identity_linear_bounds(&self) -> LinearBounds {
        LinearBounds::identity(self.flat_dim)
    }

    pub(crate) fn restore_concrete(
        &self,
        bounds: BoundedTensor,
        context: &'static str,
    ) -> Result<BoundedTensor> {
        self.restore_to_shape(bounds, &self.tensor_shape, context)
    }

    pub(crate) fn reshape_for_forward_match(
        &self,
        candidate: BoundedTensor,
        forward: &BoundedTensor,
        context: &'static str,
    ) -> Result<BoundedTensor> {
        self.restore_to_shape(candidate, forward.shape(), context)
    }

    pub(crate) fn validate_spec_cols(
        &self,
        spec_cols: usize,
        _context: &'static str,
    ) -> Result<()> {
        if spec_cols == self.flat_dim {
            Ok(())
        } else {
            Err(NyError::shape_mismatch(
                vec![self.flat_dim],
                vec![spec_cols],
            ))
        }
    }

    fn restore_to_shape(
        &self,
        bounds: BoundedTensor,
        expected_shape: &[usize],
        context: &'static str,
    ) -> Result<BoundedTensor> {
        if bounds.shape() == expected_shape {
            return Ok(bounds);
        }
        let expected_len = checked_shape_product(expected_shape).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "{context} for node '{}' expected shape product overflows: {:?}",
                self.node_name, expected_shape
            ))
        })?;
        if bounds.len() != expected_len {
            return Err(NyError::shape_mismatch(
                expected_shape.to_vec(),
                bounds.shape().to_vec(),
            ));
        }
        let got_shape = bounds.shape().to_vec();
        bounds.reshape(expected_shape).map_err(|err| {
            NyError::InvalidSpec(format!(
                "{context} for node '{}' failed to reshape {:?} to {:?}: {err}",
                self.node_name, got_shape, expected_shape
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    #[test]
    fn test_restore_to_shape_rejects_expected_shape_overflow_3012() {
        let bounds = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0_f32]).expect("lower"),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0_f32]).expect("upper"),
        )
        .expect("valid bounds");
        let contract = GraphTargetShapeContract::from_bounds("overflow-node", &bounds);
        let err = contract
            .restore_to_shape(bounds, &[2, (usize::MAX / 2) + 1], "restore")
            .expect_err("overflowing expected shape should fail");

        assert!(
            matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("expected shape product overflows")),
            "expected expected-shape overflow error, got: {err:?}"
        );
    }
}
