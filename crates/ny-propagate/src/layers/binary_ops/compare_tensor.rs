// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CompareTensor layer: element-wise comparison of two bounded tensors.
//!
//! Output is {0.0, 1.0}. IBP-only — no meaningful CROWN linear relaxation
//! exists for binary comparison (same pattern as WhereLayer).
//!
//! Reference: design doc `designs/2026-03-21-issue-4269-compare-propagation-layer.md`
//! Supports transformer masking ops for external verifier consumers.

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::layers::misc::CompareOp;

/// Element-wise comparison of two bounded tensors.
///
/// Output is in {0.0, 1.0}. For IBP, we use cross-bounds analysis:
/// compare bounds of input A against bounds of input B.
///
/// CROWN backward returns UnsupportedOp — no meaningful linear relaxation
/// for binary comparison. Same pattern as WhereLayer.
#[derive(Debug, Clone)]
pub struct CompareTensorLayer {
    pub op: CompareOp,
}

impl CompareTensorLayer {
    pub fn new(op: CompareOp) -> Self {
        CompareTensorLayer { op }
    }

    /// Propagate IBP bounds through element-wise tensor comparison.
    ///
    /// For inputs A in [a_l, a_u] and B in [b_l, b_u]:
    ///
    /// | Op  | Output [1, 1]        | Output [0, 0]        | Output [0, 1] |
    /// |-----|----------------------|----------------------|----------------|
    /// | Gt  | `a_l > b_u`          | `a_u <= b_l`         | otherwise      |
    /// | Ge  | `a_l >= b_u`         | `a_u < b_l`          | otherwise      |
    /// | Lt  | `a_u < b_l`          | `a_l >= b_u`         | otherwise      |
    /// | Le  | `a_u <= b_l`         | `a_l > b_u`          | otherwise      |
    /// | Eq  | all bounds identical | intervals disjoint   | otherwise      |
    /// | Ne  | intervals disjoint   | all bounds identical  | otherwise      |
    pub fn propagate_ibp_binary(
        &self,
        input_a: &BoundedTensor,
        input_b: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // ONNX Equal/Less/Greater use NumPy multidirectional broadcasting
        // (e.g. cctsdb Equal_200 compares an ArgMax [1] against a scalar []).
        // Broadcasting replicates values verbatim, so comparing the broadcast
        // views elementwise is exact.
        let output_shape = crate::shape::broadcast_shapes(input_a.shape(), input_b.shape())
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: input_a.shape().to_vec(),
                got: input_b.shape().to_vec(),
            })?;
        let broadcast_err = || NyError::ShapeMismatch {
            expected: output_shape.clone(),
            got: input_a.shape().to_vec(),
        };
        let a_lower = input_a
            .lower()
            .broadcast(IxDyn(&output_shape))
            .ok_or_else(broadcast_err)?;
        let a_upper = input_a
            .upper()
            .broadcast(IxDyn(&output_shape))
            .ok_or_else(broadcast_err)?;
        let b_lower = input_b
            .lower()
            .broadcast(IxDyn(&output_shape))
            .ok_or_else(broadcast_err)?;
        let b_upper = input_b
            .upper()
            .broadcast(IxDyn(&output_shape))
            .ok_or_else(broadcast_err)?;

        let op = self.op;
        let mut out_lower = ArrayD::zeros(IxDyn(&output_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(&output_shape));

        for (idx, &a_l) in a_lower.indexed_iter() {
            let a_u = a_upper[idx.clone()];
            let b_l = b_lower[idx.clone()];
            let b_u = b_upper[idx.clone()];
            let (ol, ou) = compare_tensor_interval(a_l, a_u, b_l, b_u, op);
            out_lower[idx.clone()] = ol;
            out_upper[idx] = ou;
        }

        BoundedTensor::new(out_lower, out_upper)
    }
}

/// Compute output interval bounds for tensor-vs-tensor comparison.
fn compare_tensor_interval(a_l: f32, a_u: f32, b_l: f32, b_u: f32, op: CompareOp) -> (f32, f32) {
    match op {
        CompareOp::Gt => {
            if a_l > b_u {
                (1.0, 1.0)
            } else if a_u <= b_l {
                (0.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        }
        CompareOp::Ge => {
            if a_l >= b_u {
                (1.0, 1.0)
            } else if a_u < b_l {
                (0.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        }
        CompareOp::Lt => {
            if a_u < b_l {
                (1.0, 1.0)
            } else if a_l >= b_u {
                (0.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        }
        CompareOp::Le => {
            if a_u <= b_l {
                (1.0, 1.0)
            } else if a_l > b_u {
                (0.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        }
        CompareOp::Eq => {
            if a_l == a_u && b_l == b_u && a_l == b_l {
                (1.0, 1.0)
            } else if a_l > b_u || a_u < b_l {
                // Intervals are completely disjoint
                (0.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        }
        CompareOp::Ne => {
            if a_l > b_u || a_u < b_l {
                // Intervals are completely disjoint — always not-equal
                (1.0, 1.0)
            } else if a_l == a_u && b_l == b_u && a_l == b_l {
                // Both concrete and identical
                (0.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    fn make_bounds(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        let l = ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).unwrap();
        BoundedTensor::new(l, u).unwrap()
    }

    #[test]
    fn test_compare_tensor_gt_determined_true() {
        let layer = CompareTensorLayer::new(CompareOp::Gt);
        let a = make_bounds(&[5.0, 10.0], &[6.0, 12.0]);
        let b = make_bounds(&[1.0, 3.0], &[4.0, 11.0]); // a_l > b_u for first element
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        assert_eq!(result.lower()[[0]], 1.0); // 5 > 4: a_l=5 > b_u=4, deterministic
        assert_eq!(result.upper()[[0]], 1.0);
        assert_eq!(result.lower()[[1]], 0.0); // 10..12 vs 3..11: overlapping (a_l=10 < b_u=11)
        assert_eq!(result.upper()[[1]], 1.0);
    }

    #[test]
    fn test_compare_tensor_gt_determined_false() {
        let layer = CompareTensorLayer::new(CompareOp::Gt);
        let a = make_bounds(&[1.0], &[3.0]);
        let b = make_bounds(&[3.0], &[5.0]); // a_u <= b_l
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        assert_eq!(result.lower()[[0]], 0.0);
        assert_eq!(result.upper()[[0]], 0.0);
    }

    #[test]
    fn test_compare_tensor_eq_concrete_match() {
        let layer = CompareTensorLayer::new(CompareOp::Eq);
        let a = make_bounds(&[3.0], &[3.0]);
        let b = make_bounds(&[3.0], &[3.0]);
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        assert_eq!(result.lower()[[0]], 1.0);
        assert_eq!(result.upper()[[0]], 1.0);
    }

    #[test]
    fn test_compare_tensor_eq_disjoint() {
        let layer = CompareTensorLayer::new(CompareOp::Eq);
        let a = make_bounds(&[5.0], &[6.0]);
        let b = make_bounds(&[1.0], &[3.0]); // disjoint
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        assert_eq!(result.lower()[[0]], 0.0);
        assert_eq!(result.upper()[[0]], 0.0);
    }

    #[test]
    fn test_compare_tensor_shape_mismatch() {
        // Non-broadcastable shapes ([2] vs [3]) must error.
        let layer = CompareTensorLayer::new(CompareOp::Gt);
        let a = make_bounds(&[1.0, 2.0], &[3.0, 4.0]);
        let b = make_bounds(&[1.0, 1.0, 1.0], &[2.0, 2.0, 2.0]);
        let result = layer.propagate_ibp_binary(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_compare_tensor_broadcast_vector_vs_scalar() {
        // ONNX broadcast: [2] vs [] compares each element against the scalar
        // (cctsdb Equal_200: ArgMax [1] vs ground-truth label scalar []).
        let layer = CompareTensorLayer::new(CompareOp::Eq);
        let a = make_bounds(&[1.0, 3.0], &[1.0, 3.0]);
        let scalar_l = ArrayD::from_elem(IxDyn(&[]), 1.0_f32);
        let scalar_u = ArrayD::from_elem(IxDyn(&[]), 1.0_f32);
        let b = BoundedTensor::new(scalar_l, scalar_u).unwrap();
        let result = layer.propagate_ibp_binary(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.lower()[[0]], 1.0); // 1 == 1 definitely
        assert_eq!(result.upper()[[0]], 1.0);
        assert_eq!(result.lower()[[1]], 0.0); // 3 != 1 definitely
        assert_eq!(result.upper()[[1]], 0.0);
    }
}
