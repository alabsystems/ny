// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Compare layer: element-wise comparison (tensor vs scalar threshold).
//!
//! Output is {0.0, 1.0}. IBP uses paired bounds analysis; CROWN uses
//! zero-slope relaxation (piecewise constant, like Sign/Floor/Ceil).
//!
//! Reference: design doc `designs/2026-03-21-issue-4269-compare-propagation-layer.md`
//! Supports transformer masking ops for external verifier consumers.

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;
use std::fmt;
use std::str::FromStr;
use tracing::debug;

use crate::layers::activations::LinearRelaxation;
use crate::layers::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, non_finite_domain_guard, BoundPropagation,
};
use crate::{BatchedLinearBounds, LinearBounds};

/// Comparison operator for element-wise compare layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareOp {
    /// x > threshold
    Gt,
    /// x >= threshold
    Ge,
    /// x < threshold
    Lt,
    /// x <= threshold
    Le,
    /// x == threshold
    Eq,
    /// x != threshold
    Ne,
}

impl fmt::Display for CompareOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompareOp::Gt => write!(f, "Gt"),
            CompareOp::Ge => write!(f, "Ge"),
            CompareOp::Lt => write!(f, "Lt"),
            CompareOp::Le => write!(f, "Le"),
            CompareOp::Eq => write!(f, "Eq"),
            CompareOp::Ne => write!(f, "Ne"),
        }
    }
}

impl FromStr for CompareOp {
    type Err = NyError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "Gt" | "gt" | ">" => Ok(CompareOp::Gt),
            "Ge" | "ge" | ">=" => Ok(CompareOp::Ge),
            "Lt" | "lt" | "<" => Ok(CompareOp::Lt),
            "Le" | "le" | "<=" => Ok(CompareOp::Le),
            "Eq" | "eq" | "==" => Ok(CompareOp::Eq),
            "Ne" | "ne" | "!=" => Ok(CompareOp::Ne),
            _ => Err(NyError::InvalidSpec(format!(
                "Unknown comparison operator: {s}"
            ))),
        }
    }
}

/// Element-wise comparison layer (tensor vs scalar threshold).
///
/// Output is in {0.0, 1.0}. For IBP, we use paired bounds analysis
/// to determine whether the output is definitively 0, definitively 1,
/// or could be either.
///
/// CROWN relaxation uses zero-slope (same pattern as Sign/Floor/Ceil):
/// piecewise constant functions have zero derivative almost everywhere.
#[derive(Debug, Clone)]
pub struct CompareLayer {
    pub threshold: f32,
    pub op: CompareOp,
}

impl CompareLayer {
    pub fn new(threshold: f32, op: CompareOp) -> Self {
        CompareLayer { threshold, op }
    }
}

/// Compute output interval bounds for a comparison operation.
///
/// Given input bounds [l, u] and threshold t, returns (out_lower, out_upper)
/// where output is in {0.0, 1.0}.
///
/// | Op  | Output [1, 1]      | Output [0, 0]      | Output [0, 1]  |
/// |-----|--------------------|---------------------|----------------|
/// | Gt  | `l > t`            | `u <= t`            | otherwise      |
/// | Ge  | `l >= t`           | `u < t`             | otherwise      |
/// | Lt  | `u < t`            | `l >= t`            | otherwise      |
/// | Le  | `u <= t`           | `l > t`             | otherwise      |
/// | Eq  | `l == u == t`      | `l > t` or `u < t`  | otherwise      |
/// | Ne  | `l > t` or `u < t` | `l == u == t`       | otherwise      |
fn compare_interval_bounds(l: f32, u: f32, t: f32, op: CompareOp) -> (f32, f32) {
    match op {
        CompareOp::Gt => {
            if l > t {
                (1.0, 1.0)
            } else if u <= t {
                (0.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        }
        CompareOp::Ge => {
            if l >= t {
                (1.0, 1.0)
            } else if u < t {
                (0.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        }
        CompareOp::Lt => {
            if u < t {
                (1.0, 1.0)
            } else if l >= t {
                (0.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        }
        CompareOp::Le => {
            if u <= t {
                (1.0, 1.0)
            } else if l > t {
                (0.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        }
        CompareOp::Eq => {
            if l == t && u == t {
                (1.0, 1.0)
            } else if l > t || u < t {
                (0.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        }
        CompareOp::Ne => {
            if l > t || u < t {
                (1.0, 1.0)
            } else if l == t && u == t {
                (0.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        }
    }
}

/// CROWN linear relaxation for Compare.
///
/// Piecewise-constant: zero slope in the general case.
/// Determined case (constant output across entire domain): exact.
/// Undetermined case (straddles threshold): lower=0, upper=1 (IBP fallback).
pub(crate) fn compare_crown_relaxation(l: f32, u: f32, t: f32, op: CompareOp) -> LinearRelaxation {
    let (out_l, out_u) = compare_interval_bounds(l, u, t, op);
    // Determined case: both bounds equal, exact relaxation
    // slope = 0, intercept = constant value
    LinearRelaxation::new(0.0, out_l, 0.0, out_u)
}

impl BoundPropagation for CompareLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let t = self.threshold;
        let op = self.op;
        let mut out_lower = ArrayD::zeros(IxDyn(input.shape()));
        let mut out_upper = ArrayD::zeros(IxDyn(input.shape()));
        for (idx, &lb) in input.lower().indexed_iter() {
            let ub = input.upper()[idx.clone()];
            let (ol, ou) = compare_interval_bounds(lb, ub, t, op);
            out_lower[idx.clone()] = ol;
            out_upper[idx] = ou;
        }
        BoundedTensor::new(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "Compare is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
                .to_string(),
        ))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        self.propagate_linear_with_bounds_impl(bounds, pre_activation)
    }
}

impl CompareLayer {
    fn propagate_linear_with_bounds_impl(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        non_finite_domain_guard("Compare", pre_activation)?;
        debug!("Compare layer CROWN backward propagation with pre-activation bounds");
        let t = self.threshold;
        let op = self.op;
        crown_elementwise_backward(bounds, pre_activation, |l, u| {
            compare_crown_relaxation(l, u, t, op)
        })
    }

    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        non_finite_domain_guard("Compare", pre_activation)?;
        debug!("Compare layer batched CROWN backward propagation");
        let t = self.threshold;
        let op = self.op;
        crown_elementwise_backward_batched(bounds, pre_activation, |l, u| {
            compare_crown_relaxation(l, u, t, op)
        })
    }

    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("Compare", pre_activation)?;
        let t = self.threshold;
        let op = self.op;
        crown_elementwise_backward_patches(bounds, pre_activation, |l, u| {
            compare_crown_relaxation(l, u, t, op)
        })
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
    fn test_compare_gt_determined_true() {
        let layer = CompareLayer::new(2.0, CompareOp::Gt);
        let input = make_bounds(&[3.0, 5.0], &[4.0, 6.0]); // all > 2.0
        let result = layer.propagate_ibp(&input).unwrap();
        assert_eq!(result.lower()[[0]], 1.0);
        assert_eq!(result.lower()[[1]], 1.0);
        assert_eq!(result.upper()[[0]], 1.0);
        assert_eq!(result.upper()[[1]], 1.0);
    }

    #[test]
    fn test_compare_gt_determined_false() {
        let layer = CompareLayer::new(5.0, CompareOp::Gt);
        let input = make_bounds(&[1.0, 2.0], &[3.0, 5.0]); // all <= 5.0
        let result = layer.propagate_ibp(&input).unwrap();
        assert_eq!(result.lower()[[0]], 0.0);
        assert_eq!(result.lower()[[1]], 0.0);
        assert_eq!(result.upper()[[0]], 0.0);
        assert_eq!(result.upper()[[1]], 0.0);
    }

    #[test]
    fn test_compare_gt_undetermined() {
        let layer = CompareLayer::new(3.0, CompareOp::Gt);
        let input = make_bounds(&[1.0], &[5.0]); // straddles threshold
        let result = layer.propagate_ibp(&input).unwrap();
        assert_eq!(result.lower()[[0]], 0.0);
        assert_eq!(result.upper()[[0]], 1.0);
    }

    #[test]
    fn test_compare_le_determined() {
        let layer = CompareLayer::new(5.0, CompareOp::Le);
        let input = make_bounds(&[1.0, 2.0], &[3.0, 5.0]); // all <= 5
        let result = layer.propagate_ibp(&input).unwrap();
        assert_eq!(result.lower()[[0]], 1.0);
        assert_eq!(result.upper()[[0]], 1.0);
        assert_eq!(result.lower()[[1]], 1.0);
        assert_eq!(result.upper()[[1]], 1.0);
    }

    #[test]
    fn test_compare_eq_point_match() {
        let layer = CompareLayer::new(3.0, CompareOp::Eq);
        // Concrete point exactly at threshold
        let input = make_bounds(&[3.0], &[3.0]);
        let result = layer.propagate_ibp(&input).unwrap();
        assert_eq!(result.lower()[[0]], 1.0);
        assert_eq!(result.upper()[[0]], 1.0);
    }

    #[test]
    fn test_compare_eq_disjoint() {
        let layer = CompareLayer::new(3.0, CompareOp::Eq);
        let input = make_bounds(&[4.0], &[5.0]); // entirely above threshold
        let result = layer.propagate_ibp(&input).unwrap();
        assert_eq!(result.lower()[[0]], 0.0);
        assert_eq!(result.upper()[[0]], 0.0);
    }

    #[test]
    fn test_compare_ne_disjoint() {
        let layer = CompareLayer::new(3.0, CompareOp::Ne);
        let input = make_bounds(&[4.0], &[5.0]); // entirely above, never equal
        let result = layer.propagate_ibp(&input).unwrap();
        assert_eq!(result.lower()[[0]], 1.0);
        assert_eq!(result.upper()[[0]], 1.0);
    }

    #[test]
    fn test_compare_op_from_str() {
        assert_eq!(CompareOp::from_str("Gt").unwrap(), CompareOp::Gt);
        assert_eq!(CompareOp::from_str(">=").unwrap(), CompareOp::Ge);
        assert_eq!(CompareOp::from_str("lt").unwrap(), CompareOp::Lt);
        assert!(CompareOp::from_str("invalid").is_err());
    }

    #[test]
    fn test_compare_crown_relaxation_determined() {
        // Fully determined: l > t → output is always 1
        let relax = compare_crown_relaxation(5.0, 10.0, 3.0, CompareOp::Gt);
        assert_eq!(relax.lower_slope, 0.0);
        assert_eq!(relax.lower_intercept, 1.0);
        assert_eq!(relax.upper_slope, 0.0);
        assert_eq!(relax.upper_intercept, 1.0);
    }

    #[test]
    fn test_compare_crown_relaxation_undetermined() {
        // Undetermined: straddles threshold → [0, 1]
        let relax = compare_crown_relaxation(1.0, 5.0, 3.0, CompareOp::Gt);
        assert_eq!(relax.lower_slope, 0.0);
        assert_eq!(relax.lower_intercept, 0.0);
        assert_eq!(relax.upper_slope, 0.0);
        assert_eq!(relax.upper_intercept, 1.0);
    }
}
