// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Element-wise conditional (ONNX Where) for bound propagation.

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;

use crate::bounds::{nan_propagating_max, nan_propagating_min};
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

/// Element-wise conditional: Where(condition, x, y) = x if condition else y.
///
/// For interval bound propagation, since the condition may vary across the
/// input domain, we conservatively take the union (convex hull) of x and y bounds:
/// - lower = min(x_lower, y_lower)
/// - upper = max(x_upper, y_upper)
///
/// This is sound but potentially loose when the condition is deterministically
/// true or false.
///
/// For cases where true_value or false_value are constants (e.g., from ConstantOfShape),
/// they can be embedded in the layer to avoid requiring them as graph inputs.
#[derive(Debug, Clone)]
pub struct WhereLayer {
    /// Optional constant true value (used when ONNX input is a Constant node)
    pub const_true: Option<ArrayD<f32>>,
    /// Optional constant false value (used when ONNX input is a Constant node)
    pub const_false: Option<ArrayD<f32>>,
}

impl WhereLayer {
    /// Create a WhereLayer with no embedded constants (all 3 inputs come from graph).
    pub fn new() -> Self {
        WhereLayer {
            const_true: None,
            const_false: None,
        }
    }

    /// Create a WhereLayer with constant true/false values embedded.
    pub fn with_constants(
        const_true: Option<ArrayD<f32>>,
        const_false: Option<ArrayD<f32>>,
    ) -> Self {
        WhereLayer {
            const_true,
            const_false,
        }
    }

    /// Propagate IBP bounds through element-wise Where.
    ///
    /// Takes three inputs:
    /// - condition: ignored for bounds (condition may vary within input bounds)
    /// - x: bounds for the "true" branch
    /// - y: bounds for the "false" branch
    ///
    /// Returns union of x and y bounds.
    pub fn propagate_ibp_ternary(
        &self,
        _condition: &BoundedTensor,
        x: &BoundedTensor,
        y: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        // Shapes must match or be broadcastable
        // For simplicity, require exact match for now
        if x.shape() != y.shape() {
            return Err(NyError::ShapeMismatch {
                expected: x.shape().to_vec(),
                got: y.shape().to_vec(),
            });
        }

        // Union of bounds: [min(x_lower, y_lower), max(x_upper, y_upper)]
        // Use NaN-propagating variants so upstream NaN is never silently absorbed.
        // IEEE 754 f32::min/max return the non-NaN operand, which would produce
        // unsound (too-tight) bounds. See #2577 for the same fix in Max/MinBinaryLayer.
        let out_lower = ndarray::Zip::from(x.lower())
            .and(y.lower())
            .map_collect(|&xl, &yl| nan_propagating_min(xl, yl));
        let out_upper = ndarray::Zip::from(x.upper())
            .and(y.upper())
            .map_collect(|&xu, &yu| nan_propagating_max(xu, yu));

        BoundedTensor::new(out_lower, out_upper)
    }

    /// Propagate IBP with the condition input and embedded constants.
    ///
    /// Used when true_value and/or false_value are constants embedded in the layer.
    pub fn propagate_ibp_with_condition(&self, condition: &BoundedTensor) -> Result<BoundedTensor> {
        // Get true_value bounds
        let true_bounds = if let Some(ref const_true) = self.const_true {
            BoundedTensor::concrete(const_true.clone())?
        } else {
            return Err(NyError::InvalidSpec(
                "Where: const_true is None but propagate_ibp_with_condition was called".to_string(),
            ));
        };

        // Get false_value bounds
        let false_bounds = if let Some(ref const_false) = self.const_false {
            BoundedTensor::concrete(const_false.clone())?
        } else {
            return Err(NyError::InvalidSpec(
                "Where: const_false is None but propagate_ibp_with_condition was called"
                    .to_string(),
            ));
        };

        // Broadcast to condition shape if needed
        let true_bounds = self.broadcast_to_shape(&true_bounds, condition.shape())?;
        let false_bounds = self.broadcast_to_shape(&false_bounds, condition.shape())?;

        self.propagate_ibp_ternary(condition, &true_bounds, &false_bounds)
    }

    /// Broadcast a tensor to a target shape.
    fn broadcast_to_shape(
        &self,
        tensor: &BoundedTensor,
        target_shape: &[usize],
    ) -> Result<BoundedTensor> {
        if tensor.shape() == target_shape {
            return Ok(tensor.clone());
        }

        // For scalar or single-element tensors, broadcast to target shape
        if tensor.lower().len() == 1 {
            let val_lower = tensor.lower().iter().next().copied().unwrap_or(0.0);
            let val_upper = tensor.upper().iter().next().copied().unwrap_or(0.0);
            let out_lower = ArrayD::from_elem(IxDyn(target_shape), val_lower);
            let out_upper = ArrayD::from_elem(IxDyn(target_shape), val_upper);
            return BoundedTensor::new(out_lower, out_upper);
        }

        // Try numpy-style broadcasting
        let broadcast_lower = tensor
            .lower()
            .broadcast(IxDyn(target_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: target_shape.to_vec(),
                got: tensor.shape().to_vec(),
            })?;
        let broadcast_upper = tensor
            .upper()
            .broadcast(IxDyn(target_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: target_shape.to_vec(),
                got: tensor.shape().to_vec(),
            })?;

        BoundedTensor::new(broadcast_lower.to_owned(), broadcast_upper.to_owned())
    }

    /// Check if this Where layer has embedded constants (for IBP with just condition input).
    pub fn has_embedded_constants(&self) -> bool {
        self.const_true.is_some() && self.const_false.is_some()
    }

    /// Exact (or sound-union) output bounds for an embedded-constant Where, given
    /// the condition's bounds.
    ///
    /// Both branches are constants, so the output is a constant *vector* w.r.t.
    /// the network input — it carries no linear dependence on the upstream that
    /// produced `condition`. The CROWN backward for such a node therefore folds
    /// the entire output into the bias and routes zero coefficients to `cond`.
    /// This method computes the bias bounds to fold in.
    ///
    /// Two regimes, both sound:
    /// - **Constant condition** (`lower == upper` everywhere): the select is
    ///   fully determined per element, so the output is an EXACT constant
    ///   `const_true[i] if cond[i] >= 0.5 else const_false[i]`. This is tighter
    ///   than the IBP union below.
    /// - **Data-dependent condition**: each element may take either branch, so we
    ///   return the per-element interval `[min(t,f), max(t,f)]` — identical to the
    ///   IBP union from [`propagate_ibp_with_condition`]. Sound, not tightened.
    ///
    /// Returns `Err` if the embedded constants are absent or fail to broadcast to
    /// the condition shape (the caller then keeps the existing sound fallback).
    pub fn embedded_constant_select_output(
        &self,
        condition: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let true_bounds = match self.const_true {
            Some(ref c) => BoundedTensor::concrete(c.clone())?,
            None => {
                return Err(NyError::InvalidSpec(
                    "Where::embedded_constant_select_output: const_true is None".to_string(),
                ))
            }
        };
        let false_bounds = match self.const_false {
            Some(ref c) => BoundedTensor::concrete(c.clone())?,
            None => {
                return Err(NyError::InvalidSpec(
                    "Where::embedded_constant_select_output: const_false is None".to_string(),
                ))
            }
        };
        let true_bounds = self.broadcast_to_shape(&true_bounds, condition.shape())?;
        let false_bounds = self.broadcast_to_shape(&false_bounds, condition.shape())?;

        // Constant condition => exact per-element select; else => sound union.
        // We process element-wise so a *partially* constant condition still gets
        // exact selection on the determined positions and the union elsewhere.
        let cond_lower = condition.lower();
        let cond_upper = condition.upper();
        let out_lower = ndarray::Zip::from(cond_lower)
            .and(cond_upper)
            .and(true_bounds.lower())
            .and(false_bounds.lower())
            .map_collect(|&clo, &chi, &tl, &fl| {
                if clo == chi && clo.is_finite() {
                    // Determined select: pick the branch's lower (== value here).
                    if clo >= 0.5 {
                        tl
                    } else {
                        fl
                    }
                } else {
                    // Data-dependent: lower of the union over both branches.
                    nan_propagating_min(tl, fl)
                }
            });
        let out_upper = ndarray::Zip::from(cond_lower)
            .and(cond_upper)
            .and(true_bounds.upper())
            .and(false_bounds.upper())
            .map_collect(|&clo, &chi, &tu, &fu| {
                if clo == chi && clo.is_finite() {
                    if clo >= 0.5 {
                        tu
                    } else {
                        fu
                    }
                } else {
                    nan_propagating_max(tu, fu)
                }
            });
        BoundedTensor::new(out_lower, out_upper)
    }
}

impl Default for WhereLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl BoundPropagation for WhereLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // If we have embedded constants, we can propagate with just the condition input
        if self.has_embedded_constants() {
            return self.propagate_ibp_with_condition(input);
        }
        // Otherwise, Where requires 3 inputs - use propagate_ibp_ternary instead
        Err(NyError::UnsupportedOp(
            "Where requires 3 inputs - use propagate_ibp_ternary".to_string(),
        ))
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "Where is nonlinear - use propagate_ibp_ternary".to_string(),
        ))
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

    /// Exact per-element select when the condition is constant (lower == upper).
    /// `embedded_constant_select_output` must return the degenerate interval equal
    /// to the selected branch constant — tighter than the IBP union.
    #[test]
    fn test_embedded_constant_select_exact_for_constant_condition() {
        let layer = WhereLayer::with_constants(
            Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0_f32, 20.0]).unwrap()),
            Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![-10.0_f32, -20.0]).unwrap()),
        );
        // Constant condition: position 0 -> true, position 1 -> false.
        let cond = make_bounds(&[1.0, 0.0], &[1.0, 0.0]);
        let out = layer.embedded_constant_select_output(&cond).unwrap();
        assert_eq!(out.lower()[[0]], 10.0);
        assert_eq!(out.upper()[[0]], 10.0);
        assert_eq!(out.lower()[[1]], -20.0);
        assert_eq!(out.upper()[[1]], -20.0);
    }

    /// Data-dependent condition (non-degenerate interval) must fall back to the
    /// sound per-element union [min(t,f), max(t,f)] — identical to IBP, never
    /// tighter. Soundness gate for the embedded-constant data-dependent regime.
    #[test]
    fn test_embedded_constant_select_union_for_data_dependent_condition() {
        let layer = WhereLayer::with_constants(
            Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0_f32, 20.0]).unwrap()),
            Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![-10.0_f32, -20.0]).unwrap()),
        );
        // Position 0 data-dependent (interval [0,1]); position 1 constant-false.
        let cond = make_bounds(&[0.0, 0.0], &[1.0, 0.0]);
        let out = layer.embedded_constant_select_output(&cond).unwrap();
        // pos 0: union of {10, -10} -> [-10, 10]; pos 1: exact false -> [-20, -20].
        assert_eq!(out.lower()[[0]], -10.0);
        assert_eq!(out.upper()[[0]], 10.0);
        assert_eq!(out.lower()[[1]], -20.0);
        assert_eq!(out.upper()[[1]], -20.0);
    }

    /// Dense-sampling soundness for `embedded_constant_select_output`: for random
    /// constant 0/1 conditions and random branch constants, the returned interval
    /// must contain the true elementwise select at the (single, since cond is
    /// constant) realized output — exactly, with zero width.
    #[test]
    fn test_embedded_constant_select_dense_sampling_soundness() {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
        for _ in 0..500 {
            let n = rng.random_range(1..=6);
            let cond_vals: Vec<f32> = (0..n)
                .map(|_| if rng.random_bool(0.5) { 1.0 } else { 0.0 })
                .collect();
            let tvals: Vec<f32> = (0..n).map(|_| rng.random_range(-50.0..50.0)).collect();
            let fvals: Vec<f32> = (0..n).map(|_| rng.random_range(-50.0..50.0)).collect();
            let layer = WhereLayer::with_constants(
                Some(ArrayD::from_shape_vec(IxDyn(&[n]), tvals.clone()).unwrap()),
                Some(ArrayD::from_shape_vec(IxDyn(&[n]), fvals.clone()).unwrap()),
            );
            let cond = make_bounds(&cond_vals, &cond_vals); // constant condition
            let out = layer.embedded_constant_select_output(&cond).unwrap();
            for i in 0..n {
                let truth = if cond_vals[i] >= 0.5 {
                    tvals[i]
                } else {
                    fvals[i]
                };
                assert!(
                    truth >= out.lower()[[i]] - 1e-4 && truth <= out.upper()[[i]] + 1e-4,
                    "select[{i}]={truth} not in [{}, {}]",
                    out.lower()[[i]],
                    out.upper()[[i]],
                );
                // Constant condition => exact (zero-width) interval.
                assert!((out.upper()[[i]] - out.lower()[[i]]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_where_ibp_normal_bounds() {
        let layer = WhereLayer::new();
        let cond = make_bounds(&[0.0], &[1.0]);
        let x = make_bounds(&[1.0, 3.0], &[2.0, 5.0]);
        let y = make_bounds(&[0.0, 4.0], &[1.5, 6.0]);

        let result = layer.propagate_ibp_ternary(&cond, &x, &y).unwrap();
        // Union: lower = min(x_l, y_l), upper = max(x_u, y_u)
        assert_eq!(result.lower()[[0]], 0.0); // min(1.0, 0.0)
        assert_eq!(result.lower()[[1]], 3.0); // min(3.0, 4.0)
        assert_eq!(result.upper()[[0]], 2.0); // max(2.0, 1.5)
        assert_eq!(result.upper()[[1]], 6.0); // max(5.0, 6.0)
    }

    /// Regression test (#2853): NaN in x's lower bound must propagate through to
    /// output, causing BoundedTensor::new to reject the result. Before the fix,
    /// f32::min silently returned the non-NaN operand (IEEE 754), producing
    /// unsound (too-tight) bounds. nan_propagating_min preserves NaN so the
    /// output constructor catches it.
    ///
    /// Uses new_unchecked to bypass BoundedTensor's NaN rejection at construction,
    /// simulating upstream computational NaN (e.g., inf - inf). The previous
    /// version of this test used new_allow_infinite which rejects NaN, so the
    /// test never reached propagate_ibp_ternary.
    #[test]
    fn test_where_ibp_nan_propagates_from_x_lower() {
        let layer = WhereLayer::new();
        let cond = make_bounds(&[0.0], &[1.0]);
        // Construct BoundedTensor with NaN in lower bound via unchecked constructor.
        let x_l = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 1.0]).unwrap();
        let x_u = ArrayD::from_shape_vec(IxDyn(&[2]), vec![5.0, 2.0]).unwrap();
        let x = BoundedTensor::new_unchecked(x_l, x_u).unwrap();
        let y = make_bounds(&[0.0, 3.0], &[1.0, 4.0]);

        let result = layer.propagate_ibp_ternary(&cond, &x, &y);
        // nan_propagating_min(NaN, 0.0) = NaN, which makes BoundedTensor::new
        // reject the output. This is the correct defense-in-depth: NaN propagates
        // through the computation and is caught at the output constructor.
        assert!(
            result.is_err(),
            "NaN in x lower should propagate through nan_propagating_min \
             and be rejected by BoundedTensor::new, but got Ok"
        );
    }

    /// Regression test (#2853): NaN in y's upper bound must propagate.
    /// Same mechanism as the x_lower test above. See #2577.
    #[test]
    fn test_where_ibp_nan_propagates_from_y_upper() {
        let layer = WhereLayer::new();
        let cond = make_bounds(&[0.0], &[1.0]);
        let x = make_bounds(&[1.0, 2.0], &[3.0, 4.0]);
        let y_l = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).unwrap();
        let y_u = ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, f32::NAN]).unwrap();
        let y = BoundedTensor::new_unchecked(y_l, y_u).unwrap();

        let result = layer.propagate_ibp_ternary(&cond, &x, &y);
        // nan_propagating_max(4.0, NaN) = NaN, rejected by BoundedTensor::new.
        assert!(
            result.is_err(),
            "NaN in y upper should propagate through nan_propagating_max \
             and be rejected by BoundedTensor::new, but got Ok"
        );
    }

    /// Verify that without nan_propagating_min/max, NaN would be silently
    /// absorbed. This is the negative-control test: f32::min/max absorb NaN,
    /// producing a finite (unsound) result.
    #[test]
    fn test_where_f32_min_max_absorbs_nan_negative_control() {
        // IEEE 754: NaN.min(x) = x, NaN.max(x) = x
        assert_eq!(f32::NAN.min(1.0), 1.0);
        assert_eq!(f32::NAN.max(1.0), 1.0);
        assert_eq!(1.0_f32.min(f32::NAN), 1.0);
        assert_eq!(1.0_f32.max(f32::NAN), 1.0);

        // Contrast with nan_propagating variants:
        assert!(nan_propagating_min(f32::NAN, 1.0).is_nan());
        assert!(nan_propagating_max(f32::NAN, 1.0).is_nan());
        assert!(nan_propagating_min(1.0, f32::NAN).is_nan());
        assert!(nan_propagating_max(1.0, f32::NAN).is_nan());
    }
}
