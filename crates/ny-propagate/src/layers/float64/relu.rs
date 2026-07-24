// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Double-precision ReLU layer: y = max(0, x).
//!
//! IBP: element-wise max(0, bound).
//! CROWN backward: standard triangle relaxation.
//!
//! Reference: alpha-beta-CROWN `auto_lirpa/operators/activations.py`.

use ny_core::Result;
use ny_tensor::BoundedTensor64;

use crate::bounds::LinearBounds64;

/// IBP propagation for ReLU in f64.
///
/// lower = max(0, x_lower), upper = max(0, x_upper).
pub(crate) fn propagate_relu_ibp_f64(input: &BoundedTensor64) -> Result<BoundedTensor64> {
    let lower = input.lower().mapv(|x| x.max(0.0));
    let upper = input.upper().mapv(|x| x.max(0.0));
    BoundedTensor64::new(lower, upper)
}

/// Linear relaxation parameters for a single ReLU neuron in f64.
struct RelaxationF64 {
    lower_slope: f64,
    lower_intercept: f64,
    upper_slope: f64,
    upper_intercept: f64,
}

/// Compute the triangle relaxation for a single ReLU neuron.
///
/// Three cases:
/// - l >= 0: identity (pass-through), slope = 1, intercept = 0
/// - u <= 0: zero (dead neuron), slope = 0, intercept = 0
/// - l < 0 < u (crossing): triangle relaxation
///   - Upper: lambda = u / (u - l), intercept = -lambda * l
///   - Lower: heuristic — alpha = 1.0 if u > -l, else 0.0 (area-based)
///
/// Reference: alpha-beta-CROWN `auto_lirpa/operators/relu.py:bound_relax()`
fn relu_relaxation_f64(l: f64, u: f64) -> RelaxationF64 {
    if l >= 0.0 {
        // Entirely positive: identity
        RelaxationF64 {
            lower_slope: 1.0,
            lower_intercept: 0.0,
            upper_slope: 1.0,
            upper_intercept: 0.0,
        }
    } else if u <= 0.0 {
        // Entirely negative: zero
        RelaxationF64 {
            lower_slope: 0.0,
            lower_intercept: 0.0,
            upper_slope: 0.0,
            upper_intercept: 0.0,
        }
    } else {
        // Crossing: l < 0 < u
        let lambda = u / (u - l);
        let upper_intercept = -lambda * l;

        // Lower bound heuristic: use the slope that gives tighter area.
        // If u > |l|, use slope=1 (pass-through); otherwise slope=0 (zero).
        let alpha = if u > -l { 1.0 } else { 0.0 };

        RelaxationF64 {
            lower_slope: alpha,
            lower_intercept: 0.0,
            upper_slope: lambda,
            upper_intercept,
        }
    }
}

/// CROWN backward propagation for ReLU in f64.
///
/// Composes the incoming linear bounds with per-neuron relaxation:
/// For each output row i and neuron j:
///   If A[i,j] >= 0: use upper relaxation for upper bound, lower for lower
///   If A[i,j] < 0:  use lower relaxation for upper bound, upper for lower
///
/// Reference: alpha-beta-CROWN `auto_lirpa/operators/relu.py:bound_backward()`
pub(crate) fn propagate_relu_crown_backward_f64(
    bounds: &LinearBounds64,
    pre_activation: &BoundedTensor64,
) -> Result<LinearBounds64> {
    let m = bounds.num_outputs();
    let n = bounds.num_inputs();

    let (pre_l, pre_u) = pre_activation.flatten_to_1d();
    if pre_l.len() != n {
        return Err(ny_core::NyError::shape_mismatch(vec![n], vec![pre_l.len()]));
    }

    // Precompute per-neuron relaxation
    let relaxations: Vec<RelaxationF64> = (0..n)
        .map(|j| relu_relaxation_f64(pre_l[j], pre_u[j]))
        .collect();

    let mut new_lower_a = ndarray::Array2::<f64>::zeros((m, n));
    let mut new_upper_a = ndarray::Array2::<f64>::zeros((m, n));
    let mut new_lower_b = bounds.lower_b().clone();
    let mut new_upper_b = bounds.upper_b().clone();

    for i in 0..m {
        for j in 0..n {
            let la = bounds.lower_a()[[i, j]];
            let ua = bounds.upper_a()[[i, j]];
            let r = &relaxations[j];

            // Lower bound of the composition:
            // If la >= 0, lower bound uses r.lower_slope (tighter lower from below)
            // If la < 0, lower bound uses r.upper_slope (tighter lower from above)
            if la >= 0.0 {
                new_lower_a[[i, j]] = la * r.lower_slope;
                new_lower_b[i] += la * r.lower_intercept;
            } else {
                new_lower_a[[i, j]] = la * r.upper_slope;
                new_lower_b[i] += la * r.upper_intercept;
            }

            // Upper bound of the composition:
            // If ua >= 0, upper bound uses r.upper_slope
            // If ua < 0, upper bound uses r.lower_slope
            if ua >= 0.0 {
                new_upper_a[[i, j]] = ua * r.upper_slope;
                new_upper_b[i] += ua * r.upper_intercept;
            } else {
                new_upper_a[[i, j]] = ua * r.lower_slope;
                new_upper_b[i] += ua * r.lower_intercept;
            }
        }
    }

    LinearBounds64::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}
