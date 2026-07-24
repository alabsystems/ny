// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Double-precision Linear (FC) layer: y = Wx + b.
//!
//! IBP: standard positive/negative weight splitting.
//! CROWN backward: A_new = A @ W, b_new = A @ bias + b_old.
//!
//! Reference: alpha-beta-CROWN `auto_lirpa/operators/linear.py`.

use ndarray::{Array1, Array2};
use ny_core::Result;
use ny_tensor::BoundedTensor64;

use crate::bounds::LinearBounds64;

/// IBP propagation for a linear layer in f64.
///
/// For y = Wx + b:
///   lower_y = W+ @ x_lower + W- @ x_upper + b
///   upper_y = W+ @ x_upper + W- @ x_lower + b
///
/// where W+ = max(W, 0), W- = min(W, 0).
pub(crate) fn propagate_linear_ibp_f64(
    weight: &Array2<f64>,
    bias: &Array1<f64>,
    input: &BoundedTensor64,
) -> Result<BoundedTensor64> {
    let (in_l, in_u) = input.flatten_to_1d();
    let out_features = weight.nrows();
    let in_features = weight.ncols();

    if in_l.len() != in_features {
        return Err(ny_core::NyError::shape_mismatch(
            vec![in_features],
            vec![in_l.len()],
        ));
    }

    let mut lower = Array1::<f64>::zeros(out_features);
    let mut upper = Array1::<f64>::zeros(out_features);

    for i in 0..out_features {
        let mut sum_l = bias[i];
        let mut sum_u = bias[i];

        for j in 0..in_features {
            let w = weight[[i, j]];
            if w > 0.0 {
                sum_l += w * in_l[j];
                sum_u += w * in_u[j];
            } else if w < 0.0 {
                sum_l += w * in_u[j];
                sum_u += w * in_l[j];
            }
            // w == 0.0: skip (0 * anything = 0, including 0 * inf = 0 for soundness)
        }

        lower[i] = sum_l;
        upper[i] = sum_u;
    }

    BoundedTensor64::new(lower.into_dyn(), upper.into_dyn())
}

/// CROWN backward propagation for a linear layer in f64.
///
/// For y = Wx + b and current bounds A @ y + c:
///   new_A = A @ W
///   new_b = A @ bias + c
pub(crate) fn propagate_linear_crown_backward_f64(
    weight: &Array2<f64>,
    bias: &Array1<f64>,
    bounds: &LinearBounds64,
) -> Result<LinearBounds64> {
    let m = bounds.num_outputs();
    let k = weight.nrows(); // out_features of the linear layer

    if bounds.num_inputs() != k {
        return Err(ny_core::NyError::shape_mismatch(
            vec![k],
            vec![bounds.num_inputs()],
        ));
    }

    // new_A = A @ W: (m, k) @ (k, n) -> (m, n)
    let new_lower_a = bounds.lower_a().dot(weight);
    let new_upper_a = bounds.upper_a().dot(weight);

    // new_b = A @ bias + b_old
    let mut new_lower_b = Array1::<f64>::zeros(m);
    let mut new_upper_b = Array1::<f64>::zeros(m);

    for i in 0..m {
        let mut dot_l = bounds.lower_b()[i];
        let mut dot_u = bounds.upper_b()[i];

        for j in 0..k {
            dot_l += bounds.lower_a()[[i, j]] * bias[j];
            dot_u += bounds.upper_a()[[i, j]] * bias[j];
        }

        new_lower_b[i] = dot_l;
        new_upper_b[i] = dot_u;
    }

    // NaN firewall: if dot product produced NaN, fall back to conservative
    LinearBounds64::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}

/// Pre-convert f32 weights to f64 for the f64 propagation path.
///
/// f32→f64 is exact (no rounding needed).
pub fn weights_to_f64(
    weight: &Array2<f32>,
    bias: Option<&Array1<f32>>,
) -> (Array2<f64>, Array1<f64>) {
    let w64 = weight.mapv(|x| x as f64);
    let b64 = match bias {
        Some(b) => b.mapv(|x| x as f64),
        None => Array1::zeros(weight.nrows()),
    };
    (w64, b64)
}
