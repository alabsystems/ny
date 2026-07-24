// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Dense warm-start vector builder for HiGHS MIP solving.
//
// Replays the MIP encoder's column creation order and forward-propagates a PGD
// candidate through the stripped feedforward network to produce a full primal
// column vector for HiGHS's `try_set_solution()` API.
//
// Part of #3865.

use ndarray::ArrayD;
use ny_core::Bound;

/// Build a dense HiGHS primal column vector from a PGD candidate input.
///
/// Replays the encoder's column creation order (see `encode_feedforward`,
/// `encode_linear`, `encode_relu` in `ny-mip/src/encoder.rs`) and fills in
/// values by forward-propagating the candidate through the stripped feedforward
/// network.
///
/// Returns `None` if the candidate dimension doesn't match the network input,
/// or if `num_cols` is 0 (no columns to fill).
///
/// Column creation order in the encoder:
/// 1. Input columns: `layer_dims[0]`
/// 2. For each layer `i` in `0..num_layers`:
///    a. Linear: `layer_dims[i+1]` pre-activation columns
///    b. ReLU (if not final layer): for each neuron:
///       - `lb >= 0` (active): no new column
///       - `ub <= 0` (inactive): 1 column (y_var, value 0.0)
///       - unstable: 2 columns (y_var=max(z,0), z_var=indicator)
pub(super) fn build_warm_start_vector(
    candidate: &ArrayD<f32>,
    weights: &[Vec<f64>],
    biases: &[Vec<f64>],
    layer_dims: &[usize],
    intermediate_bounds: &[Vec<Bound>],
    num_cols: usize,
) -> Option<Vec<f64>> {
    if num_cols == 0 {
        return None;
    }

    let input_flat: Vec<f64> = candidate.iter().map(|&v| v as f64).collect();
    if input_flat.len() != layer_dims[0] {
        tracing::warn!(
            "Warm-start candidate dimension {} doesn't match network input {}",
            input_flat.len(),
            layer_dims[0]
        );
        return None;
    }

    let num_layers = weights.len();
    let mut dense = vec![0.0f64; num_cols];
    let mut col_idx = 0usize;

    // 1. Input columns
    for &val in &input_flat {
        if col_idx >= num_cols {
            tracing::warn!("Warm-start vector overflow at input columns");
            return None;
        }
        dense[col_idx] = val;
        col_idx += 1;
    }

    // Forward propagation state: current layer activations
    let mut current_activations = input_flat;

    for layer_idx in 0..num_layers {
        let out_dim = layer_dims[layer_idx + 1];
        let in_dim = current_activations.len();

        // 2a. Linear layer: z = W*x + b
        let mut pre_activations = Vec::with_capacity(out_dim);
        for i in 0..out_dim {
            let mut z = biases[layer_idx][i];
            for (j, &x) in current_activations.iter().enumerate() {
                z += weights[layer_idx][i * in_dim + j] * x;
            }
            if col_idx >= num_cols {
                tracing::warn!("Warm-start vector overflow at linear layer {}", layer_idx);
                return None;
            }
            dense[col_idx] = z;
            col_idx += 1;
            pre_activations.push(z);
        }

        // 2b. ReLU layer (all layers except the last)
        if layer_idx < num_layers - 1 {
            col_idx = fill_relu_columns(
                &pre_activations,
                &intermediate_bounds[layer_idx],
                &mut dense,
                col_idx,
                num_cols,
                &mut current_activations,
            )?;
        } else {
            current_activations = pre_activations;
        }
    }

    if col_idx != num_cols {
        tracing::warn!(
            "Warm-start vector size mismatch: filled {} of {} columns",
            col_idx,
            num_cols
        );
    }

    Some(dense)
}

/// Fill ReLU columns in the dense warm-start vector, returning the updated column index.
fn fill_relu_columns(
    pre_activations: &[f64],
    bounds: &[Bound],
    dense: &mut [f64],
    mut col_idx: usize,
    num_cols: usize,
    post_activations: &mut Vec<f64>,
) -> Option<usize> {
    post_activations.clear();
    for (i, &z) in pre_activations.iter().enumerate() {
        let lb = bounds[i].lower() as f64;
        let ub = bounds[i].upper() as f64;

        if lb >= 0.0 {
            // Always active: no new column, post-act = pre-act
            post_activations.push(z);
        } else if ub <= 0.0 {
            // Always inactive: 1 new column (y_var = 0.0)
            if col_idx >= num_cols {
                tracing::warn!("Warm-start vector overflow at inactive ReLU");
                return None;
            }
            dense[col_idx] = 0.0;
            col_idx += 1;
            post_activations.push(0.0);
        } else {
            // Unstable: 2 new columns (y_var, z_var)
            let y = z.max(0.0);
            let indicator = if z >= 0.0 { 1.0 } else { 0.0 };
            if col_idx + 1 >= num_cols {
                tracing::warn!("Warm-start vector overflow at unstable ReLU");
                return None;
            }
            dense[col_idx] = y;
            dense[col_idx + 1] = indicator;
            col_idx += 2;
            post_activations.push(y);
        }
    }
    Some(col_idx)
}
