// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GCP-CROWN cut (λ) gradient computation.

use ny_tensor::BoundedTensor;
use std::sync::Arc;

use crate::beta_crown::bab_cuts::CutPool;
use crate::beta_crown::branching::SplitHistory;

/// Compute cut gradients using ReLU indicators (GCP-CROWN).
///
/// For the Lagrangian term: `lambda * (bias - constraint_min)`,
/// the gradient is `d(lb)/d(λ) = bias - constraint_min`.
/// Positive gradient means increasing lambda will increase the lower bound.
pub(super) fn compute_cut_gradients(
    cut_pool: &mut CutPool,
    history: &SplitHistory,
    layer_bounds: &[Arc<BoundedTensor>],
) {
    let relevant_indices: Vec<usize> = cut_pool
        .cuts
        .iter()
        .enumerate()
        .filter(|(_, cut)| !cut.is_redundant_for(history))
        .map(|(i, _)| i)
        .collect();

    for idx in relevant_indices {
        // Compute bias and constraint_min from immutable borrow, then
        // drop the borrow before the mutable set_lambda_grad call.
        let (bias, constraint_min) = {
            let cut = &cut_pool.cuts[idx];
            let bias = cut.bias();

            // Compute minimum value of the constraint: sum_i(coeff_i * z_i)
            // where z_i is the ReLU indicator (0 if inactive, 1 if active)
            let constraint_min: f32 = cut
                .terms
                .iter()
                .filter_map(|term| {
                    if term.layer_idx == 0 || term.layer_idx > layer_bounds.len() {
                        return None;
                    }
                    let pre_bounds = &layer_bounds[term.layer_idx - 1];
                    let flat = pre_bounds.flatten();

                    if term.neuron_idx >= flat.len() {
                        return None;
                    }

                    let l = flat.lower()[[term.neuron_idx]];
                    let u = flat.upper()[[term.neuron_idx]];

                    // Determine ReLU indicator bounds [z_min, z_max]
                    let (z_min, z_max) = if l >= 0.0 {
                        (1.0, 1.0) // Stable active
                    } else if u <= 0.0 {
                        (0.0, 0.0) // Stable inactive
                    } else {
                        (0.0, 1.0) // Unstable
                    };

                    let value = if term.coefficient > 0.0 {
                        term.coefficient * z_min
                    } else {
                        term.coefficient * z_max
                    };
                    Some(value)
                })
                .sum();
            (bias, constraint_min)
        };

        // Gradient: d(lb)/d(lambda) = bias - constraint_min
        // The cut adds: lambda * (bias - constraint_min) to the lower bound
        cut_pool.cuts[idx].set_lambda_grad(bias - constraint_min);
    }
}
