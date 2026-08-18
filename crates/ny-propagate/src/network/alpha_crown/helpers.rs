// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use ndarray::Array1;
use ny_core::Result;
use ny_tensor::BoundedTensor;

use tracing::info;

use crate::bounds::{AlphaCrownConfig, AlphaCrownIntermediate, AlphaState};
use crate::layers::Layer;

pub(super) fn build_layer_to_relu_idx(layers: &[Layer]) -> (Vec<usize>, HashMap<usize, usize>) {
    let relu_layer_indices: Vec<usize> = layers
        .iter()
        .enumerate()
        .filter(|(_, layer)| matches!(layer, Layer::ReLU(_)))
        .map(|(idx, _)| idx)
        .collect();

    let mut layer_to_relu_idx = HashMap::with_capacity(relu_layer_indices.len());
    for (relu_idx, &layer_idx) in relu_layer_indices.iter().enumerate() {
        layer_to_relu_idx.insert(layer_idx, relu_idx);
    }

    (relu_layer_indices, layer_to_relu_idx)
}

pub(super) fn compute_chain_rule_gradients(
    alpha_state: &AlphaState,
    intermediate: &AlphaCrownIntermediate,
) -> Vec<Array1<f32>> {
    if intermediate.a_at_relu.is_empty() {
        // Fall back to empty gradients if no intermediates were stored.
        return alpha_state
            .alphas
            .iter()
            .map(|a| Array1::zeros(a.len()))
            .collect();
    }

    let num_relus = intermediate.a_at_relu.len();
    let mut gradients: Vec<Array1<f32>> = Vec::with_capacity(num_relus);

    // For each ReLU layer
    for relu_idx in 0..num_relus {
        let a_at_relu = &intermediate.a_at_relu[relu_idx];
        let (pre_lower, pre_upper) = &intermediate.pre_relu_bounds[relu_idx];
        let n_neurons = pre_lower.len();

        let mut grad = Array1::<f32>::zeros(n_neurons);

        // For each neuron in this ReLU layer
        for i in 0..n_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];

            // Guard: non-finite pre-ReLU bounds cannot produce meaningful gradients.
            // IEEE-754: NaN comparisons return false, so `l >= 0.0 || u <= 0.0`
            // would fail for NaN bounds, treating them as "unstable" and flowing
            // NaN into gradient arithmetic. Explicitly skip non-finite. (#2809)
            if !l.is_finite() || !u.is_finite() {
                continue;
            }

            // Only unstable neurons (l < 0 < u) have non-zero gradient.
            if l >= 0.0 || u <= 0.0 {
                continue;
            }

            // Compute gradient contribution from all output dimensions.
            //
            // For maximizing the lower bound:
            // For lower relaxation y >= alpha*x with x in [l, u] where l < 0 < u:
            // - Contribution to lower bound = A[j,i] * alpha * min(x) = A[j,i] * alpha * l
            // - Gradient d(bound)/d(alpha) = A[j,i] * l
            // Note: l < 0 for unstable neurons, so gradient is typically negative
            // when A[j,i] > 0, meaning increasing alpha decreases the lower bound.
            let num_outputs = a_at_relu.nrows();
            let mut grad_i = 0.0f32;

            for j in 0..num_outputs {
                let a_ji = a_at_relu[[j, i]];

                // Guard: non-finite A coefficient cannot produce meaningful
                // gradient contributions. Without this, NaN a_ji silently
                // drops the contribution via `a_ji > 0.0` returning false. (#2809)
                if !a_ji.is_finite() {
                    continue;
                }

                // When A >= 0, lower relaxation uses y >= alpha*x.
                // The binding point is x = l (lower bound), not u.
                if a_ji > 0.0 {
                    // Lower relaxation active: y >= alpha*x.
                    // Contribution to lower bound: A[j,i] * alpha * l.
                    // Gradient w.r.t. alpha: A[j,i] * l.
                    grad_i += a_ji * l;
                }
                // When A < 0, upper relaxation y <= (u/(u-l))*(x-l) is used.
                // This doesn't depend on alpha, so gradient is 0.
            }

            grad[i] = grad_i;
        }

        gradients.push(grad);
    }

    gradients
}

/// Initialize INVPROP state on `alpha_state` if enabled in `config`.
///
/// Returns `true` if INVPROP was enabled and initialized, `false` otherwise.
///
/// `input_len` is the number of input neurons, used to allocate NETWORK_INPUT
/// gammas when `should_apply_to_input()` is true (#2928).
pub(super) fn init_invprop_if_enabled(
    config: &AlphaCrownConfig,
    alpha_state: &mut AlphaState,
    relu_layer_indices: &[usize],
    pre_activation_bounds: &[BoundedTensor],
    input_len: usize,
) -> Result<bool> {
    let enabled = config.invprop.enabled && config.output_constraints.is_some();
    if !enabled {
        return Ok(false);
    }
    let oc = match config.output_constraints {
        Some(ref oc) => oc,
        None => return Ok(false),
    };

    // Initialize invprop_state with output constraints (batch_size=1)
    alpha_state.init_invprop_state(oc.clone(), 1)?;

    if let Some(ref mut state) = alpha_state.invprop_state {
        let num_constraints = oc.num_constraints();

        // Output-seed duals (the shipped, output-node-only assume-violation channel).
        // Always allocated: "neuron" dim = output_dim (one dual per output coord per
        // constraint), folded into the identity seed by the sign-aware re-seed.
        let seed_gammas = crate::invprop::LayerGammas::new(
            num_constraints,
            oc.output_dim(),
            config.invprop.share_gammas,
        );
        state.add_layer_gammas(crate::invprop::INVPROP_OUTPUT_SEED.to_string(), seed_gammas);

        // Per-layer intermediate-bound gammas (research per-layer channel) only when
        // explicitly enabled. Default output-node-only leaves these unallocated, so
        // the historical per-layer / input-level augment sites find no gammas and skip.
        if config.invprop.per_layer_gammas {
            for (relu_idx, &layer_idx) in relu_layer_indices.iter().enumerate() {
                let layer_name = format!("/layer.{}", layer_idx);
                let layer_type = "BoundReLU";

                if config.invprop.should_apply_to(&layer_name, layer_type) {
                    let num_neurons = pre_activation_bounds[relu_idx].len();
                    let gammas = crate::invprop::LayerGammas::new(
                        num_constraints,
                        num_neurons,
                        config.invprop.share_gammas,
                    );
                    state.add_layer_gammas(layer_name, gammas);
                }
            }

            if config.invprop.should_apply_to_input()
                && state.layer_gammas(crate::NETWORK_INPUT).is_none()
            {
                let gammas = crate::invprop::LayerGammas::new(
                    num_constraints,
                    input_len,
                    config.invprop.share_gammas,
                );
                state.add_layer_gammas(crate::NETWORK_INPUT.to_string(), gammas);
            }
        }
    }

    info!(
        "Alpha-CROWN: INVPROP enabled with {} constraints, {} layers with gammas",
        oc.num_constraints(),
        alpha_state
            .invprop_state
            .as_ref()
            .map(|s| s.layer_gammas.len())
            .unwrap_or(0)
    );
    if alpha_state.invprop_state.is_some() {
        crate::execution_telemetry::record_invprop_alpha_initialization();
    }

    Ok(true)
}
