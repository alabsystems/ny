// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Clip-intermediate adaptation and NaN-aware merge helpers for domain bounds.

use std::sync::Arc;

use ndarray::{Array1, Array2};
use ny_core::{nan_propagating_max, nan_propagating_min, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::beta_crown::branching::SplitHistory;
use crate::beta_crown::domain::IntermediateLinearBounds;
use crate::clip_interm_domain::clip_interm_domain_full;

use super::super::BetaCrownVerifier;

/// Merge old and new domain bounds using NaN-propagating min/max.
///
/// For lower bounds: `max(old, new)` (tighten from below).
/// For upper bounds: `min(old, new)` (tighten from above).
///
/// Uses `nan_propagating_max`/`nan_propagating_min` so that NaN in either old
/// or new bounds is never silently absorbed into a finite (unsound) bound.
/// IEEE 754 `f32::max`/`f32::min` return the non-NaN operand, hiding corruption.
///
/// Reference: #2858, #2577.
pub(super) fn merge_domain_bounds(
    old_lower: &Array1<f32>,
    new_lower: &Array1<f32>,
    old_upper: &Array1<f32>,
    new_upper: &Array1<f32>,
) -> (Array1<f32>, Array1<f32>) {
    let merged_lower: Array1<f32> = old_lower
        .iter()
        .zip(new_lower.iter())
        .map(|(&o, &n): (&f32, &f32)| nan_propagating_max(o, n))
        .collect();
    let merged_upper: Array1<f32> = old_upper
        .iter()
        .zip(new_upper.iter())
        .map(|(&o, &n): (&f32, &f32)| nan_propagating_min(o, n))
        .collect();
    (merged_lower, merged_upper)
}

pub(super) fn has_infeasible_layer_bounds(layer_bounds: &[Arc<BoundedTensor>]) -> bool {
    layer_bounds.iter().any(|bt| {
        ndarray::Zip::from(bt.lower())
            .and(bt.upper())
            .any(|&l, &u| l > u)
    })
}

impl BetaCrownVerifier {
    /// Apply clip_interm_domain to tighten intermediate bounds using split constraints.
    ///
    /// This adapts the engine's data formats to the `clip_interm_domain_full` API:
    /// - Converts `SplitHistory` to `GraphSplitHistory`
    /// - Creates adapter closures for linear bounds access
    /// - Updates layer bounds with tightened values
    pub(in crate::beta_crown::engine) fn apply_clip_interm_domain(
        &self,
        history: &SplitHistory,
        mut layer_bounds: Vec<Arc<BoundedTensor>>,
        intermediate_bounds: &IntermediateLinearBounds,
        input: &BoundedTensor,
        parent_input_bounds: Option<&BoundedTensor>,
    ) -> Result<Vec<Arc<BoundedTensor>>> {
        // Convert SplitHistory to GraphSplitHistory for clip_interm_domain API
        let graph_history = history.to_graph_split_history()?;

        // Get input bounds (use tightened bounds if available)
        let effective_input = parent_input_bounds.unwrap_or(input);
        let input_flat = effective_input.flatten();
        let input_lower: Array1<f32> =
            Array1::from_vec(input_flat.lower().iter().copied().collect());
        let input_upper: Array1<f32> =
            Array1::from_vec(input_flat.upper().iter().copied().collect());

        // Convert layer_bounds to the expected format: Vec<(Array1<f32>, Array1<f32>)>
        let layer_bounds_flat: Vec<(Array1<f32>, Array1<f32>)> = layer_bounds
            .iter()
            .map(|bt| {
                let flat = bt.flatten();
                let lower_arr: Array1<f32> =
                    Array1::from_vec(flat.lower().iter().copied().collect());
                let upper_arr: Array1<f32> =
                    Array1::from_vec(flat.upper().iter().copied().collect());
                (lower_arr, upper_arr)
            })
            .collect();

        // Create adapter for split neuron linear bounds
        // node_name format: "layer_N" where N is the layer index
        let linear_bounds_for_split =
            |node_name: &str, neuron_idx: usize| -> Option<(Array1<f32>, f32, Array1<f32>, f32)> {
                // Parse layer index from node name "layer_N"
                let layer_idx: usize = node_name.strip_prefix("layer_")?.parse().ok()?;

                // Get linear bounds for this layer
                let lb = intermediate_bounds.get(layer_idx)?;

                // Check bounds
                if neuron_idx >= lb.num_outputs() {
                    return None;
                }

                // Extract single neuron's bounds
                let l_a = lb.lower_a().row(neuron_idx).to_owned();
                let l_bias = lb.lower_b()[neuron_idx];
                let u_a = lb.upper_a().row(neuron_idx).to_owned();
                let u_bias = lb.upper_b()[neuron_idx];

                Some((l_a, l_bias, u_a, u_bias))
            };

        // Create adapter for objective neuron linear bounds
        // Justification: Closure returns (lA, lbias, uA, ubias) tuple — the natural
        // representation of linear bound coefficients; a named struct would add indirection.
        #[allow(clippy::type_complexity)]
        let linear_bounds_for_objective =
            |layer_idx: usize,
             neuron_indices: &[usize]|
             -> Option<(Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>)> {
                // Get linear bounds for this layer
                let lb = intermediate_bounds.get(layer_idx)?;

                let n_selected = neuron_indices.len();
                let n_inputs = lb.num_inputs();

                // Extract rows for selected neurons
                let mut l_a = Array2::zeros((n_selected, n_inputs));
                let mut l_bias = Array1::zeros(n_selected);
                let mut u_a = Array2::zeros((n_selected, n_inputs));
                let mut u_bias = Array1::zeros(n_selected);

                for (i, &neuron_idx) in neuron_indices.iter().enumerate() {
                    if neuron_idx >= lb.num_outputs() {
                        return None;
                    }
                    l_a.row_mut(i).assign(&lb.lower_a().row(neuron_idx));
                    l_bias[i] = lb.lower_b()[neuron_idx];
                    u_a.row_mut(i).assign(&lb.upper_a().row(neuron_idx));
                    u_bias[i] = lb.upper_b()[neuron_idx];
                }

                Some((l_a, l_bias, u_a, u_bias))
            };

        // Compute coefficient magnitudes from CROWN linear bounds for neuron selection
        // Following alpha-beta-CROWN: use mean absolute value of A matrix per neuron
        let coeff_magnitudes: Vec<Array1<f32>> = (0..layer_bounds_flat.len())
            .map(|layer_idx| {
                intermediate_bounds
                    .get(layer_idx)
                    .map(|lb| {
                        // Compute |lA|.mean(axis=1) as per-neuron magnitude
                        // Shape: (num_neurons, num_inputs) -> (num_neurons,)
                        let num_outputs = lb.lower_a().nrows();
                        let num_inputs = lb.lower_a().ncols();
                        let mut mags = Array1::zeros(num_outputs);
                        for i in 0..num_outputs {
                            let row_sum: f32 = lb.lower_a().row(i).mapv(|v| v.abs()).sum();
                            mags[i] = row_sum / (num_inputs.max(1) as f32);
                        }
                        mags
                    })
                    .unwrap_or_else(|| Array1::ones(layer_bounds_flat[layer_idx].0.len()))
            })
            .collect();

        // Call clip_interm_domain_full to get tightened bounds
        let tightened = clip_interm_domain_full(
            &graph_history,
            linear_bounds_for_split,
            linear_bounds_for_objective,
            &layer_bounds_flat,
            &input_lower,
            &input_upper,
            self.config.clip_interm_topk,
            Some(&coeff_magnitudes),
        )?;

        // Update layer_bounds with tightened values
        for (layer_idx, (new_lower, new_upper)) in tightened.into_iter().enumerate() {
            if layer_idx >= layer_bounds.len() {
                break;
            }

            let old_bt = &layer_bounds[layer_idx];
            let old_flat = old_bt.flatten();
            let old_shape = old_bt.lower().shape().to_vec();

            // Merge: new = max(old, tightened) for lower, min(old, tightened) for upper
            let old_lower: Array1<f32> =
                Array1::from_vec(old_flat.lower().iter().copied().collect());
            let old_upper: Array1<f32> =
                Array1::from_vec(old_flat.upper().iter().copied().collect());

            // Check if bounds changed.
            // NaN-aware: treat NaN as a change (NaN comparisons return false,
            // which would silently classify NaN-corrupted domains as "unchanged").
            let mut changed = false;
            for i in 0..new_lower.len().min(old_lower.len()) {
                if new_lower[i] > old_lower[i]
                    || new_upper[i] < old_upper[i]
                    || new_lower[i].is_nan()
                    || new_upper[i].is_nan()
                    || old_lower[i].is_nan()
                    || old_upper[i].is_nan()
                {
                    changed = true;
                    break;
                }
            }

            if changed {
                let (merged_lower, merged_upper) =
                    merge_domain_bounds(&old_lower, &new_lower, &old_upper, &new_upper);

                // Reshape to original shape and create new BoundedTensor
                let lower_dyn = merged_lower
                    .into_shape_clone(ndarray::IxDyn(&old_shape))
                    .map_err(|err| {
                        NyError::InternalError(format!(
                            "apply_clip_interm_domain: reshape lower bounds failed for layer {} to {:?}: {}",
                            layer_idx, old_shape, err
                        ))
                    })?;
                let upper_dyn = merged_upper
                    .into_shape_clone(ndarray::IxDyn(&old_shape))
                    .map_err(|err| {
                        NyError::InternalError(format!(
                            "apply_clip_interm_domain: reshape upper bounds failed for layer {} to {:?}: {}",
                            layer_idx, old_shape, err
                        ))
                    })?;

                layer_bounds[layer_idx] = Arc::new(BoundedTensor::new(lower_dyn, upper_dyn)?);
            }
        }

        Ok(layer_bounds)
    }
}
