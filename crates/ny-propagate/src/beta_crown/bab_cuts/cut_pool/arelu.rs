// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Arelu state builder for cut integration.
//!
//! Converts the sparse cut representation into dense per-layer coefficient
//! matrices suitable for the arelu_cut backward pass modification.

use super::CutPool;

impl CutPool {
    /// Build an `AreluState` from the current cuts for arelu_cut integration.
    ///
    /// This converts the sparse cut representation into dense per-layer coefficient
    /// matrices suitable for the arelu_cut backward pass modification.
    ///
    /// # Arguments
    /// * `layer_bounds` - Bounds for each layer (used to determine layer sizes)
    ///
    /// # Returns
    /// `AreluState` with weighted coefficients for each layer that has active cuts.
    ///
    /// # Algorithm
    /// For each cut with non-negligible lambda (|lambda| >= 1e-10):
    /// 1. For each term (layer_idx, neuron_idx, coefficient):
    ///    - Add lambda * coefficient to arelu_coeffs[layer_idx][neuron_idx]
    ///
    /// This implements the transformation needed for the arelu_cut method:
    /// beta_mm_coeffs = sum over cuts (lambda_c * coeff_c)
    ///
    /// # Reference
    /// alpha-beta-CROWN: `auto_LiRPA/operators/cut_ops.py:298-491`
    pub fn build_arelu_state(
        &self,
        layer_bounds: &[std::sync::Arc<ny_tensor::BoundedTensor>],
    ) -> crate::beta_crown::state::AreluState {
        use crate::beta_crown::state::AreluState;
        use ndarray::Array1;
        use std::collections::HashMap;

        // Build sparse coefficients first (layer_idx -> neuron_idx -> weighted_coeff)
        let mut sparse_coeffs: HashMap<usize, HashMap<usize, f32>> = HashMap::new();

        for cut in &self.cuts {
            // Skip cuts with negligible lambda
            if cut.lambda.abs() < 1e-10 {
                continue;
            }

            for term in &cut.terms {
                // Validate layer index
                if term.layer_idx >= layer_bounds.len() {
                    continue;
                }

                // Accumulate weighted coefficient
                sparse_coeffs
                    .entry(term.layer_idx)
                    .or_default()
                    .entry(term.neuron_idx)
                    .and_modify(|c| *c += cut.lambda * term.coefficient)
                    .or_insert(cut.lambda * term.coefficient);
            }
        }

        // Convert to dense arrays and build masks
        let mut weighted_coeffs: HashMap<usize, Array1<f32>> = HashMap::new();
        let mut has_cut_mask: HashMap<usize, Vec<bool>> = HashMap::new();

        for (&layer_idx, neuron_coeffs) in &sparse_coeffs {
            // Get layer size from bounds
            let layer_size = layer_bounds[layer_idx].len();
            if layer_size == 0 {
                continue;
            }

            let mut dense = Array1::<f32>::zeros(layer_size);
            let mut mask = vec![false; layer_size];

            for (&neuron_idx, &coeff) in neuron_coeffs {
                if neuron_idx >= layer_size {
                    tracing::debug!(
                        "arelu_cut: skipping out-of-bounds neuron {} (layer_size={})",
                        neuron_idx,
                        layer_size
                    );
                    continue;
                }
                if coeff.abs() > 1e-10 {
                    dense[neuron_idx] = coeff;
                    mask[neuron_idx] = true;
                }
            }

            if mask.iter().any(|&m| m) {
                weighted_coeffs.insert(layer_idx, dense);
                has_cut_mask.insert(layer_idx, mask);
            }
        }

        AreluState {
            weighted_coeffs,
            has_cut_mask,
        }
    }
}
