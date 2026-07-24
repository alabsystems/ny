// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! AreluState: State for arelu_cut backward integration in GCP-CROWN.

use ndarray::{Array1, Array2};
use ny_core::{nan_propagating_max, nan_propagating_min};

/// State for arelu_cut backward integration in GCP-CROWN.
///
/// The arelu_cut method modifies ReLU backward pass to incorporate integer
/// activation indicator cuts, enabling tighter bounds. Based on alpha-beta-CROWN's
/// `cut_ops.py:arelu_cut` method.
///
/// # Algorithm (from alpha-beta-CROWN)
///
/// For unstable ReLU neurons with bounds [l, u] where l < 0 < u:
///
/// 1. Compute `beta_mm_coeffs = einsum(general_beta, arelu_coeffs)`
///    - This weights the cut coefficients by learnable multipliers
///
/// 2. Compute pi using closed-form solution:
///    - `pi = (u * nu_hat_pos + beta_mm_coeffs[0]) / (u - l)`
///    - `pi = clamp(min(pi, nu_hat_pos), 0)`
///
/// 3. Modified upper bound slope:
///    - `new_upper_d = pi / nu_hat_pos`
///
/// 4. Piecewise bias computation:
///    - If `beta_mm_coeffs[0] <= -u * nu_hat_pos`: lbias = 0
///    - If `beta_mm_coeffs[0] >= -l * nu_hat_pos`: lbias = -beta_mm_coeffs[0]
///    - Otherwise: lbias = pi * l
///
/// # Reference
///
/// alpha-beta-CROWN: `auto_LiRPA/operators/cut_ops.py:298-491`
#[derive(Debug, Clone)]
pub struct AreluState {
    /// Per-layer arelu coefficients weighted by lambdas: [num_neurons]
    /// This is the pre-computed `beta_mm_coeffs[0]` for each neuron in the layer.
    /// Maps layer_idx -> weighted coefficient array.
    pub weighted_coeffs: std::collections::HashMap<usize, Array1<f32>>,

    /// Per-layer mask of neurons that have active arelu cuts.
    /// Maps layer_idx -> boolean array indicating which neurons have cuts.
    pub has_cut_mask: std::collections::HashMap<usize, Vec<bool>>,
}

impl AreluState {
    /// Create empty arelu state.
    pub fn empty() -> Self {
        Self {
            weighted_coeffs: std::collections::HashMap::new(),
            has_cut_mask: std::collections::HashMap::new(),
        }
    }

    /// Create AreluState from CutModule arelu coefficients and lambdas.
    ///
    /// # Arguments
    /// * `arelu_coeffs` - Per-layer arelu coefficient matrices [num_cuts, num_neurons]
    /// * `lambdas` - Lagrangian multipliers [num_cuts]
    ///
    /// # Soundness Contract
    ///
    /// REQUIRES: arelu_coeffs[layer].shape() == (lambdas.len(), layer_size)
    /// ENSURES: weighted_coeffs[layer] = sum over cuts (lambdas[c] * arelu_coeffs[layer][c, :])
    pub fn from_cut_module(
        arelu_coeffs: &std::collections::HashMap<usize, Array2<f32>>,
        lambdas: &Array1<f32>,
    ) -> Self {
        let mut weighted_coeffs = std::collections::HashMap::new();
        let mut has_cut_mask = std::collections::HashMap::new();

        for (&layer_idx, coeffs) in arelu_coeffs {
            let num_neurons = coeffs.ncols();
            let mut weighted = Array1::<f32>::zeros(num_neurons);
            let mut mask = vec![false; num_neurons];

            // Compute weighted sum: beta_mm_coeffs = sum over cuts (lambda_c * coeff_c)
            for (cut_idx, &lambda) in lambdas.iter().enumerate() {
                if lambda.abs() < 1e-10 {
                    continue;
                }
                for j in 0..num_neurons {
                    let coeff = coeffs[[cut_idx, j]];
                    if coeff.abs() > 1e-10 {
                        weighted[j] += lambda * coeff;
                        mask[j] = true;
                    }
                }
            }

            if mask.iter().any(|&m| m) {
                weighted_coeffs.insert(layer_idx, weighted);
                has_cut_mask.insert(layer_idx, mask);
            }
        }

        Self {
            weighted_coeffs,
            has_cut_mask,
        }
    }

    /// Check if the arelu state is empty (no active cuts).
    pub fn is_empty(&self) -> bool {
        self.weighted_coeffs.is_empty()
    }

    /// The weighted arelu coefficient for a specific neuron.
    ///
    /// Returns Some(coefficient) if this neuron has an active arelu cut,
    /// None otherwise.
    pub fn weighted_coeff(&self, layer_idx: usize, neuron_idx: usize) -> Option<f32> {
        if let Some(mask) = self.has_cut_mask.get(&layer_idx) {
            if let Some(&has_cut) = mask.get(neuron_idx) {
                if has_cut {
                    return self
                        .weighted_coeffs
                        .get(&layer_idx)
                        .and_then(|w| w.get(neuron_idx).copied());
                }
            }
        }
        None
    }

    /// Check if a specific neuron has an active arelu cut.
    pub fn has_cut(&self, layer_idx: usize, neuron_idx: usize) -> bool {
        self.has_cut_mask
            .get(&layer_idx)
            .and_then(|mask| mask.get(neuron_idx).copied())
            .unwrap_or(false)
    }

    /// The weighted coefficients array for a layer, if any.
    pub fn layer_coeffs(&self, layer_idx: usize) -> Option<&Array1<f32>> {
        self.weighted_coeffs.get(&layer_idx)
    }

    /// Set the weighted arelu coefficient for a specific neuron.
    ///
    /// This is primarily for testing; production code typically uses
    /// `from_cut_module` to construct the state.
    pub fn set_weighted_coeff(&mut self, layer_idx: usize, neuron_idx: usize, coeff: f32) {
        // Ensure the weighted_coeffs array exists and is large enough
        let weighted = self
            .weighted_coeffs
            .entry(layer_idx)
            .or_insert_with(|| Array1::zeros(neuron_idx + 1));
        if weighted.len() <= neuron_idx {
            let mut new_arr = Array1::zeros(neuron_idx + 1);
            new_arr
                .slice_mut(ndarray::s![..weighted.len()])
                .assign(weighted);
            *weighted = new_arr;
        }
        weighted[neuron_idx] = coeff;

        // Mark this neuron as having a cut
        let mask = self
            .has_cut_mask
            .entry(layer_idx)
            .or_insert_with(|| vec![false; neuron_idx + 1]);
        if mask.len() <= neuron_idx {
            mask.resize(neuron_idx + 1, false);
        }
        mask[neuron_idx] = true;
    }
}

/// Compute modified slope and bias for arelu_cut integration.
///
/// This implements the pi/ny computation from alpha-beta-CROWN's arelu_cut method.
///
/// # Arguments
/// * `lower` - Pre-activation lower bound (l < 0 for unstable)
/// * `upper` - Pre-activation upper bound (u > 0 for unstable)
/// * `a_coeff` - Backward coefficient (A matrix entry, typically < 0 when called)
/// * `beta_mm_coeff` - Weighted arelu coefficient (lambda * arelu_coeff sum)
///
/// # Returns
/// * `(new_slope, lbias)` - Modified slope and direct bias contribution
///
/// Note: `lbias` is the TOTAL bias for this neuron (not a delta). The caller should
/// replace the standard `la_ij * upper_intercept` with `lbias` directly.
///
/// # Algorithm
///
/// For `a_coeff < 0` (upper bound relaxation is used):
/// 1. `nu_hat_pos = |a_coeff|`
/// 2. `pi = (upper * nu_hat_pos + beta_mm_coeff) / (upper - lower)`
/// 3. `pi = clamp(min(pi, nu_hat_pos), 0)`
/// 4. `new_slope = pi / nu_hat_pos` (replaces standard `upper / (upper - lower)`)
/// 5. `lbias` is computed piecewise based on beta_mm_coeff thresholds
///
/// # Reference
/// alpha-beta-CROWN: `auto_LiRPA/operators/cut_ops.py:298-491`
pub fn compute_arelu_cut_slope_bias(
    lower: f32,
    upper: f32,
    a_coeff: f32,
    beta_mm_coeff: f32,
) -> (f32, f32) {
    const EPS: f32 = 1e-10;

    // Only applies to unstable neurons (l < 0 < u)
    // For stable neurons, caller should not invoke this function.
    // Return identity values if called incorrectly.
    if lower >= 0.0 || upper <= 0.0 {
        // Stable neuron: slope depends on stability
        // This function should not be called for stable neurons, but return
        // safe defaults: slope=1 for positive-stable, slope=0 for negative-stable
        let slope = if lower >= 0.0 { 1.0 } else { 0.0 };
        return (slope, 0.0);
    }

    // Only applies when using upper bound relaxation (a_coeff < 0)
    // When a_coeff >= 0, lower bound relaxation is used and arelu_cut doesn't apply.
    if a_coeff >= 0.0 {
        // Return standard upper bound slope (caller will use lower slope anyway)
        let standard_slope = upper / (upper - lower);
        let standard_intercept = -lower * upper / (upper - lower);
        // Return slope and the standard intercept (for consistency, though unused)
        return (standard_slope, a_coeff * standard_intercept);
    }

    // nu_hat_pos = |a_coeff| (positive contribution from backward pass)
    let nu_hat_pos = a_coeff.abs();

    // Compute pi using closed-form solution
    // pi = (upper * nu_hat_pos + beta_mm_coeff) / (upper - lower)
    let pi = (upper * nu_hat_pos + beta_mm_coeff) / (upper - lower + EPS);

    // pi = clamp(min(pi, nu_hat_pos), 0)
    // NaN-safe: propagate NaN through clamping instead of silently becoming 0.0 (#2643)
    let pi = nan_propagating_max(nan_propagating_min(pi, nu_hat_pos), 0.0);

    // New slope: new_upper_d = pi / nu_hat_pos
    let new_slope = pi / (nu_hat_pos + EPS);

    // Piecewise bias computation (lbias)
    // Threshold values from alpha-beta-CROWN
    let uc = -upper * nu_hat_pos; // Upper cutoff: uC = -u * nu_hat_pos
    let lc = -lower * nu_hat_pos; // Lower cutoff: lC = -l * nu_hat_pos

    let lbias = if beta_mm_coeff <= uc {
        // Case: beta_mm_coeff <= -upper * nu_hat_pos
        // lbias = 0 (cut is inactive on upper side)
        0.0
    } else if beta_mm_coeff >= lc {
        // Case: beta_mm_coeff >= -lower * nu_hat_pos
        // lbias = -beta_mm_coeff (cut is fully active)
        -beta_mm_coeff
    } else {
        // Default case: -u * nu_hat_pos < beta_mm_coeff < -l * nu_hat_pos
        // lbias = pi * lower
        pi * lower
    };

    // Return lbias directly (not as a delta from standard bias).
    // The caller should use: new_lower_b[i] += lbias
    // instead of: new_lower_b[i] += la_ij * upper_intercept
    (new_slope, lbias)
}
