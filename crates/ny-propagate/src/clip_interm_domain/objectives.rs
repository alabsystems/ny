// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Objective neuron selection for intermediate domain clipping.
//!
//! Selects the top-k most impactful unstable neurons per layer for
//! constrained concretization using a kFSB-style scoring heuristic.

use ndarray::Array1;

/// Select top-k objective neurons per layer for tightening.
///
/// Uses a kFSB-style scoring heuristic:
/// `score = intercept * coeff_mag`
/// where `intercept = (-l*u)/(u-l)` for unstable neurons and `coeff_mag` is the
/// CROWN coefficient magnitude (L1 norm of linear bound coefficients).
///
/// # Arguments
///
/// * `lower_bounds` - Pre-activation lower bounds, shape: `(n_neurons,)`
/// * `upper_bounds` - Pre-activation upper bounds, shape: `(n_neurons,)`
/// * `coeff_magnitudes` - CROWN coefficient magnitudes, shape: `(n_neurons,)`
/// * `topk` - Number of neurons to select
///
/// # Returns
///
/// Indices of selected neurons (up to topk, only unstable neurons).
///
/// # References
///
/// - `designs/2026-01-29-clip-interm-domain.md` Section "Select objectives"
/// - `alpha-beta-CROWN/complete_verifier/docs/abcrown_all_params.yaml:201`
pub fn select_objective_neurons(
    lower_bounds: &Array1<f32>,
    upper_bounds: &Array1<f32>,
    coeff_magnitudes: &Array1<f32>,
    topk: usize,
) -> Vec<usize> {
    let n = lower_bounds.len();

    // Validate array dimensions match (#2136: active in release builds).
    assert_eq!(
        lower_bounds.len(),
        upper_bounds.len(),
        "bounds length mismatch"
    );
    assert_eq!(
        lower_bounds.len(),
        coeff_magnitudes.len(),
        "coeff_magnitudes length mismatch"
    );

    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n);

    for i in 0..n {
        let l = lower_bounds[i];
        let u = upper_bounds[i];

        // Only consider unstable neurons (l < 0 < u)
        if l >= 0.0 || u <= 0.0 {
            continue;
        }

        // Score = intercept * coeff_mag
        // intercept = (-l*u) / (u-l) for triangle relaxation
        let width = u - l;
        if width < 1e-10 {
            continue;
        }

        let intercept = (-l * u) / width;
        let score = intercept * coeff_magnitudes[i];

        scores.push((i, score));
    }

    // Sort by score descending (NaN-scored items sort last — #2995)
    scores.sort_by(|a, b| crate::cmp_utils::nan_last_descending_cmp(&a.1, &b.1));

    // Take top-k
    scores.iter().take(topk).map(|(idx, _)| *idx).collect()
}
