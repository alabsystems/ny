// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared alpha optimization helper kernels.
//!
//! These free functions contain the core optimization logic shared between
//! `AlphaState` (sequential networks, Vec-indexed) and `GraphAlphaState`
//! (DAG networks, BTreeMap-indexed). Extracted per #2201 to prevent divergence.

use ndarray::Array1;
use ny_core::Result;
use ny_tensor::BoundedTensor;

use super::AdamParams;

/// Extract contiguous lower/upper slices from a BoundedTensor.
///
/// Uses `as_standard_layout().into_owned()` to handle non-contiguous tensors (#1933).
/// Returns (lower_owned, upper_owned) as contiguous Array1 views.
pub(super) fn extract_contiguous_bounds(
    pre_flat: &BoundedTensor,
) -> Result<(ndarray::ArrayD<f32>, ndarray::ArrayD<f32>)> {
    let lower_std = pre_flat.lower().as_standard_layout().into_owned();
    let upper_std = pre_flat.upper().as_standard_layout().into_owned();
    Ok((lower_std, upper_std))
}

/// Initialize alpha and unstable mask from pre-activation bounds for one ReLU group.
///
/// Classifies each neuron as:
/// - Always positive (l >= 0): alpha = 1, mask = false
/// - Always negative (u <= 0): alpha = 0, mask = false
/// - Unstable (l < 0 < u): alpha = adaptive heuristic (1 if u > -l, else 0), mask = true
///
/// Returns (alpha, mask) arrays of length `num_neurons`.
pub(super) fn init_alpha_from_bounds(
    lower_arr: &[f32],
    upper_arr: &[f32],
) -> (Array1<f32>, Array1<bool>) {
    let num_neurons = lower_arr.len();
    let mut alpha = Array1::<f32>::zeros(num_neurons);
    let mut mask = Array1::<bool>::from_elem(num_neurons, false);

    for i in 0..num_neurons {
        let l = lower_arr[i];
        let u = upper_arr[i];

        if l >= 0.0 {
            // Always positive: alpha = 1 (identity)
            alpha[i] = 1.0;
            mask[i] = false;
        } else if u <= 0.0 {
            // Always negative: alpha = 0
            alpha[i] = 0.0;
            mask[i] = false;
        } else {
            // Crossing: unstable, initialize with adaptive heuristic
            // α = 1 if u > -l (more positive area), else 0 (more negative area)
            alpha[i] = if u > -l { 1.0 } else { 0.0 };
            mask[i] = true;
        }
    }

    (alpha, mask)
}

/// SGD with momentum update for alpha parameters (inner loop).
///
/// For each unstable neuron (mask = true):
/// - vel = momentum * vel - lr * gradient
/// - alpha += vel
/// - Clamp to [0, 1] and sanitize NaN (#2025)
pub(super) fn update_alphas_sgd(
    alpha: &mut Array1<f32>,
    gradient: &Array1<f32>,
    mask: &Array1<bool>,
    vel: &mut Array1<f32>,
    learning_rate: f32,
    momentum: f32,
) {
    for i in 0..alpha.len() {
        if mask[i] {
            vel[i] = momentum * vel[i] - learning_rate * gradient[i];
            alpha[i] += vel[i];
            alpha[i] = alpha[i].clamp(0.0, 1.0);
            if alpha[i].is_nan() {
                tracing::warn!(
                    "NaN in alpha update (SGD), resetting alpha to 0.5 and velocity to 0.0: neuron_idx={i}"
                );
                alpha[i] = 0.5;
                vel[i] = 0.0;
            }
        }
    }
}

/// Adam optimizer update for alpha parameters (inner loop).
///
/// For each unstable neuron (mask = true):
/// - m = β₁ * m + (1 - β₁) * grad
/// - v = β₂ * v + (1 - β₂) * grad²
/// - m_hat = m / (1 - β₁^t), v_hat = v / (1 - β₂^t)
/// - alpha -= lr * m_hat / (√v_hat + ε)
/// - Clamp to [0, 1] and sanitize NaN (#2025)
pub(super) fn update_alphas_adam(
    alpha: &mut Array1<f32>,
    gradient: &Array1<f32>,
    mask: &Array1<bool>,
    m: &mut Array1<f32>,
    v: &mut Array1<f32>,
    params: &AdamParams,
) {
    let t_f = params.t.max(1) as f32;
    // Guard: beta=1.0 makes bias_correction=0, causing division by zero (#2315).
    let bias_correction1 = (1.0 - params.beta1.powf(t_f)).max(f32::EPSILON);
    let bias_correction2 = (1.0 - params.beta2.powf(t_f)).max(f32::EPSILON);

    for i in 0..alpha.len() {
        if mask[i] {
            let g = gradient[i];

            m[i] = params.beta1 * m[i] + (1.0 - params.beta1) * g;
            v[i] = params.beta2 * v[i] + (1.0 - params.beta2) * g * g;

            let m_hat = m[i] / bias_correction1;
            let v_hat = v[i] / bias_correction2;

            alpha[i] -= params.learning_rate * m_hat / (v_hat.sqrt() + params.epsilon);

            // Clamp to [0, 1] and sanitize NaN (#2025).
            // NaN.clamp(0.0, 1.0) returns NaN in Rust, so NaN from gradient
            // computation would permanently corrupt alpha/m/v state.
            // Matches sanitize_alpha() in beta_crown/state/alpha/mod.rs.
            alpha[i] = alpha[i].clamp(0.0, 1.0);
            if alpha[i].is_nan() {
                tracing::warn!(
                    "NaN in alpha update (Adam), resetting alpha to 0.5 and m/v to 0.0: neuron_idx={i}"
                );
                alpha[i] = 0.5;
                m[i] = 0.0;
                v[i] = 0.0;
            }
        }
    }
}
