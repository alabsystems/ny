// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Streaming chunk boundary verification for audio vocoder models.
//!
//! Implements the CROWN verification approach for avoice streaming TTS (#3500):
//! - Bound vocoder output at chunk boundaries (first/last N samples)
//! - Prove crossfade overlap-add energy stays in [0.5, 1.5] of steady-state
//! - Verify phase alignment of overlapping regions
//!
//! The real Kokoro ONNX structural surface now lives in
//! `crates/ny-onnx/src/tests/core/avoice/kokoro_vocoder/`; these tests
//! remain the fast proof harness for overlap-add, energy, and phase continuity
//! while the real-weight runtime lane is still too expensive for CPU-only unit
//! tests (`T=6` IBP exceeded the 600s measurement budget on 2026-03-11).
//!
//! Design: `designs/2026-03-11-issue-3500-kokoro-vocoder-boundary-surface.md`
//! Follow-up: `designs/2026-03-11-issue-3500-kokoro-runtime-floor-followup.md`

use super::*;
use ndarray::{Array1, Array2, ArrayD};
use proptest::prelude::Strategy;

/// Number of boundary samples in the crossfade region.
/// Real avoice uses 240 (10ms at 24kHz); we use 4 for fast testing.
pub(super) const BOUNDARY_SAMPLES: usize = 4;

/// Build a synthetic vocoder-like sequential network.
///
/// Architecture: Conv1d(1->2, k=3) -> SiLU -> Conv1d(2->1, k=3)
///
/// This mimics a simplified 1D CNN vocoder that takes mel features
/// and produces a waveform. The SiLU activation is common in modern
/// vocoders (Kokoro, BigVGAN).
///
/// Returns (network, output_length) where output_length is the spatial
/// dimension of the final output.
pub(super) fn build_synthetic_vocoder(input_length: usize) -> (Network, usize) {
    let mut network = Network::new();

    // Layer 1: Conv1d(in_c=1, out_c=2, kernel=3, stride=1, pad=0)
    // Output length: input_length - 2
    let mut k1 = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 3]));
    // Smoothing filter on channel 0
    k1[[0, 0, 0]] = 0.3;
    k1[[0, 0, 1]] = 0.4;
    k1[[0, 0, 2]] = 0.3;
    // Edge-detecting filter on channel 1
    k1[[1, 0, 0]] = -0.5;
    k1[[1, 0, 1]] = 1.0;
    k1[[1, 0, 2]] = -0.5;
    let b1 = Array1::from_elem(2, 0.1f32);
    let conv1 = Conv1dLayer::with_input_length(k1, Some(b1), 1, 0, input_length)
        .expect("valid conv1 params");
    network.add_layer(Layer::Conv1d(conv1));

    let mid_length = input_length - 2;

    // Activation: SiLU (x * sigmoid(x)) — common in modern vocoders
    network.add_layer(Layer::SiLU(SiLULayer));

    // Layer 2: Conv1d(in_c=2, out_c=1, kernel=3, stride=1, pad=0)
    // Output length: mid_length - 2
    let mut k2 = ArrayD::zeros(ndarray::IxDyn(&[1, 2, 3]));
    // Channel 0 contribution (smoothed features)
    k2[[0, 0, 0]] = 0.5;
    k2[[0, 0, 1]] = 0.3;
    k2[[0, 0, 2]] = 0.2;
    // Channel 1 contribution (edge detail)
    k2[[0, 1, 0]] = 0.1;
    k2[[0, 1, 1]] = 0.2;
    k2[[0, 1, 2]] = 0.1;
    let b2 = Array1::from_elem(1, 0.0f32);
    // Input to conv2 is [2, mid_length] flattened = 2 * mid_length
    let conv2 =
        Conv1dLayer::with_input_length(k2, Some(b2), 1, 0, mid_length).expect("valid conv2 params");
    network.add_layer(Layer::Conv1d(conv2));

    let output_length = mid_length - 2;
    (network, output_length)
}

/// Create a bounded input representing mel spectrogram features with small perturbation.
///
/// Shape: [1, input_length] (1 channel, input_length time steps)
/// Center: 0.5 (typical normalized mel value)
/// Epsilon: perturbation radius modeling upstream TTS uncertainty
pub(super) fn mel_input(input_length: usize, epsilon: f32) -> BoundedTensor {
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, input_length]), 0.5);
    BoundedTensor::from_epsilon(center, epsilon).expect("valid mel input")
}

/// Extract boundary sample bounds from a CROWN output tensor.
///
/// Returns (first_N_lower, first_N_upper, last_N_lower, last_N_upper)
/// where each is a Vec<f32> of length `boundary_size`.
pub(super) fn extract_boundary_bounds(
    output: &BoundedTensor,
    output_length: usize,
    boundary_size: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let flat = output.flatten();
    let lower = flat.lower().as_slice().expect("contiguous lower");
    let upper = flat.upper().as_slice().expect("contiguous upper");

    // Output shape is [1, output_length], flattened to [output_length]
    assert!(
        lower.len() >= output_length,
        "output too short: {} < {}",
        lower.len(),
        output_length
    );
    assert!(
        boundary_size <= output_length,
        "boundary_size {} > output_length {}",
        boundary_size,
        output_length
    );

    let first_lower = lower[..boundary_size].to_vec();
    let first_upper = upper[..boundary_size].to_vec();
    let last_lower = lower[output_length - boundary_size..output_length].to_vec();
    let last_upper = upper[output_length - boundary_size..output_length].to_vec();

    (first_lower, first_upper, last_lower, last_upper)
}

/// Compute crossfade overlap-add bounds from two chunks' boundary bounds.
///
/// The overlap-add formula is:
///   output[i] = fade_out[i] * chunk_A[end - N + i] + fade_in[i] * chunk_B[i]
/// where:
///   fade_out[i] = (N - i) / N   (linear fade from 1 to 0)
///   fade_in[i]  = i / N          (linear fade from 0 to 1)
///
/// Since fade_out, fade_in >= 0 and chunk bounds are interval-valued,
/// the crossfade output bounds are:
///   lower[i] = fade_out[i] * min(l_A, u_A) + fade_in[i] * min(l_B, u_B)
///            = fade_out[i] * l_A + fade_in[i] * l_B  (when l_A, l_B are true lowers)
///   upper[i] = fade_out[i] * u_A + fade_in[i] * u_B
///
/// This is sound because the crossfade weights are non-negative constants.
///
/// Reference: designs/2026-03-10-avoice-crown-capability-triage.md, Step 5
pub(super) fn crossfade_overlap_add_bounds(
    chunk_a_last_lower: &[f32],
    chunk_a_last_upper: &[f32],
    chunk_b_first_lower: &[f32],
    chunk_b_first_upper: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let n = chunk_a_last_lower.len();
    debug_assert!(n > 0, "crossfade requires at least one boundary sample");
    assert_eq!(n, chunk_a_last_upper.len());
    assert_eq!(n, chunk_b_first_lower.len());
    assert_eq!(n, chunk_b_first_upper.len());

    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);

    for i in 0..n {
        let fade_out = (n - i) as f32 / n as f32;
        let fade_in = i as f32 / n as f32;

        // Sound interval arithmetic: non-negative weights * interval bounds
        // For lower bound: positive_weight * lower_endpoint
        // For upper bound: positive_weight * upper_endpoint
        let l_a = chunk_a_last_lower[i];
        let u_a = chunk_a_last_upper[i];
        let l_b = chunk_b_first_lower[i];
        let u_b = chunk_b_first_upper[i];

        // Handle potential negative values correctly
        let lo = fade_out * l_a.min(u_a) + fade_in * l_b.min(u_b);
        let hi = fade_out * l_a.max(u_a) + fade_in * l_b.max(u_b);

        lower.push(lo);
        upper.push(hi);
    }

    (lower, upper)
}

/// Compute the energy bounds for a signal with interval-valued samples.
///
/// Energy[i] = sample[i]^2. Given sample[i] ∈ [l, u]:
/// - If l >= 0: energy ∈ [l^2, u^2]
/// - If u <= 0: energy ∈ [u^2, l^2]
/// - If l < 0 < u: energy ∈ [0, max(l^2, u^2)]
pub(super) fn energy_bounds(lower: &[f32], upper: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = lower.len();
    let mut e_lower = Vec::with_capacity(n);
    let mut e_upper = Vec::with_capacity(n);

    for i in 0..n {
        let l = lower[i];
        let u = upper[i];

        if l >= 0.0 {
            e_lower.push(l * l);
            e_upper.push(u * u);
        } else if u <= 0.0 {
            e_lower.push(u * u);
            e_upper.push(l * l);
        } else {
            // Interval straddles zero
            e_lower.push(0.0);
            e_upper.push(l.abs().max(u.abs()).powi(2));
        }
    }

    (e_lower, e_upper)
}

// ---------------------------------------------------------------------------
// Shared test helpers
// ---------------------------------------------------------------------------

/// Construct output constraints for boundary samples.
///
/// Creates an `OutputConstraints` matrix that constrains the first and last
/// `boundary_size` output samples to lie within [-bound, +bound].
///
/// This demonstrates the constraint construction that would be used with
/// the real Kokoro vocoder model for INVPROP backward propagation.
fn boundary_output_constraints(
    output_dim: usize,
    boundary_size: usize,
    amplitude_bound: f32,
) -> OutputConstraints {
    // Each boundary sample gets two rows: sample[i] <= bound, -sample[i] <= bound
    let num_boundary_outputs = 2 * boundary_size; // first + last
    let num_constraints = 2 * num_boundary_outputs; // upper + lower per sample

    let mut a_matrix = Array2::<f32>::zeros((num_constraints, output_dim));
    let mut rhs = Array1::<f32>::zeros(num_constraints);

    let mut row = 0;
    // First boundary: indices 0..boundary_size
    for i in 0..boundary_size {
        // sample[i] <= amplitude_bound
        a_matrix[[row, i]] = 1.0;
        rhs[row] = amplitude_bound;
        row += 1;
        // -sample[i] <= amplitude_bound  (i.e. sample[i] >= -amplitude_bound)
        a_matrix[[row, i]] = -1.0;
        rhs[row] = amplitude_bound;
        row += 1;
    }
    // Last boundary: indices (output_dim - boundary_size)..output_dim
    for i in 0..boundary_size {
        let idx = output_dim - boundary_size + i;
        // sample[idx] <= amplitude_bound
        a_matrix[[row, idx]] = 1.0;
        rhs[row] = amplitude_bound;
        row += 1;
        // -sample[idx] <= amplitude_bound
        a_matrix[[row, idx]] = -1.0;
        rhs[row] = amplitude_bound;
        row += 1;
    }

    OutputConstraints::new(a_matrix, rhs, true).expect("valid boundary constraints")
}

/// Generate a random concrete input within the mel bounds.
///
/// Returns an ArrayD of shape [1, input_length] with values in
/// [0.5 - epsilon, 0.5 + epsilon].
fn concrete_input_strategy(
    input_length: usize,
    epsilon: f32,
) -> impl Strategy<Value = ArrayD<f32>> {
    let center = 0.5_f32;
    let lo = center - epsilon;
    let hi = center + epsilon;
    proptest::collection::vec(lo..=hi, input_length).prop_map(move |vals| {
        ArrayD::from_shape_vec(ndarray::IxDyn(&[1, input_length]), vals).expect("valid shape")
    })
}

mod boundary_samples;
mod constraints;
mod crossfade;
mod proptests;
