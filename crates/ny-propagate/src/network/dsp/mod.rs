// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Graph builders for DSP kernel verification.
//!
//! These builders construct [`GraphNetwork`]s that represent common DSP
//! operations as compositions of existing ny layers, enabling
//! bound propagation (IBP, CROWN, α-CROWN) without new layer types.
//!
//! Part of #3252 — avoice DSP kernel verification.

mod envelope_follower;
mod kokoro_forward_stft;
mod mel128;
#[cfg(test)]
mod tests;

pub use envelope_follower::{
    build_envelope_follower_step_graph, build_gain_reduction_graph, build_one_pole_step_graph,
    verify_envelope_follower_conservative, EnvelopeFollowerParams, GainReductionParams,
};
#[cfg(test)]
pub(crate) use kokoro_forward_stft::kokoro_forward_stft_frame_count;
pub use kokoro_forward_stft::{
    build_kokoro_forward_stft_full_graph, build_kokoro_forward_stft_magnitude_graph,
    build_kokoro_forward_stft_phase_graph, KOKORO_STFT_FREQ_BINS, KOKORO_STFT_HOP,
    KOKORO_STFT_MAG_EPS, KOKORO_STFT_N_FFT, KOKORO_STFT_PAD,
};
pub use mel128::build_qwen3_mel128_graph;
#[cfg(test)]
pub(crate) use mel128::{
    qwen3_mel128_frame_count, QWEN3_SPEAKER_FREQ_BINS, QWEN3_SPEAKER_HOP, QWEN3_SPEAKER_LOG_FLOOR,
    QWEN3_SPEAKER_MAG_EPS, QWEN3_SPEAKER_MELS, QWEN3_SPEAKER_N_FFT, QWEN3_SPEAKER_PAD,
};

use ndarray::arr2;
use ny_core::Result;

use crate::layers::arithmetic::MulConstantLayer;
use crate::layers::trigonometric::TanhLayer;
use crate::layers::{Layer, LinearLayer};

use super::core::{GraphNetwork, GraphNode};

/// Build a waveshaper (tanh soft-clip) verification graph for a fixed drive.
///
/// Implements `f(x) = tanh(d·x) / tanh(d)` as a 3-node graph:
///
/// ```text
/// x → MulConstant(d) → Tanh → MulConstant(1/tanh(d)) → y
/// ```
///
/// All operations are pointwise, so the graph works for any input shape.
/// For `drive < 1e-6`, the function approaches identity (L'Hôpital), so
/// we return a single `MulConstant(1.0)` node.
///
/// # Mathematical properties
///
/// - Monotonically increasing in x for d > 0
/// - |f(x, d)| < 1 for |x| ≤ 1 when d > 0
/// - Odd function: f(-x, d) = -f(x, d)
///
/// # Reference
///
/// Zölzer, "DAFX: Digital Audio Effects" (2nd ed., 2011), Chapter 5.
/// avoice `crates/avoice-music/src/synth/bass.rs:157-164`.
pub fn build_waveshaper_graph(drive: f32) -> Result<GraphNetwork> {
    let mut graph = GraphNetwork::new();

    if drive < 1e-6 {
        // Identity case: tanh(d·x)/tanh(d) → x as d → 0
        graph.try_add_node(GraphNode::from_input(
            "identity",
            Layer::MulConstant(MulConstantLayer::scalar(1.0)),
        ))?;
        graph.set_output("identity");
        return Ok(graph);
    }

    // Compute normalization in f64 for precision, then cast to f32.
    let tanh_d = (drive as f64).tanh() as f32;

    // Node 1: Scale input by drive parameter
    graph.try_add_node(GraphNode::from_input(
        "mul_drive",
        Layer::MulConstant(MulConstantLayer::scalar(drive)),
    ))?;

    // Node 2: Apply tanh nonlinearity
    graph.try_add_node(GraphNode::new(
        "tanh",
        Layer::Tanh(TanhLayer::new()),
        vec!["mul_drive".to_string()],
    ))?;

    // Node 3: Normalize by 1/tanh(d) to preserve unit amplitude
    graph.try_add_node(GraphNode::new(
        "mul_normalize",
        Layer::MulConstant(MulConstantLayer::scalar(1.0 / tanh_d)),
        vec!["tanh".to_string()],
    ))?;

    graph.set_output("mul_normalize");
    Ok(graph)
}

/// Biquad IIR filter coefficients (Direct Form II Transposed).
///
/// Feedforward: `b0, b1, b2`. Feedback: `a1, a2`.
/// Follows the Audio EQ Cookbook convention where `a0 = 1` (pre-normalized).
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoefficients {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

/// Build a single-step biquad IIR filter as a GraphNetwork.
///
/// Implements one timestep of the Direct Form II Transposed biquad:
///
/// ```text
/// y[n]  = b0·x[n] + z1[n-1]
/// z1[n] = (b1 - a1·b0)·x[n] - a1·z1[n-1] + z2[n-1]
/// z2[n] = (b2 - a2·b0)·x[n] - a2·z1[n-1]
/// ```
///
/// This is a purely linear system, so the entire step is a single 3×3
/// `LinearLayer` with no bias:
///
/// ```text
/// [y]      [b0,         1,   0] [x]
/// [z1]  =  [b1-a1·b0, -a1,  1] [z1_prev]
/// [z2]     [b2-a2·b0, -a2,  0] [z2_prev]
/// ```
///
/// Since all operations are linear, IBP and CROWN give **exact bounds**
/// (zero relaxation gap).
///
/// # Inputs
///
/// A 3-element tensor `[x, z1_prev, z2_prev]` where:
/// - `x`: current audio sample
/// - `z1_prev`, `z2_prev`: filter state from previous timestep
///
/// # Outputs
///
/// A 3-element tensor `[y, z1_new, z2_new]`.
///
/// # Reference
///
/// Bristow-Johnson, "Cookbook formulae for audio EQ biquad filter coefficients"
/// (Audio EQ Cookbook, 2005).
/// avoice `crates/avoice-mixer/src/eq.rs:99-104`.
pub fn build_biquad_single_step_graph(coeff: &BiquadCoefficients) -> Result<GraphNetwork> {
    let BiquadCoefficients { b0, b1, b2, a1, a2 } = *coeff;

    // State-space weight matrix: output = W * input, no bias.
    //   Row 0 (y):      b0·x + 1·z1 + 0·z2
    //   Row 1 (z1_new): (b1-a1·b0)·x + (-a1)·z1 + 1·z2
    //   Row 2 (z2_new): (b2-a2·b0)·x + (-a2)·z1 + 0·z2
    let weight = arr2(&[
        [b0, 1.0, 0.0],
        [b1 - a1 * b0, -a1, 1.0],
        [b2 - a2 * b0, -a2, 0.0],
    ]);

    let linear = LinearLayer::new(weight, None)?;

    let mut graph = GraphNetwork::new();
    graph.try_add_node(GraphNode::from_input("biquad_step", Layer::Linear(linear)))?;
    graph.set_output("biquad_step");
    Ok(graph)
}
