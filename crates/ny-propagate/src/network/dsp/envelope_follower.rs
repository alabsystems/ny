// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Envelope follower verification: gain reduction, one-pole smoother,
//! and conservative combined verification.
//!
//! Part of #3252 — avoice DSP kernel verification.

use ndarray::{arr2, ArrayD, IxDyn};
use ny_core::Result;

use crate::layers::activations::ClipLayer;
use crate::layers::arithmetic::{AddConstantLayer, MulConstantLayer};
use crate::layers::misc::ReciprocalLayer;
use crate::layers::{Layer, LinearLayer};

use crate::network::core::{GraphNetwork, GraphNode};

/// Envelope follower gain reduction parameters.
///
/// These are fixed constants for a given compressor configuration.
/// The gain function maps an envelope value to a gain multiplier.
#[derive(Debug, Clone, Copy)]
pub struct GainReductionParams {
    /// Threshold in linear amplitude (e.g., 10^(-30/20) ≈ 0.0316 for -30 dB).
    /// Must be strictly positive.
    pub threshold: f32,
    /// Compression ratio ∈ [0, 1]. 0 = no compression, 1 = full limiting.
    pub ratio: f32,
    /// Maximum attenuation floor (e.g., 0.01 for -40 dB).
    /// Must be in (0, 1].
    pub max_atten: f32,
}

/// Build a stateless gain reduction function as a GraphNetwork.
///
/// Implements the piecewise gain function from avoice's envelope follower:
///
/// ```text
/// gain(e) = 1.0                                             if e ≤ threshold
/// gain(e) = max(1 - ratio · min((e - threshold)/e, 1), max_atten)  otherwise
/// ```
///
/// Decomposed into a chain of unary operations:
///
/// ```text
/// e → Clip(threshold, 1) → Reciprocal → MulConstant(-threshold)
///   → AddConstant(1) → Clip(0, 1) → MulConstant(-ratio)
///   → AddConstant(1) → Clip(max_atten, 1) → gain
/// ```
///
/// The initial Clip ensures `e ≥ threshold`, which:
/// 1. Prevents division by zero in the Reciprocal
/// 2. Makes the `e ≤ threshold` branch produce gain = 1.0 (since
///    `1 - threshold/threshold = 0`, so `1 - ratio·0 = 1`)
///
/// # Mathematical properties
///
/// - gain ∈ [max_atten, 1.0] for all e ∈ [0, 1] (guaranteed by Clip clamps)
/// - Monotonically non-increasing in e for e > threshold
/// - Continuous at e = threshold (gain = 1.0 from both branches)
///
/// # Reference
///
/// Giannoulis et al., "Digital Dynamic Range Compressor Design" (JAES, 2012).
/// avoice `crates/avoice-mixer/src/ducking.rs:58-86`.
/// Design: `designs/2026-03-04-envelope-follower-verification.md` Task 1.
pub fn build_gain_reduction_graph(params: &GainReductionParams) -> Result<GraphNetwork> {
    let GainReductionParams {
        threshold,
        ratio,
        max_atten,
    } = *params;

    let mut graph = GraphNetwork::new();

    // Node 1: Clip envelope to [threshold, 1.0].
    // For e ≤ threshold, this clamps to threshold, making the rest of the
    // chain produce gain = 1.0. Also ensures Reciprocal input > 0.
    graph.try_add_node(GraphNode::from_input(
        "clip_env",
        Layer::Clip(ClipLayer::new(threshold, 1.0)),
    ))?;

    // Node 2: Reciprocal → 1/clip_env. Safe because clip_env ≥ threshold > 0.
    graph.try_add_node(GraphNode::new(
        "recip",
        Layer::Reciprocal(ReciprocalLayer::new()),
        vec!["clip_env".to_string()],
    ))?;

    // Node 3: Multiply by -threshold → -threshold/clip_env.
    graph.try_add_node(GraphNode::new(
        "mul_neg_threshold",
        Layer::MulConstant(MulConstantLayer::scalar(-threshold)),
        vec!["recip".to_string()],
    ))?;

    // Node 4: Add 1.0 → 1 - threshold/clip_env = compression ratio.
    // When clip_env = threshold: 1 - 1 = 0.
    // When clip_env > threshold: positive, < 1.
    graph.try_add_node(GraphNode::new(
        "compression_ratio",
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[]), 1.0))),
        vec!["mul_neg_threshold".to_string()],
    ))?;

    // Node 5: Clip to [0, 1] = min(compression_ratio, 1) with floor at 0.
    graph.try_add_node(GraphNode::new(
        "clip_ratio",
        Layer::Clip(ClipLayer::new(0.0, 1.0)),
        vec!["compression_ratio".to_string()],
    ))?;

    // Node 6: Multiply by -ratio → -ratio · clipped_ratio.
    graph.try_add_node(GraphNode::new(
        "mul_neg_ratio",
        Layer::MulConstant(MulConstantLayer::scalar(-ratio)),
        vec!["clip_ratio".to_string()],
    ))?;

    // Node 7: Add 1.0 → 1 - ratio · clipped_ratio.
    graph.try_add_node(GraphNode::new(
        "one_minus",
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[]), 1.0))),
        vec!["mul_neg_ratio".to_string()],
    ))?;

    // Node 8: Final clip to [max_atten, 1.0].
    // The max(·, max_atten) floor and the ≤ 1 ceiling.
    graph.try_add_node(GraphNode::new(
        "gain_out",
        Layer::Clip(ClipLayer::new(max_atten, 1.0)),
        vec!["one_minus".to_string()],
    ))?;

    graph.set_output("gain_out");
    Ok(graph)
}

/// Envelope follower parameters for conservative verification.
///
/// Combines one-pole smoother coefficients with gain reduction parameters.
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeFollowerParams {
    /// Attack coefficient: `c_attack = 1 - exp(-1/(sr·τ_attack))`.
    /// Higher value → faster response to rising input.
    pub c_attack: f32,
    /// Release coefficient: `c_release = 1 - exp(-1/(sr·τ_release))`.
    /// Lower value → slower response to falling input.
    pub c_release: f32,
    /// Gain reduction parameters (threshold, ratio, max attenuation).
    pub gain: GainReductionParams,
}

/// Build a single-step one-pole smoother as a GraphNetwork.
///
/// Implements one timestep of the IIR envelope follower:
///
/// ```text
/// envelope[n] = c · rms[n] + (1-c) · envelope[n-1]
/// ```
///
/// This is a purely linear system (weighted average), so a single 1×2
/// `LinearLayer` gives **exact bounds** for both IBP and CROWN.
///
/// # Inputs
///
/// A 2-element tensor `[rms, envelope_prev]` where:
/// - `rms`: current RMS amplitude measurement
/// - `envelope_prev`: envelope state from previous timestep
///
/// # Outputs
///
/// A 1-element tensor `[envelope_new]`.
///
/// # Stability
///
/// Since 0 < c < 1, this is a convex combination of rms and envelope_prev.
/// If both inputs are in [0, 1], the output is guaranteed to be in [0, 1].
/// The pole is at (1-c), inside the unit circle → BIBO stable.
///
/// # Reference
///
/// Smith, "Introduction to Digital Filters with Audio Applications"
/// (CCRMA, 2007), Section 1.1.2.
/// Design: `designs/2026-03-04-envelope-follower-verification.md` Task 2.
pub fn build_one_pole_step_graph(coefficient: f32) -> Result<GraphNetwork> {
    let weight = arr2(&[[coefficient, 1.0 - coefficient]]);
    let linear = LinearLayer::new(weight, None)?;

    let mut graph = GraphNetwork::new();
    graph.try_add_node(GraphNode::from_input(
        "one_pole_step",
        Layer::Linear(linear),
    ))?;
    graph.set_output("one_pole_step");
    Ok(graph)
}

/// Build a combined envelope follower step (one-pole + gain reduction).
///
/// Implements a single timestep of the envelope follower with gain reduction:
///
/// ```text
/// envelope[n] = c · rms[n] + (1-c) · envelope[n-1]
/// gain = gain_reduction(envelope[n])
/// ```
///
/// For **conservative verification** (Task 3 from the design), call this
/// twice with `c_attack` and `c_release` separately:
/// - Attack-only gives the upper bound on envelope (faster tracking)
/// - Release-only gives the lower bound on envelope (slower tracking)
///
/// The union of gain bounds from both runs is a sound bound on the
/// actual gain with coefficient switching.
///
/// # Inputs
///
/// A 2-element tensor `[rms, envelope_prev]`.
///
/// # Outputs
///
/// A 1-element tensor `[gain]` where gain ∈ [max_atten, 1.0].
///
/// # Reference
///
/// Giannoulis et al., "Digital Dynamic Range Compressor Design" (JAES, 2012).
/// Design: `designs/2026-03-04-envelope-follower-verification.md` Task 3.
pub fn build_envelope_follower_step_graph(
    coefficient: f32,
    gain_params: &GainReductionParams,
) -> Result<GraphNetwork> {
    let GainReductionParams {
        threshold,
        ratio,
        max_atten,
    } = *gain_params;

    let mut graph = GraphNetwork::new();

    // Stage 1: One-pole smoother
    let weight = arr2(&[[coefficient, 1.0 - coefficient]]);
    let linear = LinearLayer::new(weight, None)?;
    graph.try_add_node(GraphNode::from_input(
        "one_pole_step",
        Layer::Linear(linear),
    ))?;

    // Stage 2: Gain reduction chain
    graph.try_add_node(GraphNode::new(
        "clip_env",
        Layer::Clip(ClipLayer::new(threshold, 1.0)),
        vec!["one_pole_step".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "recip",
        Layer::Reciprocal(ReciprocalLayer::new()),
        vec!["clip_env".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "mul_neg_threshold",
        Layer::MulConstant(MulConstantLayer::scalar(-threshold)),
        vec!["recip".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "compression_ratio",
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[]), 1.0))),
        vec!["mul_neg_threshold".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "clip_ratio",
        Layer::Clip(ClipLayer::new(0.0, 1.0)),
        vec!["compression_ratio".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "mul_neg_ratio",
        Layer::MulConstant(MulConstantLayer::scalar(-ratio)),
        vec!["clip_ratio".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "one_minus",
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[]), 1.0))),
        vec!["mul_neg_ratio".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "gain_out",
        Layer::Clip(ClipLayer::new(max_atten, 1.0)),
        vec!["one_minus".to_string()],
    ))?;

    graph.set_output("gain_out");
    Ok(graph)
}

/// Verify the combined envelope follower using conservative bounding.
///
/// Runs the combined step with both attack and release coefficients,
/// then unions the resulting gain bounds. This gives sound bounds on
/// the actual gain regardless of which coefficient the real system
/// selects at each timestep.
pub fn verify_envelope_follower_conservative(
    params: &EnvelopeFollowerParams,
    input: &ny_tensor::BoundedTensor,
) -> Result<ny_tensor::BoundedTensor> {
    // Attack-only: faster tracking
    let attack_graph = build_envelope_follower_step_graph(params.c_attack, &params.gain)?;
    let attack_bounds = attack_graph.propagate_ibp(input)?;

    // Release-only: slower tracking
    let release_graph = build_envelope_follower_step_graph(params.c_release, &params.gain)?;
    let release_bounds = release_graph.propagate_ibp(input)?;

    // Union of gain bounds: min of lowers, max of uppers
    let lower = attack_bounds
        .lower()
        .iter()
        .zip(release_bounds.lower().iter())
        .map(|(&a, &r)| a.min(r))
        .collect::<Vec<f32>>();
    let upper = attack_bounds
        .upper()
        .iter()
        .zip(release_bounds.upper().iter())
        .map(|(&a, &r)| a.max(r))
        .collect::<Vec<f32>>();

    ny_tensor::BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower)
            .map_err(|e| ny_core::NyError::InternalError(e.to_string()))?,
        ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper)
            .map_err(|e| ny_core::NyError::InternalError(e.to_string()))?,
    )
}
