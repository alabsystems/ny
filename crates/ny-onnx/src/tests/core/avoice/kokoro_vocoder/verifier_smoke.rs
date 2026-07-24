// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Kokoro vocoder verifier-smoke helpers.
//!
//! Graph construction and scalar-energy assertions for the Kokoro vocoder
//! verifier smokes. Test functions live in the root
//! `avoice/verifier_smoke.rs` entrypoint (#4029).
//!
//! Property: `sum(prefix_output^2) in [0, +inf)` — a sum of squares is
//! always non-negative. The scalar energy head avoids exposing the raw
//! ~15k prefix output coordinates in the `VerificationSpec`.
//!
//! Reference: designs/2026-03-17-issue-4029-kokoro-prefix-verifier-smoke.md

use super::super::common;
use super::graph_support::{first_conv_transpose_node, vocoder_prefix_subgraph};
use super::model::{
    bounded_kokoro_features_input, load_kokoro_vocoder_with_fixed_aux,
    KOKORO_VOCODER_MIN_FIXED_AUX_T,
};
use ny_core::VerificationResult;
use ny_propagate::layers::{PowConstantLayer, ReduceSumLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use ny_tensor::BoundedTensor;

fn build_kokoro_prefix_energy_graph(
    full_graph: GraphNetwork,
    input: &BoundedTensor,
) -> GraphNetwork {
    let cut_node = first_conv_transpose_node(&full_graph);
    let prefix = vocoder_prefix_subgraph(&full_graph, &cut_node);
    let prefix_output_name = prefix.output_name().to_string();

    // Discover prefix output rank via a quick IBP pass for axis construction.
    let prefix_ibp = prefix
        .propagate_ibp(input)
        .expect("prefix IBP sanity check should succeed");
    let axes: Vec<i64> = (0..prefix_ibp.shape().len() as i64).collect();

    eprintln!(
        "kokoro prefix energy setup: prefix {} nodes, output '{}' shape {:?}",
        prefix.num_nodes(),
        prefix_output_name,
        prefix_ibp.shape()
    );

    // Build the energy graph: prefix -> PowConstant(2.0) -> ReduceSum(all)
    // Node names follow the design doc (Packet B1 step 6) for stability.
    let mut energy_graph = prefix;
    energy_graph.add_node(GraphNode::new(
        "kokoro_prefix_energy_sq",
        Layer::PowConstant(PowConstantLayer::new(2.0)),
        vec![prefix_output_name],
    ));
    energy_graph.add_node(GraphNode::new(
        "kokoro_prefix_energy",
        Layer::ReduceSum(ReduceSumLayer::new(axes, false)),
        vec!["kokoro_prefix_energy_sq".to_string()],
    ));
    energy_graph.set_output("kokoro_prefix_energy");
    energy_graph
}

/// Build the Kokoro vocoder prefix energy graph and bounded input.
///
/// Loads the real Kokoro vocoder model, freezes auxiliary inputs, extracts
/// the smallest viable prefix (up to the first ConvTranspose1d), and appends
/// a scalar energy head: `PowConstant(2.0) → ReduceSum(all axes)`.
///
/// Returns `(energy_graph, bounded_features_input)`.
pub(crate) fn kokoro_prefix_energy_verifier_setup() -> (GraphNetwork, BoundedTensor) {
    let model = load_kokoro_vocoder_with_fixed_aux(KOKORO_VOCODER_MIN_FIXED_AUX_T);
    let full_graph = model
        .to_graph_network()
        .expect("kokoro vocoder graph conversion should succeed");
    let input = bounded_kokoro_features_input(&model, KOKORO_VOCODER_MIN_FIXED_AUX_T, 1e-3);
    let energy_graph = build_kokoro_prefix_energy_graph(full_graph, &input);
    (energy_graph, input)
}

pub(crate) fn kokoro_graph_model_round_trip_prefix_energy_verifier_setup(
) -> (GraphNetwork, BoundedTensor) {
    let model = load_kokoro_vocoder_with_fixed_aux(KOKORO_VOCODER_MIN_FIXED_AUX_T);
    let input = bounded_kokoro_features_input(&model, KOKORO_VOCODER_MIN_FIXED_AUX_T, 1e-3);
    let full_graph = model
        .to_graph_model()
        .build_graph_network(crate::GraphNetworkOptions::default())
        .expect("kokoro GraphModel round-trip build should succeed");
    let energy_graph = build_kokoro_prefix_energy_graph(full_graph, &input);
    (energy_graph, input)
}

/// Assert that a scalar energy verification result is non-negative when
/// Verified, or structurally sound when Unknown. Timeout and Violated are
/// hard failures.
pub(crate) fn assert_kokoro_prefix_energy_verifier_result(
    result: &VerificationResult,
    label: &str,
) {
    match result {
        VerificationResult::Verified {
            output_bounds,
            actual_method,
            ..
        } => {
            assert_eq!(
                output_bounds.len(),
                1,
                "{label}: verifier output should be scalar (1 bound), got {}",
                output_bounds.len()
            );
            assert!(
                output_bounds[0].lower() >= 0.0,
                "{label}: energy lower bound should be non-negative, got {}",
                output_bounds[0].lower()
            );
            eprintln!(
                "{label}: verified via {:?}, bounds=[{}, {}]",
                actual_method,
                output_bounds[0].lower(),
                output_bounds[0].upper()
            );
        }
        VerificationResult::Unknown { reason, bounds, .. } => {
            common::assert_unknown_verifier_bounds_sound(bounds, 1, label);
            eprintln!(
                "{label}: unknown ({reason:?}), {} output bounds all finite and ordered, \
                 bounds={bounds:?}",
                bounds.len()
            );
        }
        VerificationResult::Timeout { .. } => {
            panic!("{label}: timed out — budget may need increase");
        }
        VerificationResult::Violated {
            counterexample,
            output,
            ..
        } => {
            panic!(
                "{label}: violated — energy (sum of squares) should always be >= 0; \
                 counterexample={counterexample:?}, output={output:?}"
            );
        }
    }
}
