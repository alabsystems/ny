// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Talker attention verifier-smoke helpers.
//!
//! Graph construction and softmax-specific assertions for the talker
//! verifier smokes. Test functions live in the root
//! `avoice/verifier_smoke.rs` entrypoint (#3950).

use super::super::common;
use super::fixtures::{
    bounded_hidden_states_input, configure_sound_softmax_modes,
    load_talker_attention_with_fixed_aux_for_seq_len,
    talker_attention_softmax_output_graph_for_seq_len,
    talker_attention_softmax_output_graph_real_rope, TALKER_ATTENTION_EPSILON,
    TALKER_ATTENTION_SEQ_LEN, TALKER_ATTENTION_SHORT_SEQ_LEN,
};
use ny_core::{Bound, VerificationResult};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;

/// Build the short-sequence talker softmax graph, bounded input, and
/// compute the output size via IBP. Returns everything the root
/// verifier smoke tests need for both the IBP and CROWN talker lanes.
pub(crate) fn talker_softmax_verifier_setup() -> (GraphNetwork, String, BoundedTensor, usize) {
    let (softmax_graph, softmax_name) =
        talker_attention_softmax_output_graph_for_seq_len(TALKER_ATTENTION_SHORT_SEQ_LEN)
            .unwrap_or_else(|e| panic!("short-seq talker softmax graph should build: {e}"));
    let input =
        bounded_hidden_states_input(TALKER_ATTENTION_SHORT_SEQ_LEN, TALKER_ATTENTION_EPSILON);
    let softmax_ibp = softmax_graph
        .propagate_ibp(&input)
        .unwrap_or_else(|e| panic!("talker softmax IBP (for output size) should succeed: {e}"));
    let output_size = softmax_ibp.lower().len();
    (softmax_graph, softmax_name, input, output_size)
}

/// Build the canonical real-RoPE talker softmax graph, bounded input, and
/// compute the output size via IBP for the root verifier smoke.
pub(crate) fn talker_softmax_real_rope_verifier_setup(
) -> (GraphNetwork, String, BoundedTensor, usize) {
    let (softmax_graph, softmax_name) = talker_attention_softmax_output_graph_real_rope();
    let input = bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, TALKER_ATTENTION_EPSILON);
    let softmax_ibp = softmax_graph.propagate_ibp(&input).unwrap_or_else(|e| {
        panic!("real-RoPE talker softmax IBP (for output size) should succeed: {e}")
    });
    let output_size = softmax_ibp.lower().len();
    (softmax_graph, softmax_name, input, output_size)
}

/// Build the talker attention softmax graph through the GraphModel
/// round-trip path (ONNX → GraphModel → GraphNetwork), with bounded
/// input and output size computed via IBP.
pub(crate) fn talker_graph_model_round_trip_verifier_setup(
) -> (GraphNetwork, String, BoundedTensor, usize) {
    let graph_model =
        load_talker_attention_with_fixed_aux_for_seq_len(TALKER_ATTENTION_SHORT_SEQ_LEN)
            .to_graph_model();
    let graph = configure_sound_softmax_modes(
        graph_model
            .build_graph_network(crate::GraphNetworkOptions::default())
            .expect("talker GraphModel round-trip build should succeed"),
    );
    let softmax_name = common::node_names_by_layer_types(&graph, &["Softmax", "CausalSoftmax"])
        .into_iter()
        .next()
        .expect("talker GraphModel round-trip should expose a softmax node");
    let mut softmax_graph = graph;
    softmax_graph.set_output(softmax_name.clone());

    let input =
        bounded_hidden_states_input(TALKER_ATTENTION_SHORT_SEQ_LEN, TALKER_ATTENTION_EPSILON);
    let softmax_ibp = softmax_graph.propagate_ibp(&input).unwrap_or_else(|e| {
        panic!("talker GraphModel round-trip softmax IBP (for output size) should succeed: {e}")
    });
    let output_size = softmax_ibp.lower().len();
    (softmax_graph, softmax_name, input, output_size)
}

/// Assert a softmax verifier result: bounds in `[0, 1]` when Verified,
/// finite+ordered when Unknown, hard failure on Timeout/Violated.
pub(crate) fn assert_softmax_verifier_result(
    result: &VerificationResult,
    output_size: usize,
    softmax_name: &str,
    label_prefix: &str,
) {
    match result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_all_bounds_in_unit_interval(output_bounds, output_size, label_prefix);
            eprintln!(
                "{label_prefix} ({softmax_name}): verified, \
                 {output_size} output bounds all in [0, 1]"
            );
        }
        VerificationResult::Unknown { reason, bounds, .. } => {
            let label = format!("{label_prefix} ({softmax_name})");
            common::assert_unknown_verifier_bounds_sound(bounds, output_size, &label);
            eprintln!(
                "{label}: unknown ({reason:?}), {} bounds finite+ordered, \
                 sample={:?}",
                bounds.len(),
                bounds.iter().take(4).collect::<Vec<_>>()
            );
        }
        VerificationResult::Timeout { .. } => {
            panic!("{label_prefix} ({softmax_name}) timed out");
        }
        VerificationResult::Violated {
            counterexample,
            output,
            ..
        } => {
            panic!(
                "{label_prefix} ({softmax_name}) violated: \
                 softmax outputs should be in [0,1]; counterexample={:?}, output={:?}",
                &counterexample[..counterexample.len().min(8)],
                &output[..output.len().min(8)]
            );
        }
    }
}

fn assert_all_bounds_in_unit_interval(output_bounds: &[Bound], expected_size: usize, label: &str) {
    assert_eq!(
        output_bounds.len(),
        expected_size,
        "{label}: output count should match ({expected_size}), got {}",
        output_bounds.len()
    );
    for (idx, bound) in output_bounds.iter().enumerate() {
        assert!(
            bound.lower() >= -1e-6,
            "{label}: output[{idx}] lower should be >= 0, got {}",
            bound.lower()
        );
        assert!(
            bound.upper() <= 1.0 + 1e-6,
            "{label}: output[{idx}] upper should be <= 1, got {}",
            bound.upper()
        );
    }
}
