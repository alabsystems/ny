// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ECAPA-TDNN compositional verification helpers for #3499.
//!
//! This module implements the structural MFA boundary discovery and single-input
//! subgraph extraction described in
//! `designs/2026-03-12-issue-3499-ecapa-composition-packet-a-execution.md`.
//!
//! The ECAPA architecture saves three SE-Res2Net block outputs (x2, x3, x4)
//! and concatenates them at a multi-layer feature aggregation (MFA) node.
//! The compositional cut discovers that node structurally, slices the graph
//! into stage-local and suffix-local single-input subgraphs, and verifies
//! that the suffix reproduces the full encoder output when fed exact MFA bounds.

// Explicit owner imports from speaker_encoder support leaves (#3837 Packet C step 2)
use super::cosine_head::{
    build_component_node_bounds, build_speaker_cosine_component_graphs, scalar_width,
    speaker_cosine_distance_upper,
};
use super::shared::{
    assert_bounded_tensors_close, assert_crown_tighter_than_ibp, avoice_speaker_encoder,
    avoice_speaker_encoder_graph, bounded_speaker_encoder_cosine_input,
    bounded_speaker_encoder_input, SPEAKER_COMPONENT_CROWN_IBP_DEADLINE_SECS,
    SPEAKER_COMPONENT_SPEC_DEADLINE_SECS, SPEAKER_ENCODER_EPSILON, SPEAKER_ENCODER_SEQUENCE_LEN,
};
use super::*;
use std::collections::{HashMap, HashSet};

mod boundary;
mod packet_bc;
mod packet_linear;
mod subgraph;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "external-avoice")]
fn test_ecapa_composition_boundary_discovers_unique_mfa_chain_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let graph = avoice_speaker_encoder_graph();
    let boundary = boundary::discover_ecapa_composition_boundary(graph)
        .expect("MFA boundary discovery should succeed on the real ECAPA graph");

    // Exactly three block outputs are discovered.
    assert_eq!(boundary.block_outputs.len(), 3);

    // The three outputs are distinct.
    let unique: HashSet<&str> = boundary.block_outputs.iter().map(|s| s.as_str()).collect();
    assert_eq!(
        unique.len(),
        3,
        "block outputs should be distinct: {:?}",
        boundary.block_outputs
    );

    // Their topological indices are strictly increasing.
    let topo_order = graph.topological_sort().unwrap();
    let topo_index: HashMap<&str, usize> = topo_order
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.as_str(), idx))
        .collect();
    let indices: Vec<usize> = boundary
        .block_outputs
        .iter()
        .map(|name| *topo_index.get(name.as_str()).unwrap())
        .collect();
    assert!(
        indices[0] < indices[1] && indices[1] < indices[2],
        "block outputs should be in strictly increasing topological order: {:?} -> {:?}",
        boundary.block_outputs,
        indices
    );

    // The MFA concat node exists and is a Concat layer.
    let concat_node = graph
        .node(&boundary.mfa_concat)
        .expect("MFA concat node should exist");
    assert!(
        matches!(concat_node.layer(), Layer::Concat(_)),
        "MFA concat node should be a Concat layer, got {:?}",
        concat_node.layer().layer_type()
    );

    eprintln!(
        "ECAPA MFA boundary: block_outputs={:?}, mfa_concat={}, concat_axis={}",
        boundary.block_outputs, boundary.mfa_concat, boundary.concat_axis
    );
}

#[test]
#[cfg(feature = "external-avoice")]
fn test_ecapa_composition_slices_are_single_input_graphs_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let graph = avoice_speaker_encoder_graph();
    let boundary = boundary::discover_ecapa_composition_boundary(graph)
        .expect("MFA boundary discovery should succeed");

    // Stage A: _input -> x2
    let stage_a = subgraph::extract_single_input_subgraph(
        graph,
        ny_propagate::NETWORK_INPUT,
        &boundary.block_outputs[0],
    )
    .expect("stage A extraction should succeed");

    // Stage B: x2 -> x3
    let stage_b = subgraph::extract_single_input_subgraph(
        graph,
        &boundary.block_outputs[0],
        &boundary.block_outputs[1],
    )
    .expect("stage B extraction should succeed");

    // Stage C: x3 -> x4
    let stage_c = subgraph::extract_single_input_subgraph(
        graph,
        &boundary.block_outputs[1],
        &boundary.block_outputs[2],
    )
    .expect("stage C extraction should succeed");

    // Suffix: mfa_concat -> output
    let suffix =
        subgraph::extract_single_input_subgraph(graph, &boundary.mfa_concat, graph.output_name())
            .expect("suffix extraction should succeed");

    // Each extracted graph passes topological_sort().
    for (label, sub) in [
        ("stage_a", &stage_a),
        ("stage_b", &stage_b),
        ("stage_c", &stage_c),
        ("suffix", &suffix),
    ] {
        let topo = sub.topological_sort();
        assert!(
            topo.is_ok(),
            "{label}: topological sort should succeed, got {:?}",
            topo.err()
        );

        // Every retained node's inputs are either NETWORK_INPUT or within the subgraph.
        let node_set: HashSet<&str> = sub.node_names().iter().map(|s| s.as_str()).collect();
        for name in sub.node_names() {
            let node = sub.node(name).unwrap();
            for inp in node.inputs() {
                assert!(
                    inp == ny_propagate::NETWORK_INPUT || node_set.contains(inp.as_str()),
                    "{label}: node '{}' references '{}' which is not in the subgraph",
                    name,
                    inp
                );
            }
        }

        eprintln!(
            "{label}: {} nodes, output='{}'",
            sub.num_nodes(),
            sub.output_name()
        );
    }

    // The suffix graph output name matches the original encoder output name.
    assert_eq!(
        suffix.output_name(),
        graph.output_name(),
        "suffix output should match the original encoder output"
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_ecapa_suffix_ibp_matches_full_encoder_from_exact_mfa_bounds_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let model = avoice_speaker_encoder();
    let graph = avoice_speaker_encoder_graph();
    let input =
        bounded_speaker_encoder_input(model, SPEAKER_ENCODER_SEQUENCE_LEN, SPEAKER_ENCODER_EPSILON);

    // Full encoder IBP node bounds.
    let node_bounds = graph
        .collect_node_bounds(&input)
        .expect("full encoder IBP node-bound collection should succeed");

    let boundary = boundary::discover_ecapa_composition_boundary(graph)
        .expect("MFA boundary discovery should succeed");

    // Concatenate x2/x3/x4 from the node-bound map at the discovered axis.
    let concat_bounds = subgraph::concat_mfa_block_bounds(&boundary, &node_bounds);
    eprintln!("MFA concat shape={:?}", concat_bounds.lower().shape());

    // Extract the suffix graph: mfa_concat -> encoder output.
    let suffix =
        subgraph::extract_single_input_subgraph(graph, &boundary.mfa_concat, graph.output_name())
            .expect("suffix extraction should succeed");

    // Suffix IBP from exact MFA bounds should reproduce the full encoder output.
    let suffix_output = suffix
        .propagate_ibp(&concat_bounds)
        .unwrap_or_else(|e| panic!("suffix IBP should succeed: {e}"));

    let full_output = node_bounds
        .get(graph.output_name())
        .unwrap_or_else(|| panic!("full encoder output missing from node bounds"));

    assert_bounded_tensors_close(
        &suffix_output,
        full_output,
        "suffix IBP vs full encoder output",
    );
    eprintln!(
        "suffix IBP parity PASSED: shape={:?}",
        suffix_output.lower().shape()
    );
}
