// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_graph_ibp_avoice_talker_attention_fixed_aux_3497() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let model = load_talker_attention_with_fixed_aux();
    let graph = model
        .to_graph_network()
        .expect("talker attention with fixed aux should convert to GraphNetwork");

    let input = bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, TALKER_ATTENTION_EPSILON);

    let output = graph
        .propagate_ibp(&input)
        .expect("talker attention graph IBP should succeed with fixed aux inputs");

    assert_eq!(
        output.lower().shape().last().copied(),
        Some(TALKER_ATTENTION_HIDDEN_DIM),
        "talker attention IBP output should end in {TALKER_ATTENTION_HIDDEN_DIM}, got {:?}",
        output.lower().shape()
    );
    common::assert_finite_and_ordered(&output, "talker attention graph IBP output");
}

#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_collect_node_bounds_avoice_talker_attention_softmax_3497() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let model = load_talker_attention_with_fixed_aux();
    let graph = configure_sound_softmax_modes(
        model
            .to_graph_network()
            .expect("talker attention with fixed aux should convert to GraphNetwork"),
    );

    let input = bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, TALKER_ATTENTION_EPSILON);

    let node_bounds = graph
        .collect_node_bounds(&input)
        .expect("talker attention node-bound collection should succeed");

    let softmax_nodes = common::node_names_by_layer_types(&graph, &["Softmax", "CausalSoftmax"]);
    common::assert_node_bounds_finite_and_ordered(
        &node_bounds,
        &softmax_nodes,
        "talker attention softmax",
    );

    let softmax_bounds = &node_bounds[&softmax_nodes[0]];
    let _ = centroid_bounds_from_softmax(softmax_bounds, "talker attention");
}

#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_graph_crown_avoice_talker_attention_softmax_centroid_3497() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let (graph, softmax_name) = talker_attention_softmax_output_graph();
    let input = bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, TALKER_ATTENTION_EPSILON);

    let ibp_softmax = graph
        .propagate_ibp(&input)
        .expect("talker attention softmax-output IBP should succeed");
    let crown = graph
        .propagate_crown_with_provenance(&input)
        .expect("talker attention softmax-output CROWN should succeed");

    assert_eq!(
        crown.provenance,
        ny_propagate::types::BoundsProvenance::Crown,
        "talker attention softmax output should use backward CROWN for node {softmax_name}, got {:?}",
        crown.provenance
    );

    common::assert_finite_and_ordered(&crown.bounds, "talker attention softmax CROWN");

    let (ibp_centroid_lower, ibp_centroid_upper, _) =
        centroid_bounds_from_softmax(&ibp_softmax, "talker attention softmax IBP");
    let (crown_centroid_lower, crown_centroid_upper, _) =
        centroid_bounds_from_softmax(&crown.bounds, "talker attention softmax CROWN");

    assert_centroid_bounds_no_looser(
        "talker attention softmax centroid CROWN",
        &crown_centroid_lower,
        &crown_centroid_upper,
        &ibp_centroid_lower,
        &ibp_centroid_upper,
        1e-4,
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_short_seq_talker_attention_fixed_aux_smoke_3589() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let seq_len = TALKER_ATTENTION_SHORT_SEQ_LEN;
    let graph = talker_attention_graph_with_fixed_aux_for_seq_len(seq_len).unwrap_or_else(|e| {
        panic!("short-seq talker-attention graph setup failed at seq_len={seq_len}: {e}")
    });
    let softmax_name = first_talker_attention_softmax_node(&graph);
    let mut softmax_graph = graph;
    softmax_graph.set_output(softmax_name.clone());

    let input = bounded_hidden_states_input(seq_len, TALKER_ATTENTION_EPSILON);
    let ibp = softmax_graph.propagate_ibp(&input).unwrap_or_else(|e| {
        panic!(
            "short-seq talker-attention IBP failed at seq_len={seq_len} after graph conversion for node {softmax_name}: {e}"
        )
    });
    common::assert_finite_and_ordered(&ibp, "short-seq talker attention softmax IBP");

    let crown = softmax_graph
        .propagate_crown_with_provenance(&input)
        .unwrap_or_else(|e| {
            panic!(
                "short-seq talker-attention CROWN failed at seq_len={seq_len} after IBP for node {softmax_name}: {e}"
            )
        });
    assert_eq!(
        crown.provenance,
        ny_propagate::types::BoundsProvenance::Crown,
        "short-seq talker attention should use backward CROWN for node {softmax_name}, got {:?}",
        crown.provenance
    );
    common::assert_finite_and_ordered(&crown.bounds, "short-seq talker attention softmax CROWN");

    let _ = centroid_bounds_from_softmax(&ibp, "short-seq talker attention softmax IBP");
    let _ = centroid_bounds_from_softmax(&crown.bounds, "short-seq talker attention softmax CROWN");
}
