// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn assert_tensor_matches(actual: &ArrayD<f32>, expected: &ArrayD<f32>, label: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{label}: shape mismatch");
    for (idx, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual_value.to_bits(),
            expected_value.to_bits(),
            "{label}: value mismatch at index {idx}: actual={actual_value}, expected={expected_value}"
        );
    }
}

fn assert_bounds_match(actual: &BoundedTensor, expected: &BoundedTensor, label: &str) {
    assert_eq!(
        actual.lower().shape(),
        expected.lower().shape(),
        "{label}: lower shape mismatch"
    );
    assert_eq!(
        actual.upper().shape(),
        expected.upper().shape(),
        "{label}: upper shape mismatch"
    );
    for (idx, (&actual_lower, &expected_lower)) in actual
        .lower()
        .iter()
        .zip(expected.lower().iter())
        .enumerate()
    {
        if actual_lower == expected_lower {
            continue;
        }
        let tol = 1e-6 * actual_lower.abs().max(expected_lower.abs()).max(1.0);
        assert!(
            (actual_lower - expected_lower).abs() <= tol,
            "{label}: lower[{idx}] mismatch: actual={actual_lower}, expected={expected_lower}, tol={tol}"
        );
    }
    for (idx, (&actual_upper, &expected_upper)) in actual
        .upper()
        .iter()
        .zip(expected.upper().iter())
        .enumerate()
    {
        if actual_upper == expected_upper {
            continue;
        }
        let tol = 1e-6 * actual_upper.abs().max(expected_upper.abs()).max(1.0);
        assert!(
            (actual_upper - expected_upper).abs() <= tol,
            "{label}: upper[{idx}] mismatch: actual={actual_upper}, expected={expected_upper}, tol={tol}"
        );
    }
}

/// Assert frozen-aux metadata parity between the `GraphModel` and its
/// fixed-aux `OnnxModel` baseline (pre-build contract).
fn assert_talker_frozen_aux_metadata(graph_model: &ny_build::GraphModel, baseline: &OnnxModel) {
    assert_eq!(
        graph_model
            .network
            .inputs
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<Vec<_>>(),
        vec!["hidden_states"],
        "talker GraphModel round-trip should keep hidden_states as the only live input"
    );
    assert_eq!(
        graph_model.network.inputs[0].shape, baseline.network.inputs[0].shape,
        "talker GraphModel round-trip should preserve the hidden_states input shape"
    );
    assert_eq!(
        graph_model.network.inputs[0].dtype, baseline.network.inputs[0].dtype,
        "talker GraphModel round-trip should preserve the hidden_states input dtype"
    );
    assert_eq!(
        &graph_model.constant_tensors,
        baseline.constant_tensors(),
        "talker GraphModel round-trip should preserve the frozen-aux constant tensor set"
    );
    for tensor_name in ["cos", "sin", "mask"] {
        assert_tensor_matches(
            graph_model.weights.get(tensor_name).unwrap_or_else(|| {
                panic!("talker GraphModel round-trip missing weight '{tensor_name}'")
            }),
            baseline.weights.get(tensor_name).unwrap_or_else(|| {
                panic!("talker fixed-aux baseline missing weight '{tensor_name}'")
            }),
            &format!("talker GraphModel round-trip weight '{tensor_name}'"),
        );
    }
    for tensor_name in ["hidden_states", "cos", "sin", "mask"] {
        assert_eq!(
            graph_model.tensor_shapes.get(tensor_name),
            baseline.tensor_shapes().get(tensor_name),
            "talker GraphModel round-trip should preserve tensor_shapes['{tensor_name}']"
        );
    }
    assert_eq!(
        &graph_model.tensor_producer,
        baseline.tensor_producer(),
        "talker GraphModel round-trip should preserve the full tensor_producer map"
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_avoice_talker_attention_graph_model_round_trip_matches_direct_softmax_ibp_3923() {
    crate::test_fixtures::assert_test_model_available!("talker_attention_layer0.onnx");
    let seq_len = TALKER_ATTENTION_SHORT_SEQ_LEN;
    let baseline = load_talker_attention_with_fixed_aux_for_seq_len(seq_len);
    let graph_model = load_talker_attention_with_fixed_aux_for_seq_len(seq_len).to_graph_model();

    assert_talker_frozen_aux_metadata(&graph_model, &baseline);

    let direct_graph = configure_sound_softmax_modes(
        baseline
            .to_graph_network()
            .expect("talker fixed-aux direct graph build should succeed"),
    );
    let roundtrip_graph = configure_sound_softmax_modes(
        graph_model
            .build_graph_network(crate::GraphNetworkOptions::default())
            .expect("talker GraphModel round-trip build should succeed"),
    );
    assert_eq!(
        roundtrip_graph.node_names().len(),
        direct_graph.node_names().len(),
        "talker GraphModel round-trip should keep the same node count"
    );

    let direct_softmax_nodes =
        common::node_names_by_layer_types(&direct_graph, &["Softmax", "CausalSoftmax"]);
    let roundtrip_softmax_nodes =
        common::node_names_by_layer_types(&roundtrip_graph, &["Softmax", "CausalSoftmax"]);
    assert!(
        !direct_softmax_nodes.is_empty(),
        "talker GraphModel round-trip: expected at least one Softmax/CausalSoftmax node"
    );
    assert_eq!(
        roundtrip_softmax_nodes,
        direct_softmax_nodes,
        "talker GraphModel round-trip should expose the same softmax node-name list before selecting an output"
    );

    let softmax_name = direct_softmax_nodes[0].clone();
    let mut direct_softmax_graph = direct_graph;
    direct_softmax_graph.set_output(softmax_name.clone());
    let mut roundtrip_softmax_graph = roundtrip_graph;
    roundtrip_softmax_graph.set_output(softmax_name);

    let input = bounded_hidden_states_input(seq_len, TALKER_ATTENTION_EPSILON);
    let direct_ibp = direct_softmax_graph
        .propagate_ibp(&input)
        .expect("talker direct softmax IBP should succeed");
    let roundtrip_ibp = roundtrip_softmax_graph
        .propagate_ibp(&input)
        .expect("talker GraphModel round-trip softmax IBP should succeed");
    common::assert_finite_and_ordered(&direct_ibp, "talker direct softmax IBP");
    common::assert_finite_and_ordered(&roundtrip_ibp, "talker GraphModel round-trip softmax IBP");
    assert_bounds_match(
        &roundtrip_ibp,
        &direct_ibp,
        "talker GraphModel round-trip softmax IBP",
    );
}
