// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::common::{
    assert_finite_and_ordered, lock_heavy_avoice_round_trip, node_names_by_layer_types,
};
use super::graph_support::instance_norm_node_count;
use super::model::{load_kokoro_vocoder_with_fixed_aux, KOKORO_VOCODER_STRUCTURAL_T};
use super::verifier_smoke::{
    kokoro_graph_model_round_trip_prefix_energy_verifier_setup, kokoro_prefix_energy_verifier_setup,
};
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;
use std::sync::OnceLock;

fn kokoro_fixed_aux_baseline() -> &'static crate::OnnxModel {
    static MODEL: OnceLock<crate::OnnxModel> = OnceLock::new();
    MODEL.get_or_init(|| load_kokoro_vocoder_with_fixed_aux(KOKORO_VOCODER_STRUCTURAL_T))
}

fn kokoro_fixed_aux_graph() -> &'static GraphNetwork {
    static GRAPH: OnceLock<GraphNetwork> = OnceLock::new();
    GRAPH.get_or_init(|| {
        kokoro_fixed_aux_baseline()
            .to_graph_network()
            .expect("kokoro fixed-aux direct graph build should succeed")
    })
}

/// Assert frozen-aux metadata parity between the `GraphModel` and its
/// fixed-aux `OnnxModel` baseline (pre-build contract).
fn assert_kokoro_frozen_aux_metadata(
    graph_model: &ny_build::GraphModel,
    baseline: &crate::OnnxModel,
) {
    assert_eq!(
        graph_model
            .network
            .inputs
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<Vec<_>>(),
        vec!["features"],
        "kokoro GraphModel round-trip should keep features as the only live input"
    );
    assert_eq!(
        graph_model.network.inputs[0].shape, baseline.network.inputs[0].shape,
        "kokoro GraphModel round-trip should preserve the features input shape"
    );
    assert_eq!(
        &graph_model.constant_tensors,
        baseline.constant_tensors(),
        "kokoro GraphModel round-trip should preserve the frozen-aux constant tensor set"
    );
    for tensor_name in ["style", "har"] {
        let actual = graph_model.weights.get(tensor_name).unwrap_or_else(|| {
            panic!("kokoro GraphModel round-trip missing weight '{tensor_name}'")
        });
        let expected = baseline
            .weights
            .get(tensor_name)
            .unwrap_or_else(|| panic!("kokoro fixed-aux baseline missing weight '{tensor_name}'"));
        assert_eq!(
            actual.shape(),
            expected.shape(),
            "kokoro GraphModel round-trip weight '{tensor_name}' shape mismatch"
        );
        for (idx, (&actual_value, &expected_value)) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            assert_eq!(
                actual_value.to_bits(),
                expected_value.to_bits(),
                "kokoro GraphModel round-trip weight '{tensor_name}' mismatch at index {idx}: actual={actual_value}, expected={expected_value}"
            );
        }
    }
    for tensor_name in ["features", "style", "har"] {
        assert_eq!(
            graph_model.tensor_shapes.get(tensor_name),
            baseline.tensor_shapes().get(tensor_name),
            "kokoro GraphModel round-trip should preserve tensor_shapes['{tensor_name}']"
        );
    }
    assert_eq!(
        &graph_model.tensor_producer,
        baseline.tensor_producer(),
        "kokoro GraphModel round-trip should preserve the full tensor_producer map"
    );
}

/// Assert structural parity between direct and round-trip built graphs.
fn assert_kokoro_structural_parity(direct: &GraphNetwork, roundtrip: &GraphNetwork) {
    assert_eq!(
        roundtrip.output_name(),
        direct.output_name(),
        "kokoro GraphModel round-trip should keep the same output node"
    );
    assert_eq!(
        roundtrip.node_names().len(),
        direct.node_names().len(),
        "kokoro GraphModel round-trip should keep the same node count"
    );
    assert_eq!(
        roundtrip
            .topological_sort()
            .expect("kokoro round-trip topo sort should succeed"),
        direct
            .topological_sort()
            .expect("kokoro direct topo sort should succeed"),
        "kokoro GraphModel round-trip should keep the same topological node-name list"
    );
    assert_eq!(
        instance_norm_node_count(roundtrip),
        instance_norm_node_count(direct),
        "kokoro GraphModel round-trip should keep the same fused InstanceNorm1d node count"
    );
    assert!(
        instance_norm_node_count(roundtrip) > 0,
        "kokoro GraphModel round-trip should retain at least one fused InstanceNorm1d node"
    );
    assert_eq!(
        node_names_by_layer_types(roundtrip, &["Conv1d", "Conv2d"]).len(),
        node_names_by_layer_types(direct, &["Conv1d", "Conv2d"]).len(),
        "kokoro GraphModel round-trip should keep the same convolution stage count"
    );
    assert_eq!(
        node_names_by_layer_types(roundtrip, &["ConvTranspose1d", "ConvTranspose2d"]).len(),
        node_names_by_layer_types(direct, &["ConvTranspose1d", "ConvTranspose2d"]).len(),
        "kokoro GraphModel round-trip should keep the same transposed-convolution stage count"
    );
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

#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
fn test_avoice_kokoro_vocoder_graph_model_round_trip_matches_direct_inventory_3923() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    let _guard = lock_heavy_avoice_round_trip();
    let baseline = kokoro_fixed_aux_baseline();
    let graph_model =
        load_kokoro_vocoder_with_fixed_aux(KOKORO_VOCODER_STRUCTURAL_T).to_graph_model();

    assert_kokoro_frozen_aux_metadata(&graph_model, baseline);

    let direct_graph = kokoro_fixed_aux_graph();
    let roundtrip_graph = graph_model
        .build_graph_network(crate::GraphNetworkOptions::default())
        .expect("kokoro GraphModel round-trip build should succeed");

    assert_kokoro_structural_parity(direct_graph, &roundtrip_graph);
}

#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
fn test_avoice_kokoro_prefix_energy_graph_model_round_trip_matches_direct_ibp_4100() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    let _guard = lock_heavy_avoice_round_trip();
    let (direct_graph, direct_input) = kokoro_prefix_energy_verifier_setup();
    let (round_trip_graph, round_trip_input) =
        kokoro_graph_model_round_trip_prefix_energy_verifier_setup();

    assert_eq!(
        round_trip_input.lower().shape(),
        direct_input.lower().shape(),
        "kokoro GraphModel round-trip prefix energy input lower shape mismatch"
    );
    assert_eq!(
        round_trip_input.upper().shape(),
        direct_input.upper().shape(),
        "kokoro GraphModel round-trip prefix energy input upper shape mismatch"
    );
    for (idx, (&actual, &expected)) in round_trip_input
        .lower()
        .iter()
        .zip(direct_input.lower().iter())
        .enumerate()
    {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "kokoro GraphModel round-trip prefix energy lower input mismatch at index {idx}: actual={actual}, expected={expected}"
        );
    }
    for (idx, (&actual, &expected)) in round_trip_input
        .upper()
        .iter()
        .zip(direct_input.upper().iter())
        .enumerate()
    {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "kokoro GraphModel round-trip prefix energy upper input mismatch at index {idx}: actual={actual}, expected={expected}"
        );
    }

    let direct_ibp = direct_graph
        .propagate_ibp(&direct_input)
        .expect("kokoro direct prefix energy IBP should succeed");
    let round_trip_ibp = round_trip_graph
        .propagate_ibp(&round_trip_input)
        .expect("kokoro GraphModel round-trip prefix energy IBP should succeed");

    assert_finite_and_ordered(&direct_ibp, "kokoro direct prefix energy IBP");
    assert_finite_and_ordered(
        &round_trip_ibp,
        "kokoro GraphModel round-trip prefix energy IBP",
    );
    assert_eq!(
        direct_ibp.lower().len(),
        1,
        "kokoro direct prefix energy IBP should stay scalar"
    );
    assert_eq!(
        round_trip_ibp.lower().len(),
        1,
        "kokoro GraphModel round-trip prefix energy IBP should stay scalar"
    );
    assert_bounds_match(
        &round_trip_ibp,
        &direct_ibp,
        "kokoro GraphModel round-trip prefix energy IBP",
    );
}
