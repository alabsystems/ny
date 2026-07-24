// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) const SPEAKER_ENCODER_FILE: &str = "speaker_encoder.onnx";
// The upstream ECAPA-TDNN uses fixed TDNN pads [2, 2, 3, 4, 0] derived from
// kernel/dilation. Reflect padding requires pad < T, so T=5 is the smallest
// valid real-weight sequence while still minimizing the expensive IBP/node
// bound passes on the real ECAPA graph.
pub(crate) const SPEAKER_ENCODER_SEQUENCE_LEN: usize = 5;
pub(super) const SPEAKER_ENCODER_EPSILON: f32 = 1e-3;
pub(super) const SPEAKER_DISTANCE_ACCEPTANCE_UPPER: f32 = 0.1;
pub(super) const VACUOUS_COSINE_DISTANCE_UPPER: f32 = 2.0;
pub(super) const SPEAKER_COMPONENT_CROWN_IBP_DEADLINE_SECS: u64 = 60;
pub(super) const SPEAKER_COMPONENT_SPEC_DEADLINE_SECS: u64 = 30;

pub(super) fn avoice_speaker_encoder() -> &'static OnnxModel {
    static MODEL: OnceLock<OnnxModel> = OnceLock::new();
    MODEL.get_or_init(|| {
        let path = require_test_model_with_hint(SPEAKER_ENCODER_FILE, AVOICE_TEST_MODEL_HINT);
        load_onnx(&path).expect("Failed to load avoice speaker_encoder.onnx")
    })
}

pub(crate) fn avoice_speaker_encoder_graph() -> &'static GraphNetwork {
    static GRAPH: OnceLock<GraphNetwork> = OnceLock::new();
    GRAPH.get_or_init(|| {
        avoice_speaker_encoder()
            .to_graph_network()
            .expect("Failed to convert avoice speaker encoder to GraphNetwork")
    })
}

pub(super) fn speaker_encoder_input_shape(model: &OnnxModel, dynamic_t: usize) -> Vec<usize> {
    let input_spec = model
        .network
        .inputs
        .first()
        .expect("avoice speaker encoder should expose one input");
    assert_eq!(
        input_spec.shape.len(),
        3,
        "expected avoice speaker encoder input rank [B, T, 128], got {:?}",
        input_spec.shape
    );
    assert_eq!(
        input_spec.shape.last(),
        Some(&128),
        "speaker encoder mel dimension should end in 128, got {:?}",
        input_spec.shape
    );

    let shape: Vec<usize> = input_spec.shape[1..]
        .iter()
        .enumerate()
        .map(|(idx, &dim)| {
            if dim > 0 {
                dim as usize
            } else if idx == 0 {
                dynamic_t
            } else {
                1
            }
        })
        .collect();

    assert_eq!(
        shape.last(),
        Some(&128),
        "unbatched propagation input should preserve mel width 128, got {:?}",
        shape
    );
    shape
}

pub(super) fn stable_speaker_cosine_center(shape: &[usize]) -> ArrayD<f32> {
    let len = shape.iter().product();
    ArrayD::from_shape_vec(
        IxDyn(shape),
        (0..len)
            .map(|idx| if idx % 2 == 0 { 0.05 } else { 0.06 })
            .collect(),
    )
    .expect("valid stable speaker cosine center shape")
}

pub(super) fn bounded_speaker_encoder_input(
    model: &OnnxModel,
    dynamic_t: usize,
    epsilon: f32,
) -> BoundedTensor {
    let shape = speaker_encoder_input_shape(model, dynamic_t);
    let center = ArrayD::zeros(IxDyn(&shape));
    BoundedTensor::from_epsilon(center, epsilon)
        .expect("avoice bounded input helper should build a valid epsilon box")
}

pub(super) fn bounded_speaker_encoder_cosine_input(
    model: &OnnxModel,
    dynamic_t: usize,
    epsilon: f32,
) -> BoundedTensor {
    let shape = speaker_encoder_input_shape(model, dynamic_t);
    let center = stable_speaker_cosine_center(&shape);
    BoundedTensor::from_epsilon(center, epsilon)
        .expect("avoice cosine bounded input helper should build a valid epsilon box")
}

pub(super) fn assert_bounded_tensors_close(
    actual: &BoundedTensor,
    expected: &BoundedTensor,
    label: &str,
) {
    assert_eq!(
        actual.lower().shape(),
        expected.lower().shape(),
        "{label}: lower shapes differ"
    );
    assert_eq!(
        actual.upper().shape(),
        expected.upper().shape(),
        "{label}: upper shapes differ"
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
        let scale = actual_lower.abs().max(expected_lower.abs()).max(1.0);
        let tol = 1e-6 * scale;
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
        let scale = actual_upper.abs().max(expected_upper.abs()).max(1.0);
        let tol = 1e-6 * scale;
        assert!(
            (actual_upper - expected_upper).abs() <= tol,
            "{label}: upper[{idx}] mismatch: actual={actual_upper}, expected={expected_upper}, tol={tol}"
        );
    }
}

pub(super) fn assert_node_bounds_maps_match(
    actual: &HashMap<String, BoundedTensor>,
    expected: &HashMap<String, BoundedTensor>,
    graph: &GraphNetwork,
    label: &str,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: node-bound map sizes differ"
    );

    for node_name in graph.node_names() {
        let actual_bounds = actual
            .get(node_name)
            .unwrap_or_else(|| panic!("{label}: missing actual node bounds for '{node_name}'"));
        let expected_bounds = expected
            .get(node_name)
            .unwrap_or_else(|| panic!("{label}: missing expected node bounds for '{node_name}'"));
        assert_bounded_tensors_close(
            actual_bounds,
            expected_bounds,
            &format!("{label} ({node_name})"),
        );
    }
}

pub(super) fn assert_crown_tighter_than_ibp(
    crown_output: &BoundedTensor,
    ibp_output: &BoundedTensor,
    label: &str,
) {
    assert_eq!(
        crown_output.lower().shape(),
        ibp_output.lower().shape(),
        "{label}: CROWN/IBP lower shapes differ"
    );
    assert_eq!(
        crown_output.upper().shape(),
        ibp_output.upper().shape(),
        "{label}: CROWN/IBP upper shapes differ"
    );
    for (idx, (((&crown_lower, &crown_upper), &ibp_lower), &ibp_upper)) in crown_output
        .lower()
        .iter()
        .zip(crown_output.upper().iter())
        .zip(ibp_output.lower().iter())
        .zip(ibp_output.upper().iter())
        .enumerate()
    {
        let scale = crown_lower
            .abs()
            .max(crown_upper.abs())
            .max(ibp_lower.abs())
            .max(ibp_upper.abs())
            .max(1.0);
        let tol = 1e-6 * scale;
        assert!(
            crown_lower >= ibp_lower - tol,
            "{label} lower[{idx}] is looser than IBP: crown={crown_lower}, ibp={ibp_lower}"
        );
        assert!(
            crown_upper <= ibp_upper + tol,
            "{label} upper[{idx}] is looser than IBP: crown={crown_upper}, ibp={ibp_upper}"
        );
    }
}

#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
fn test_avoice_speaker_encoder_graph_model_round_trip_matches_direct_bounds_3923() {
    crate::test_fixtures::require_test_model_or_skip!("speaker_encoder.onnx");
    let _guard = common::lock_heavy_avoice_round_trip();
    let path = require_test_model_with_hint(SPEAKER_ENCODER_FILE, AVOICE_TEST_MODEL_HINT);
    let direct_model = avoice_speaker_encoder();
    let direct_graph = avoice_speaker_encoder_graph();
    let roundtrip_graph = load_onnx(&path)
        .expect("speaker encoder round-trip load should succeed")
        .to_graph_model()
        .build_graph_network(crate::GraphNetworkOptions::default())
        .expect("speaker encoder GraphModel round-trip build should succeed");

    assert_eq!(
        roundtrip_graph.output_name(),
        direct_graph.output_name(),
        "speaker encoder round-trip should keep the same output node"
    );
    assert_eq!(
        roundtrip_graph.node_names().len(),
        direct_graph.node_names().len(),
        "speaker encoder round-trip should keep the same node count"
    );

    let input = bounded_speaker_encoder_input(
        direct_model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        SPEAKER_ENCODER_EPSILON,
    );
    let direct_output = direct_graph
        .propagate_ibp(&input)
        .expect("speaker encoder direct IBP should succeed");
    let roundtrip_output = roundtrip_graph
        .propagate_ibp(&input)
        .expect("speaker encoder round-trip IBP should succeed");
    assert_bounded_tensors_close(
        &roundtrip_output,
        &direct_output,
        "speaker encoder GraphModel round-trip output",
    );

    let direct_node_bounds = direct_graph
        .collect_node_bounds(&input)
        .expect("speaker encoder direct node-bound collection should succeed");
    let roundtrip_node_bounds = roundtrip_graph
        .collect_node_bounds(&input)
        .expect("speaker encoder round-trip node-bound collection should succeed");
    assert_node_bounds_maps_match(
        &roundtrip_node_bounds,
        &direct_node_bounds,
        direct_graph,
        "speaker encoder GraphModel round-trip node bounds",
    );
}
