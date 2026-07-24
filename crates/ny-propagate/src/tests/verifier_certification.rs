// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::composition::certificate::{BoundCertificationResult, BoundProvenance};
use crate::layers::{AddLayer, NonZeroLayer, ReshapeLayer, SqrtLayer};
use crate::{
    GraphNetwork, GraphNode, Layer, LinearLayer, MulBinaryRelaxationMode, Network,
    PropagationConfig, PropagationMethod, ReLULayer, Verifier,
};
use ndarray::{arr1, arr2};
use ny_core::{HeuristicUsed, MethodUsed, NyError, VerificationSoundnessMode};
use ny_tensor::BoundedTensor;

const SMOKE_MAX_ITERATIONS: usize = 20;
const SMOKE_BETA_ITERATIONS: usize = 5;

fn smoke_config(method: PropagationMethod) -> PropagationConfig {
    let max_iterations = match method {
        PropagationMethod::BetaCrown => SMOKE_BETA_ITERATIONS,
        _ => SMOKE_MAX_ITERATIONS,
    };
    PropagationConfig {
        method,
        max_iterations,
        tolerance: 1e-4,
        use_gpu: false,
        ..Default::default()
    }
}

fn make_binary_add_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    let weight = arr2(&[[1.0, 0.5], [-0.3, 0.7]]);
    let bias = arr1(&[0.1, -0.2]);
    graph.add_node(GraphNode::from_input(
        "left_linear",
        Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "nonzero",
        Layer::NonZero(NonZeroLayer),
        vec!["left_linear".to_string()],
    ));
    let right_weight = arr2(&[[0.5, -0.25], [-0.3, 0.7]]);
    let right_bias = arr1(&[-0.4, 0.3]);
    graph.add_node(GraphNode::from_input(
        "right_linear",
        Layer::Linear(LinearLayer::new(right_weight, Some(right_bias)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "right_reshape",
        Layer::Reshape(ReshapeLayer::new(vec![1, 2])),
        vec!["right_linear".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "sum",
        Layer::Add(AddLayer),
        "nonzero",
        "right_reshape",
    ));
    graph.set_output("sum");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_certify_network_bounds_preserves_actual_method_and_shape_3920() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32, -0.5]]), Some(arr1(&[0.25]))).unwrap(),
    ));

    let verifier = Verifier::new(smoke_config(PropagationMethod::Crown));
    let input_bounds = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.0]).into_dyn(),
        arr1(&[2.0_f32, 1.0]).into_dyn(),
    )
    .expect("valid certification input bounds");

    let certification = verifier
        .certify_network_bounds("encoder", &network, &input_bounds, None)
        .expect("bound certification should succeed");

    match certification {
        BoundCertificationResult::Certified(cert) => {
            assert_eq!(cert.model_id(), "encoder");
            assert_eq!(cert.actual_method(), &MethodUsed::Crown);
            assert_eq!(cert.provenance(), BoundProvenance::Crown);
            assert_eq!(cert.output_bounds().shape(), &[1]);

            let lower = cert.output_bounds().lower().as_slice().unwrap()[0];
            let upper = cert.output_bounds().upper().as_slice().unwrap()[0];
            assert!(
                lower <= -0.25 && 2.25 <= upper,
                "certified bounds should contain the concrete extrema, got [{lower}, {upper}]"
            );
        }
        other => panic!("expected Certified, got {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_certify_network_bounds_preserves_soundness_provenance_3920() {
    let mut network = Network::new();
    network.add_layer(Layer::Sqrt(SqrtLayer));

    let verifier = Verifier::new(smoke_config(PropagationMethod::Ibp));
    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[4.0_f32]).into_dyn())
            .expect("valid certification input bounds");

    let certification = verifier
        .certify_network_bounds("sqrt", &network, &input_bounds, None)
        .expect("bound certification should preserve soundness metadata");

    match certification {
        BoundCertificationResult::Certified(cert) => {
            assert_eq!(cert.model_id(), "sqrt");
            assert_eq!(cert.actual_method(), &MethodUsed::Ibp);
            assert_eq!(cert.provenance(), BoundProvenance::Ibp);
            assert_eq!(
                cert.soundness().mode(),
                VerificationSoundnessMode::Heuristic
            );
            assert!(
                cert.soundness()
                    .heuristics_used()
                    .iter()
                    .any(|heuristic| matches!(
                        heuristic,
                        HeuristicUsed::SqrtNegativeDomain { num_nodes: 1 }
                    )),
                "expected sqrt negative-domain heuristic, got {:?}",
                cert.soundness().heuristics_used()
            );
        }
        other => panic!("expected Certified, got {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_certify_network_bounds_timeout_preserves_metadata_3920() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));

    let verifier = Verifier::new(smoke_config(PropagationMethod::Crown));
    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid certification input bounds");

    let certification = verifier
        .certify_network_bounds("encoder", &network, &input_bounds, Some(0))
        .expect("timeout should remain structured control flow");

    match certification {
        BoundCertificationResult::Timeout {
            partial,
            actual_method,
            soundness,
        } => {
            assert!(
                partial.is_none(),
                "expected timeout without partial certificate"
            );
            assert_eq!(actual_method, MethodUsed::Crown);
            assert_eq!(soundness.mode(), VerificationSoundnessMode::Sound);
            assert!(
                soundness.heuristics_used().is_empty(),
                "unexpected timeout heuristics: {:?}",
                soundness.heuristics_used()
            );
        }
        other => panic!("expected timeout without partial certificate, got {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_certify_network_bounds_rejects_beta_crown_3920() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let verifier = Verifier::new(smoke_config(PropagationMethod::BetaCrown));
    let input_bounds = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid certification input bounds");

    let err = verifier
        .certify_network_bounds("encoder", &network, &input_bounds, None)
        .expect_err("beta-crown certification should fail closed");

    assert!(
        matches!(err, NyError::UnsupportedOp(ref message) if message.contains("BetaCrown")),
        "expected unsupported-op beta-crown error, got {err:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_certify_graph_bounds_preserves_fallback_actual_method_3920() {
    let graph = make_binary_add_graph();
    let mut crown_config = smoke_config(PropagationMethod::Crown);
    crown_config.mul_binary_relaxation = MulBinaryRelaxationMode::Middle;
    let verifier = Verifier::new(crown_config);
    let input_bounds = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("valid graph certification input bounds");

    let certification = verifier
        .certify_graph_bounds("sum", &graph, &input_bounds, None)
        .expect("graph certification should degrade to IBP instead of failing");

    match certification {
        BoundCertificationResult::Certified(cert) => {
            assert_eq!(cert.actual_method(), &MethodUsed::Ibp);
            assert_eq!(cert.provenance(), BoundProvenance::Ibp);
            assert_eq!(cert.output_bounds().shape(), &[1, 2]);
        }
        other => panic!("expected Certified, got {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_certify_graph_bounds_preserves_soundness_provenance_3920() {
    let mut network = Network::new();
    network.add_layer(Layer::Sqrt(SqrtLayer));
    let graph = GraphNetwork::from_sequential(&network).expect("sequential graph conversion");

    let verifier = Verifier::new(smoke_config(PropagationMethod::Ibp));
    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[4.0_f32]).into_dyn())
            .expect("valid graph certification input bounds");

    let certification = verifier
        .certify_graph_bounds("sqrt_graph", &graph, &input_bounds, None)
        .expect("graph certification should preserve soundness metadata");

    match certification {
        BoundCertificationResult::Certified(cert) => {
            assert_eq!(cert.model_id(), "sqrt_graph");
            assert_eq!(cert.actual_method(), &MethodUsed::Ibp);
            assert_eq!(cert.provenance(), BoundProvenance::Ibp);
            assert_eq!(
                cert.soundness().mode(),
                VerificationSoundnessMode::Heuristic
            );
            assert!(
                cert.soundness()
                    .heuristics_used()
                    .iter()
                    .any(|heuristic| matches!(
                        heuristic,
                        HeuristicUsed::SqrtNegativeDomain { num_nodes: 1 }
                    )),
                "expected sqrt negative-domain heuristic, got {:?}",
                cert.soundness().heuristics_used()
            );
        }
        other => panic!("expected Certified, got {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_certify_graph_bounds_timeout_preserves_metadata_3920() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    let graph = GraphNetwork::from_sequential(&network).expect("sequential graph conversion");

    let verifier = Verifier::new(smoke_config(PropagationMethod::Crown));
    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid graph certification input bounds");

    let certification = verifier
        .certify_graph_bounds("encoder_graph", &graph, &input_bounds, Some(0))
        .expect("graph timeout should remain structured control flow");

    match certification {
        BoundCertificationResult::Timeout {
            partial,
            actual_method,
            soundness,
        } => {
            assert!(
                partial.is_none(),
                "expected timeout without partial certificate"
            );
            assert_eq!(actual_method, MethodUsed::Crown);
            assert_eq!(soundness.mode(), VerificationSoundnessMode::Sound);
            assert!(
                soundness.heuristics_used().is_empty(),
                "unexpected timeout heuristics: {:?}",
                soundness.heuristics_used()
            );
        }
        other => panic!("expected timeout without partial certificate, got {other:?}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_certify_graph_bounds_rejects_beta_crown_3920() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    let graph = GraphNetwork::from_sequential(&network).expect("sequential graph conversion");

    let verifier = Verifier::new(smoke_config(PropagationMethod::BetaCrown));
    let input_bounds = BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid graph certification input bounds");

    let err = verifier
        .certify_graph_bounds("graph", &graph, &input_bounds, None)
        .expect_err("graph beta-crown certification should fail closed");

    assert!(
        matches!(err, NyError::UnsupportedOp(ref message) if message.contains("BetaCrown")),
        "expected unsupported-op beta-crown error, got {err:?}"
    );
}
