// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verifier tests

use crate::layers::{AddLayer, LayerNormCrownMode, LayerNormLayer, NonZeroLayer, ReshapeLayer};
use crate::{
    GraphNetwork, GraphNode, Layer, LinearLayer, MulBinaryRelaxationMode, Network,
    PropagationConfig, PropagationMethod, ReLULayer, SqrtLayer, Verifier,
};
use ndarray::{arr1, arr2};
use ny_core::{
    Bound, HeuristicUsed, MethodUsed, NyError, VerificationResult, VerificationSoundnessMode,
    VerificationSpec,
};

// Keep smoke tests fast and deterministic on slower runners.
const SMOKE_MAX_ITERATIONS: usize = 20;
const SMOKE_BETA_ITERATIONS: usize = 5;
const SMOKE_TIMEOUT_MS: u64 = 10_000;
const SMOKE_VALUE_EPSILON: f32 = 1e-4;

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

fn assert_bounds_well_formed(bounds: &[Bound]) {
    for (idx, bound) in bounds.iter().enumerate() {
        assert!(
            bound.lower().is_finite() && bound.upper().is_finite(),
            "Bound[{idx}] contains non-finite value: lower={}, upper={}",
            bound.lower(),
            bound.upper()
        );
        assert!(
            bound.lower() <= bound.upper(),
            "Bound[{idx}] invalid: lower {} > upper {}",
            bound.lower(),
            bound.upper()
        );
    }
}

fn assert_bounds_no_nan(bounds: &[Bound]) {
    for (idx, bound) in bounds.iter().enumerate() {
        assert!(
            !bound.lower().is_nan() && !bound.upper().is_nan(),
            "Bound[{idx}] contains NaN: lower={}, upper={}",
            bound.lower(),
            bound.upper()
        );
        assert!(
            bound.lower() <= bound.upper(),
            "Bound[{idx}] invalid: lower {} > upper {}",
            bound.lower(),
            bound.upper()
        );
    }
}

fn assert_bounds_contain(bounds: &[Bound], values: &[f32]) {
    assert_eq!(
        bounds.len(),
        values.len(),
        "Expected {} bounds, got {}",
        values.len(),
        bounds.len()
    );
    for (idx, (bound, value)) in bounds.iter().zip(values.iter()).enumerate() {
        let lower = bound.lower() - SMOKE_VALUE_EPSILON;
        let upper = bound.upper() + SMOKE_VALUE_EPSILON;
        assert!(
            lower <= *value && *value <= upper,
            "Bound[{idx}] does not contain value {} (eps={}): [{}, {}]",
            value,
            SMOKE_VALUE_EPSILON,
            bound.lower(),
            bound.upper()
        );
    }
}

fn assert_bound_contains_values(bound: &Bound, values: &[f32]) {
    for value in values.iter().copied() {
        let lower = bound.lower() - SMOKE_VALUE_EPSILON;
        let upper = bound.upper() + SMOKE_VALUE_EPSILON;
        assert!(
            lower <= value && value <= upper,
            "Bound does not contain value {} (eps={}): [{}, {}]",
            value,
            SMOKE_VALUE_EPSILON,
            bound.lower(),
            bound.upper()
        );
    }
}

fn assert_bound_non_negative(bound: &Bound) {
    assert!(
        bound.lower() >= -SMOKE_VALUE_EPSILON,
        "Expected non-negative lower bound (eps={}): lower={}",
        SMOKE_VALUE_EPSILON,
        bound.lower()
    );
}

fn assert_result_bounds_well_formed(result: &VerificationResult) {
    match result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_bounds_well_formed(output_bounds);
        }
        VerificationResult::Unknown { bounds, .. } => {
            assert_bounds_well_formed(bounds);
        }
        VerificationResult::Timeout {
            partial_bounds: Some(bounds),
            ..
        } => {
            assert_bounds_well_formed(bounds);
        }
        VerificationResult::Timeout {
            partial_bounds: None,
            ..
        } => {}
        VerificationResult::Violated { .. } => {}
    }
}

fn assert_result_bounds_no_nan(result: &VerificationResult) {
    match result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_bounds_no_nan(output_bounds);
        }
        VerificationResult::Unknown { bounds, .. } => {
            assert_bounds_no_nan(bounds);
        }
        VerificationResult::Timeout {
            partial_bounds: Some(bounds),
            ..
        } => {
            assert_bounds_no_nan(bounds);
        }
        VerificationResult::Timeout {
            partial_bounds: None,
            ..
        } => {}
        VerificationResult::Violated { .. } => {}
    }
}

fn assert_result_bounds_len(result: &VerificationResult, expected_len: usize) {
    match result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(
                output_bounds.len(),
                expected_len,
                "Expected {} output bounds, got {}",
                expected_len,
                output_bounds.len()
            );
        }
        VerificationResult::Unknown { bounds, .. } => {
            assert_eq!(
                bounds.len(),
                expected_len,
                "Expected {} output bounds, got {}",
                expected_len,
                bounds.len()
            );
        }
        VerificationResult::Timeout {
            partial_bounds: Some(bounds),
            ..
        } => {
            assert_eq!(
                bounds.len(),
                expected_len,
                "Expected {} output bounds, got {}",
                expected_len,
                bounds.len()
            );
        }
        VerificationResult::Timeout {
            partial_bounds: None,
            ..
        } => {
            panic!(
                "Expected {} output bounds, got timeout with no bounds",
                expected_len
            );
        }
        VerificationResult::Violated { .. } => {
            panic!(
                "Expected {} output bounds, got violation result",
                expected_len
            );
        }
    }
}

fn assert_result_actual_method_tag(result: &VerificationResult, expected: MethodUsed) {
    assert_eq!(result.actual_method_tag(), Some(&expected));
}

// ============================================================
// VERIFIER TESTS
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_verifier_ibp_simple_network() {
    // Create a simple 2-layer network: Linear -> ReLU
    let mut network = Network::new();
    let weight = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let bias = arr1(&[0.1, -0.1]);
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Create verifier with IBP
    let config = smoke_config(PropagationMethod::Ibp);
    let verifier = Verifier::new(config);

    // Create specification
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); 2],
        None,
        None,
    )
    .expect("valid test spec");

    // Verify
    let result = verifier.verify(&network, &spec).unwrap();
    let provenance = result.provenance();
    assert_result_bounds_well_formed(&result);
    assert_result_bounds_len(&result, spec.output_bounds().len());

    // Should get bounds (not just input echoed back)
    match &result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(output_bounds.len(), 2);
            // Output bounds should be wider than input due to ReLU
            println!("IBP output bounds: {:?}", output_bounds);
            assert_bounds_contain(output_bounds, &[0.1, 0.0]);
        }
        _ => panic!("Expected Verified, got {:?}", result),
    }
    assert_eq!(provenance.mode(), VerificationSoundnessMode::Sound);
    assert!(
        provenance.heuristics_used().is_empty(),
        "Expected no heuristics for IBP smoke test, got {:?}",
        provenance.heuristics_used()
    );
    assert_result_actual_method_tag(&result, MethodUsed::Ibp);
}

#[ntest::timeout(10000)]
#[test]
fn test_verifier_sqrt_negative_domain_marks_heuristic() {
    let mut network = Network::new();
    network.add_layer(Layer::Sqrt(SqrtLayer));

    let config = smoke_config(PropagationMethod::Ibp);
    let verifier = Verifier::new(config);

    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 4.0)],
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        None,
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();
    // β-CROWN reports only lower-bound guarantees, so upper bounds may be infinite.
    assert_result_bounds_no_nan(&result);
    assert_result_bounds_len(&result, spec.output_bounds().len());
    let provenance = result.provenance();

    assert_eq!(provenance.mode(), VerificationSoundnessMode::Heuristic);
    assert!(
        provenance
            .heuristics_used()
            .iter()
            .any(|h| matches!(h, HeuristicUsed::SqrtNegativeDomain { num_nodes: 1 })),
        "Expected SqrtNegativeDomain heuristic, got {:?}",
        provenance.heuristics_used()
    );
    match &result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(output_bounds.len(), 1);
            assert_bounds_well_formed(output_bounds);
            assert_bound_non_negative(&output_bounds[0]);
            assert_bound_contains_values(&output_bounds[0], &[0.0, 2.0]);
        }
        _ => panic!("Expected Verified, got {:?}", result),
    }
    assert_result_actual_method_tag(&result, MethodUsed::Ibp);
}

#[ntest::timeout(10000)]
#[test]
fn test_verifier_sqrt_positive_domain_is_sound() {
    let mut network = Network::new();
    network.add_layer(Layer::Sqrt(SqrtLayer));

    let config = smoke_config(PropagationMethod::Crown);
    let verifier = Verifier::new(config);

    let spec = VerificationSpec::from_parts(
        vec![Bound::new(0.0, 4.0)],
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        None,
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();
    assert_result_bounds_well_formed(&result);
    assert_result_bounds_len(&result, spec.output_bounds().len());
    let provenance = result.provenance();

    assert_eq!(provenance.mode(), VerificationSoundnessMode::Sound);
    assert!(
        provenance.heuristics_used().is_empty(),
        "Expected no heuristics for sqrt positive domain, got {:?}",
        provenance.heuristics_used()
    );
    match &result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(output_bounds.len(), 1);
            assert_bounds_well_formed(output_bounds);
            assert_bound_non_negative(&output_bounds[0]);
            assert_bound_contains_values(&output_bounds[0], &[0.0, 1.0, 2.0]);
        }
        _ => panic!("Expected Verified, got {:?}", result),
    }
    assert_result_actual_method_tag(&result, MethodUsed::Crown);
}

#[ntest::timeout(10000)]
#[test]
fn test_verifier_graph_sqrt_negative_domain_marks_heuristic() {
    let mut network = Network::new();
    network.add_layer(Layer::Sqrt(SqrtLayer));

    let graph = GraphNetwork::from_sequential(&network).unwrap();

    let config = smoke_config(PropagationMethod::Ibp);
    let verifier = Verifier::new(config);

    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 4.0)],
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        None,
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify_graph(&graph, &spec).unwrap();
    let provenance = result.provenance();
    assert_result_bounds_well_formed(&result);
    assert_result_bounds_len(&result, spec.output_bounds().len());

    assert_eq!(provenance.mode(), VerificationSoundnessMode::Heuristic);
    assert!(
        provenance
            .heuristics_used()
            .iter()
            .any(|h| matches!(h, HeuristicUsed::SqrtNegativeDomain { num_nodes: 1 })),
        "Expected SqrtNegativeDomain heuristic, got {:?}",
        provenance.heuristics_used()
    );
    match &result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(output_bounds.len(), 1);
            assert_bounds_well_formed(output_bounds);
            assert_bound_non_negative(&output_bounds[0]);
            assert_bound_contains_values(&output_bounds[0], &[0.0, 2.0]);
        }
        _ => panic!("Expected Verified, got {:?}", result),
    }
    assert_result_actual_method_tag(&result, MethodUsed::Ibp);
}

#[ntest::timeout(10000)]
#[test]
fn test_verifier_graph_crown_uses_optimized_public_path_3619() {
    let mut network = Network::new();
    let w1 = arr2(&[[0.5, 0.3], [-0.4, 0.6], [0.2, -0.3], [-0.1, 0.4]]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1, Some(arr1(&[0.1, -0.1, 0.0, 0.05]))).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    let w2 = arr2(&[
        [0.3, -0.2, 0.4, 0.1],
        [-0.3, 0.5, -0.1, 0.2],
        [0.2, 0.1, -0.3, 0.4],
        [0.1, -0.4, 0.2, -0.1],
    ]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2, Some(arr1(&[0.0, 0.1, -0.05, 0.02]))).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    let w3 = arr2(&[[0.4, 0.3, -0.2, 0.1], [-0.3, 0.2, 0.4, -0.1]]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w3, Some(arr1(&[0.0, 0.0]))).unwrap(),
    ));
    let graph = GraphNetwork::from_sequential(&network).unwrap();

    let input_bounds = ny_tensor::BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();
    let fixed_slope_bounds = graph.propagate_crown_fixed_slope(&input_bounds).unwrap();
    let fixed_slope_width: f32 = fixed_slope_bounds
        .upper()
        .iter()
        .zip(fixed_slope_bounds.lower().iter())
        .map(|(u, l)| u - l)
        .sum();

    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-0.5, 0.5), Bound::new(-0.5, 0.5)],
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); 2],
        None,
        None,
    )
    .expect("valid test spec");

    let mut config = smoke_config(PropagationMethod::Crown);
    config.max_iterations = 50;
    let verifier = Verifier::new(config);
    let result = verifier.verify_graph(&graph, &spec).unwrap();
    let crown_bounds = match &result {
        VerificationResult::Verified { output_bounds, .. } => output_bounds,
        _ => panic!("Expected Verified, got {:?}", result),
    };
    let crown_width: f32 = crown_bounds
        .iter()
        .map(|bound| bound.upper() - bound.lower())
        .sum();

    assert_result_actual_method_tag(&result, MethodUsed::Crown);
    assert!(
        crown_width <= fixed_slope_width + 1e-4,
        "verifier graph CROWN width {crown_width} should be <= fixed-slope width \
         {fixed_slope_width}"
    );
    let improvement_pct = 100.0 * (1.0 - crown_width / fixed_slope_width);
    assert!(
        improvement_pct > 0.1,
        "verifier graph CROWN should improve over fixed-slope baseline after #3619; \
         got {improvement_pct:.4}%"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_verifier_graph_crown_preserves_soundness_refusal_3706() {
    let hidden = 4;
    let mut network = Network::new();
    let weight1 = arr2(&[
        [0.3, -0.2, 0.1, 0.0],
        [-0.1, 0.4, 0.2, -0.3],
        [0.2, 0.1, -0.4, 0.3],
        [0.0, -0.3, 0.5, 0.2],
    ]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(weight1, Some(arr1(&[0.1, -0.1, 0.0, 0.05]))).unwrap(),
    ));
    network.add_layer(Layer::LayerNorm(
        LayerNormLayer::new_default(hidden, 1e-5).unwrap(),
    ));
    let weight2 = arr2(&[
        [0.2, 0.3, -0.1, 0.4],
        [-0.4, 0.1, 0.5, -0.2],
        [0.1, -0.3, 0.2, 0.3],
        [0.3, 0.2, -0.2, 0.1],
    ]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(weight2, Some(arr1(&[0.0, 0.05, -0.05, 0.1]))).unwrap(),
    ));

    let mut graph = GraphNetwork::from_sequential(&network).expect("sequential graph conversion");
    assert_eq!(
        graph.set_layernorm_crown_mode(LayerNormCrownMode::Sound),
        1,
        "expected exactly one LayerNorm node"
    );

    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-0.5, 0.5); hidden],
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); hidden],
        None,
        None,
    )
    .expect("valid verifier spec");

    let verifier = Verifier::new(smoke_config(PropagationMethod::Crown));
    let result = verifier.verify_graph(&graph, &spec);

    assert!(
        matches!(result, Err(NyError::SoundnessRefusal(_))),
        "expected SoundnessRefusal from graph verifier, got {result:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_verifier_graph_crown_unsupported_op_still_falls_back_to_ibp_3706() {
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

    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0); 2],
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); 2],
        None,
        None,
    )
    .expect("valid verifier spec");

    let mut crown_config = smoke_config(PropagationMethod::Crown);
    // Skip the optimized alpha entrypoint so this regression exercises the
    // verifier's fixed-slope fallback chain directly.
    crown_config.mul_binary_relaxation = MulBinaryRelaxationMode::Middle;
    let verifier = Verifier::new(crown_config);
    let result = verifier
        .verify_graph(&graph, &spec)
        .expect("UnsupportedOp should still degrade to IBP");

    assert_result_bounds_well_formed(&result);
    assert_result_actual_method_tag(&result, MethodUsed::Ibp);

    let ibp_result = Verifier::new(smoke_config(PropagationMethod::Ibp))
        .verify_graph(&graph, &spec)
        .expect("direct IBP verification should succeed");
    let fallback_bounds = match &result {
        VerificationResult::Verified { output_bounds, .. } => output_bounds,
        _ => panic!("Expected Verified fallback result, got {:?}", result),
    };
    let ibp_bounds = match &ibp_result {
        VerificationResult::Verified { output_bounds, .. } => output_bounds,
        _ => panic!("Expected direct IBP verification, got {:?}", ibp_result),
    };
    assert_eq!(fallback_bounds.len(), ibp_bounds.len());
    for (idx, (fallback, ibp)) in fallback_bounds.iter().zip(ibp_bounds.iter()).enumerate() {
        assert!(
            (fallback.lower() - ibp.lower()).abs() <= 1e-6,
            "fallback lower[{idx}] = {} should match IBP {}",
            fallback.lower(),
            ibp.lower()
        );
        assert!(
            (fallback.upper() - ibp.upper()).abs() <= 1e-6,
            "fallback upper[{idx}] = {} should match IBP {}",
            fallback.upper(),
            ibp.upper()
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_verifier_crown_simple_network() {
    // Create a simple 2-layer network: Linear -> ReLU
    let mut network = Network::new();
    let weight = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let bias = arr1(&[0.1, -0.1]);
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Create verifier with CROWN
    let config = smoke_config(PropagationMethod::Crown);
    let verifier = Verifier::new(config);

    // Create specification
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); 2],
        None,
        None,
    )
    .expect("valid test spec");

    // Verify
    let result = verifier.verify(&network, &spec).unwrap();
    let provenance = result.provenance();
    assert_result_bounds_well_formed(&result);
    assert_result_bounds_len(&result, spec.output_bounds().len());

    match &result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(output_bounds.len(), 2);
            println!("CROWN output bounds: {:?}", output_bounds);
            assert_bounds_contain(output_bounds, &[0.1, 0.0]);
        }
        _ => panic!("Expected Verified, got {:?}", result),
    }
    assert_eq!(provenance.mode(), VerificationSoundnessMode::Sound);
    assert!(
        provenance.heuristics_used().is_empty(),
        "Expected no heuristics for CROWN smoke test, got {:?}",
        provenance.heuristics_used()
    );
    assert_result_actual_method_tag(&result, MethodUsed::Crown);
}

#[ntest::timeout(10000)]
#[test]
fn test_verifier_ibp_vs_crown_comparison() {
    // Compare IBP and CROWN - both should produce valid bounds
    // Note: CROWN is not always tighter than IBP, especially for shallow networks
    let mut network = Network::new();
    let weight = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let bias = arr1(&[0.1, -0.1]);
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0], [1.0, -1.0]]), None).unwrap(),
    ));

    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-0.5, 0.5), Bound::new(-0.5, 0.5)],
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); 2],
        None,
        None,
    )
    .expect("valid test spec");

    // IBP
    let ibp_config = smoke_config(PropagationMethod::Ibp);
    let ibp_verifier = Verifier::new(ibp_config);
    let ibp_result = ibp_verifier.verify(&network, &spec).unwrap();
    assert_result_bounds_well_formed(&ibp_result);
    assert_result_bounds_len(&ibp_result, spec.output_bounds().len());

    // CROWN
    let crown_config = smoke_config(PropagationMethod::Crown);
    let crown_verifier = Verifier::new(crown_config);
    let crown_result = crown_verifier.verify(&network, &spec).unwrap();
    assert_result_bounds_well_formed(&crown_result);
    assert_result_bounds_len(&crown_result, spec.output_bounds().len());

    // Extract bounds
    let (ibp_bounds, crown_bounds) = match (&ibp_result, &crown_result) {
        (
            VerificationResult::Verified {
                output_bounds: ibp, ..
            },
            VerificationResult::Verified {
                output_bounds: crown,
                ..
            },
        ) => (ibp, crown),
        _ => panic!(
            "Expected Verified for both methods, got ibp={:?}, crown={:?}",
            ibp_result, crown_result
        ),
    };
    assert_result_actual_method_tag(&ibp_result, MethodUsed::Ibp);
    assert_result_actual_method_tag(&crown_result, MethodUsed::Crown);
    assert_bounds_well_formed(ibp_bounds);
    assert_bounds_well_formed(crown_bounds);

    // Calculate widths
    let ibp_width: f32 = ibp_bounds.iter().map(|b| b.upper() - b.lower()).sum();
    let crown_width: f32 = crown_bounds.iter().map(|b| b.upper() - b.lower()).sum();

    println!("IBP total width: {}", ibp_width);
    println!("CROWN total width: {}", crown_width);
    println!("IBP bounds: {:?}", ibp_bounds);
    println!("CROWN bounds: {:?}", crown_bounds);

    // Both should produce finite bounds
    assert!(ibp_width.is_finite(), "IBP should produce finite bounds");
    assert!(
        crown_width.is_finite(),
        "CROWN should produce finite bounds"
    );

    // Both methods should be sound - this is validated by other tests
    // Note: CROWN can be looser than IBP for some shallow networks due to
    // the linear relaxation overhead when ReLU regions are wide
}

#[ntest::timeout(10000)]
#[test]
fn test_verifier_alpha_crown() {
    // Test α-CROWN verification
    let mut network = Network::new();
    let weight = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let bias = arr1(&[0.1, -0.1]);
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let config = smoke_config(PropagationMethod::AlphaCrown);
    let verifier = Verifier::new(config);

    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); 2],
        None,
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();
    assert_result_bounds_well_formed(&result);
    assert_result_bounds_len(&result, spec.output_bounds().len());
    let provenance = result.provenance();

    match &result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(output_bounds.len(), 2);
            println!("α-CROWN output bounds: {:?}", output_bounds);
            assert_bounds_contain(output_bounds, &[0.1, 0.0]);
        }
        _ => panic!("Expected Verified, got {:?}", result),
    }
    assert_eq!(provenance.mode(), VerificationSoundnessMode::Sound);
    assert!(
        provenance.heuristics_used().is_empty(),
        "Expected no heuristics for α-CROWN smoke test, got {:?}",
        provenance.heuristics_used()
    );
    assert_result_actual_method_tag(&result, MethodUsed::AlphaCrown);
}

#[ntest::timeout(10000)]
#[test]
fn test_verifier_beta_crown() {
    // Test β-CROWN verification
    let mut network = Network::new();

    // Simple network that should verify output > -10
    let weight = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let bias = arr1(&[1.0, 1.0]); // Positive bias
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    // Final layer: 2 inputs -> 1 output (weight shape [1, 2])
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let config = smoke_config(PropagationMethod::BetaCrown);
    let verifier = Verifier::new(config);

    // Specify output > -10
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        // Output is in [0, 5] for the chosen input bounds, so keep the spec finite.
        vec![Bound::new(-10.0, 5.0)],
        Some(SMOKE_TIMEOUT_MS),
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();
    // β-CROWN reports only lower-bound guarantees, so upper bounds may be infinite.
    assert_result_bounds_no_nan(&result);
    assert_result_bounds_len(&result, spec.output_bounds().len());

    println!("β-CROWN result: {:?}", result);

    // Should verify (output is positive due to ReLU + positive bias)
    match &result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(output_bounds.len(), 1);
            assert!(output_bounds[0].lower() >= -10.0);
            // β-CROWN guarantees finite lower bounds; upper bounds may be unbounded.
            assert!(
                output_bounds[0].lower().is_finite(),
                "Expected finite β-CROWN lower bound, got {:?}",
                output_bounds[0]
            );
            println!("Verified with bounds: {:?}", output_bounds);
            assert_bounds_no_nan(output_bounds);
            assert_bounds_contain(output_bounds, &[2.0]);
        }
        VerificationResult::Unknown { bounds, reason, .. } => {
            assert_eq!(bounds.len(), 1);
            println!("Unknown: {} (bounds: {:?})", reason, bounds);
            assert_bounds_no_nan(bounds);
        }
        VerificationResult::Violated { .. } | VerificationResult::Timeout { .. } => {
            panic!("β-CROWN should not violate or time out: {:?}", result);
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_verifier_spec_satisfied() {
    // Test that verification correctly checks output spec
    let mut network = Network::new();

    // Network that produces output in [0, 1]
    let weight = arr2(&[[0.5]]);
    let bias = arr1(&[0.5]);
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));

    let config = smoke_config(PropagationMethod::Ibp);
    let verifier = Verifier::new(config);

    // Tight input, should produce output near 0.5
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(0.0, 0.0)],
        vec![Bound::new(0.0, 1.0)],
        None,
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();
    assert_result_bounds_well_formed(&result);
    assert_result_bounds_len(&result, spec.output_bounds().len());

    // Should verify since 0.5 is in [0, 1]
    match result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(output_bounds.len(), 1);
            assert!(output_bounds[0].lower() >= 0.0);
            assert!(output_bounds[0].upper() <= 1.0);
            assert_bounds_well_formed(&output_bounds);
        }
        _ => panic!("Expected Verified, got {:?}", result),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_verifier_spec_violated() {
    // Test that verification correctly detects spec violation
    let mut network = Network::new();

    // Network that produces output around 1.5
    let weight = arr2(&[[1.0]]);
    let bias = arr1(&[1.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));

    let config = smoke_config(PropagationMethod::Ibp);
    let verifier = Verifier::new(config);

    // Input in [-0.5, 0.5], output will be in [0.5, 1.5]
    // Spec requires output in [0, 1] - will be violated
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-0.5, 0.5)],
        vec![Bound::new(0.0, 1.0)],
        None,
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();
    assert_result_bounds_well_formed(&result);
    assert_result_bounds_len(&result, spec.output_bounds().len());

    // Should be Unknown since output bounds exceed spec
    match result {
        VerificationResult::Unknown { bounds, reason, .. } => {
            println!(
                "Correctly detected violation: {} (bounds: {:?})",
                reason, bounds
            );
            assert!(bounds[0].upper() > 1.0, "Upper bound should exceed spec");
            assert_bounds_well_formed(&bounds);
        }
        _ => panic!("Expected Unknown, got {:?}", result),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_verifier_empty_network() {
    // Empty network should just pass input through
    let network = Network::new();

    let config = smoke_config(PropagationMethod::Ibp);
    let verifier = Verifier::new(config);

    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0)],
        vec![Bound::new(-2.0, 2.0)],
        None,
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();
    assert_result_bounds_well_formed(&result);
    assert_result_bounds_len(&result, spec.output_bounds().len());

    // Should verify since input bounds are within output spec
    match result {
        VerificationResult::Verified { output_bounds, .. } => {
            assert_eq!(output_bounds.len(), 1);
            assert!((output_bounds[0].lower() - (-1.0)).abs() < 1e-5);
            assert!((output_bounds[0].upper() - 1.0).abs() < 1e-5);
            assert_bounds_well_formed(&output_bounds);
        }
        _ => panic!("Expected Verified, got {:?}", result),
    }
}
