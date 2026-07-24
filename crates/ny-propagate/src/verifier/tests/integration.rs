// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// Licensed under the Apache License, Version 2.0

use super::super::*;
use crate::layers::{Layer, LinearLayer, ReLULayer};
use crate::tests::crown::helpers::MockGpuCrownEngine;
use ndarray::{Array1, Array2};
use ny_core::{Bound, VerificationResult, VerificationSpec};
use std::sync::Arc;

#[ntest::timeout(10000)]
#[test]
fn test_verify_simple_linear_ibp() {
    // Simple 2x2 identity linear layer
    let weight = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let linear = LinearLayer::new(weight, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Ibp,
        ..Default::default()
    });

    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        vec![Bound::new(-2.0, 2.0), Bound::new(-2.0, 2.0)],
        Some(5000),
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();

    assert!(matches!(result, VerificationResult::Verified { .. }));
}

#[ntest::timeout(10000)]
#[test]
fn test_verify_simple_linear_crown() {
    // Simple 2x2 identity linear layer
    let weight = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let linear = LinearLayer::new(weight, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Crown,
        ..Default::default()
    });

    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        vec![Bound::new(-2.0, 2.0), Bound::new(-2.0, 2.0)],
        Some(5000),
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();

    assert!(matches!(result, VerificationResult::Verified { .. }));
}

#[ntest::timeout(10000)]
#[test]
fn test_verify_tight_bounds_unknown() {
    // Simple 2x2 linear layer with scaling
    let weight = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).unwrap(); // 2x scaling
    let linear = LinearLayer::new(weight, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Ibp,
        ..Default::default()
    });

    // Input: [-1, 1], output should be [-2, 2] but we require [-1, 1]
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)], // Too tight
        Some(5000),
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();

    assert!(matches!(result, VerificationResult::Unknown { .. }));
}

#[ntest::timeout(10000)]
#[test]
fn test_verify_with_relu() {
    // Linear + ReLU network
    let weight = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let linear = LinearLayer::new(weight, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::ReLU(ReLULayer));

    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Ibp,
        ..Default::default()
    });

    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        vec![Bound::new(0.0, 2.0), Bound::new(0.0, 2.0)], // ReLU clips negative
        Some(5000),
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();

    assert!(matches!(result, VerificationResult::Verified { .. }));
}

#[ntest::timeout(10000)]
#[test]
fn test_verify_with_bias() {
    // Linear layer with bias
    let weight = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let bias = Array1::from_vec(vec![1.0, -1.0]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Ibp,
        ..Default::default()
    });

    // Input [-1, 1] + bias [1, -1] = output [0, 2] and [-2, 0]
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        vec![Bound::new(-1.0, 3.0), Bound::new(-3.0, 1.0)],
        Some(5000),
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();

    assert!(matches!(result, VerificationResult::Verified { .. }));
}

#[ntest::timeout(10000)]
#[test]
fn test_verify_with_larger_network() {
    // Test with a larger 4x4 identity network
    let weight = Array2::from_shape_vec(
        (4, 4),
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap();
    let linear = LinearLayer::new(weight, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::Ibp,
        ..Default::default()
    });

    let spec = VerificationSpec::from_parts(
        vec![
            Bound::new(-1.0, 1.0),
            Bound::new(-1.0, 1.0),
            Bound::new(-1.0, 1.0),
            Bound::new(-1.0, 1.0),
        ],
        vec![
            Bound::new(-2.0, 2.0),
            Bound::new(-2.0, 2.0),
            Bound::new(-2.0, 2.0),
            Bound::new(-2.0, 2.0),
        ],
        Some(5000),
        None, // Use default 1D shape for linear layers
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();

    assert!(matches!(result, VerificationResult::Verified { .. }));
}

#[ntest::timeout(10000)]
#[test]
fn test_verify_beta_crown_with_engine_threads_gpu_fast_path() {
    let weight1 =
        Array2::from_shape_vec((4, 2), vec![1.0, 0.5, -0.5, 1.0, 0.3, -0.7, -0.2, 0.8]).unwrap();
    let weight2 = Array2::from_shape_vec(
        (4, 4),
        vec![
            0.5, -0.3, 0.7, 0.1, -0.4, 0.6, -0.2, 0.5, 0.3, 0.2, -0.5, 0.4, -0.1, 0.4, 0.3, -0.6,
        ],
    )
    .unwrap();
    let weight3 = Array2::from_shape_vec((1, 4), vec![1.0, -0.5, 0.3, 0.2]).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(weight1, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(weight2, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(weight3, None).unwrap()));

    let input = ny_tensor::BoundedTensor::new(
        Array1::from_vec(vec![-1.0, -1.0]).into_dyn(),
        Array1::from_vec(vec![1.0, 1.0]).into_dyn(),
    )
    .unwrap();
    let expected = network
        .propagate_crown_with_engine(&input, Some(&ny_core::NaiveCpuGemmEngine))
        .unwrap();
    let crown_upper = *expected
        .upper()
        .iter()
        .next()
        .expect("single-output CROWN tensor should have one element");
    let mock_gpu = MockGpuCrownEngine::from_expected(&expected);

    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::BetaCrown,
        ..Default::default()
    });
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        vec![Bound::new(crown_upper + 1.0, crown_upper + 10.0)],
        Some(5000),
        None,
    )
    .expect("valid beta-crown spec");

    let result = verifier
        .verify_with_engine(&network, &spec, Some(&mock_gpu))
        .unwrap();

    assert!(matches!(
        result,
        VerificationResult::Violated { .. } | VerificationResult::Unknown { .. }
    ));
    assert!(
        mock_gpu.gpu_calls() > 0,
        "Verifier::verify_with_engine should preserve the beta-crown GPU fast-path"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_verify_beta_crown_with_stored_engine_threads_gpu_fast_path() {
    let weight1 =
        Array2::from_shape_vec((4, 2), vec![1.0, 0.5, -0.5, 1.0, 0.3, -0.7, -0.2, 0.8]).unwrap();
    let weight2 = Array2::from_shape_vec(
        (4, 4),
        vec![
            0.5, -0.3, 0.7, 0.1, -0.4, 0.6, -0.2, 0.5, 0.3, 0.2, -0.5, 0.4, -0.1, 0.4, 0.3, -0.6,
        ],
    )
    .unwrap();
    let weight3 = Array2::from_shape_vec((1, 4), vec![1.0, -0.5, 0.3, 0.2]).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(weight1, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(weight2, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(weight3, None).unwrap()));

    let input = ny_tensor::BoundedTensor::new(
        Array1::from_vec(vec![-1.0, -1.0]).into_dyn(),
        Array1::from_vec(vec![1.0, 1.0]).into_dyn(),
    )
    .unwrap();
    let expected = network
        .propagate_crown_with_engine(&input, Some(&ny_core::NaiveCpuGemmEngine))
        .unwrap();
    let crown_upper = *expected
        .upper()
        .iter()
        .next()
        .expect("single-output CROWN tensor should have one element");
    let mock_gpu = Arc::new(MockGpuCrownEngine::from_expected(&expected));

    let verifier = Verifier::new_with_engine(
        PropagationConfig {
            method: PropagationMethod::BetaCrown,
            ..Default::default()
        },
        mock_gpu.clone(),
    );
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        vec![Bound::new(crown_upper + 1.0, crown_upper + 10.0)],
        Some(5000),
        None,
    )
    .expect("valid beta-crown spec");

    let result = verifier.verify(&network, &spec).unwrap();

    assert!(matches!(
        result,
        VerificationResult::Violated { .. } | VerificationResult::Unknown { .. }
    ));
    assert!(
        mock_gpu.gpu_calls() > 0,
        "Verifier::with_engine should preserve the beta-crown GPU fast-path for verify()"
    );
}

/// Regression for #2241: multi-output beta-CROWN with asymmetric thresholds.
///
/// Network: 2→2 identity + ReLU + 2→2 with bias [5, 5].
/// For input in [-0.1, 0.1]², output ∈ [4.9, 5.1]² (ReLU passes positive, bias shifts).
/// Spec: output[0] > 3.0, output[1] > 4.9 (second threshold is stricter).
///
/// BaB verifies min(all_outputs) > min(3.0, 4.9) = 3.0. BaB returns per-output bounds
/// (≈[5.0, 5.1] for both), so the per-output validation checks each output against its
/// own spec and correctly returns Verified.
///
/// The companion test `_second_threshold_stricter_2241` covers the case where the
/// property does NOT hold, testing the old-bug path directly.
#[ntest::timeout(60000)]
#[test]
fn test_verify_beta_crown_multi_output_asymmetric_thresholds_2241() {
    let mut network = Network::new();
    let w1 = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let b1 = Array1::from_vec(vec![0.0, 0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    let w2 = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let b2 = Array1::from_vec(vec![5.0, 5.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::BetaCrown,
        max_iterations: 10,
        tolerance: 1e-4,
        use_gpu: false,
        ..Default::default()
    });

    // Asymmetric thresholds: output[0] > 3.0, output[1] > 4.9.
    // Both outputs ∈ [4.9, 5.1] so both constraints hold.
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-0.1, 0.1), Bound::new(-0.1, 0.1)],
        vec![
            Bound::new_allow_infinite(3.0, f32::INFINITY),
            Bound::new_allow_infinite(4.9, f32::INFINITY),
        ],
        Some(30000),
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();

    // Must be Verified — not just "not Violated". The old test used a conditional
    // `if let Verified` that silently passed on Unknown (#2346).
    let output_bounds = match &result {
        VerificationResult::Verified { output_bounds, .. } => output_bounds,
        other => panic!(
            "Expected Verified for multi-output spec where both outputs satisfy, got {:?}",
            other
        ),
    };

    // Both outputs must have 2 bounds satisfying their respective specs.
    assert_eq!(output_bounds.len(), 2, "Expected 2 output bounds");
    assert!(
        output_bounds[0].lower() >= 3.0,
        "output[0] lower {} must be >= 3.0",
        output_bounds[0].lower()
    );
    assert!(
        output_bounds[1].lower() >= 4.9,
        "output[1] lower {} must be >= 4.9",
        output_bounds[1].lower()
    );
}

/// Regression for #2241: second output's threshold is stricter and unsatisfiable.
///
/// Network: 2→2 identity + ReLU + 2→2 with bias [10, 0].
/// output[0] = relu(x0) + 10 ∈ [10, 11], output[1] = relu(x1) ∈ [0, 1].
/// Spec: output[0] > -100, output[1] > 0.5.
///
/// The first threshold is -100 (trivially satisfied). The second threshold (0.5) is
/// NOT satisfiable because output[1] can be 0 (when x1 ∈ [-1, 0]).
///
/// Old code: threshold = -100 (first only). BaB easily verifies min > -100.
///   Returns Verified — UNSOUND because output[1] > 0.5 was never checked.
/// New code: threshold = -100 (min). BaB verifies min > -100. Per-output check
///   sees output[1]'s lower bound (0.0) < required (0.5) → returns Unknown.
///
/// This test is discriminating: old code returns Verified (bug), new code cannot.
#[ntest::timeout(60000)]
#[test]
fn test_verify_beta_crown_multi_output_second_threshold_stricter_2241() {
    let mut network = Network::new();
    let w1 = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let b1 = Array1::from_vec(vec![0.0, 0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    // output[0] = relu(x0) + 10, output[1] = relu(x1) + 0
    let w2 = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let b2 = Array1::from_vec(vec![10.0, 0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let verifier = Verifier::new(PropagationConfig {
        method: PropagationMethod::BetaCrown,
        max_iterations: 10,
        tolerance: 1e-4,
        use_gpu: false,
        ..Default::default()
    });

    // First threshold trivially easy (-100), second is impossible (0.5 > actual min 0).
    // Old code used threshold=-100 (first only), BaB returns Verified (UNSOUND BUG).
    // New code: per-output validation catches output[1]'s gap.
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        vec![
            Bound::new_allow_infinite(-100.0, f32::INFINITY),
            Bound::new_allow_infinite(0.5, f32::INFINITY),
        ],
        Some(30000),
        None,
    )
    .expect("valid test spec");

    let result = verifier.verify(&network, &spec).unwrap();
    // output[1] can be 0 < 0.5, so Verified would be unsound.
    assert!(
        !matches!(result, VerificationResult::Verified { .. }),
        "Regression #2241: beta-CROWN must not return Verified when output[1] \
         can be 0.0 but spec requires > 0.5. Old code used only first threshold \
         (-100) and returned Verified. Got: {:?}",
        result
    );
}

/// Regression for #2238: Empty output_bounds must be rejected.
/// Since #2367 added custom Deserialize validation, serde itself rejects empty
/// output_bounds before the verifier is ever invoked — this is strictly better
/// (fail-fast at the trust boundary).
#[ntest::timeout(10000)]
#[test]
fn test_verify_rejects_deserialized_empty_output_bounds_2238() {
    let spec_json = r#"{
        "input_bounds": [{"lower": -1.0, "upper": 1.0}, {"lower": -1.0, "upper": 1.0}],
        "output_bounds": [],
        "timeout_ms": 5000,
        "input_shape": null
    }"#;
    let result: Result<VerificationSpec, _> = serde_json::from_str(spec_json);
    assert!(
        result.is_err(),
        "Expected deserialization to reject empty output_bounds"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("output_bounds cannot be empty"),
        "Expected 'output_bounds cannot be empty' in error, got: {err_msg}"
    );
}

/// Regression for #2238: β-CROWN path — empty output_bounds rejected at deserialization.
/// Since #2367, the custom Deserialize impl validates output_bounds, so the verifier
/// never sees the invalid spec. This test confirms the deserialization guard.
#[ntest::timeout(10000)]
#[test]
fn test_verify_beta_crown_rejects_deserialized_empty_output_bounds_2238() {
    let spec_json = r#"{
        "input_bounds": [{"lower": -1.0, "upper": 1.0}, {"lower": -1.0, "upper": 1.0}],
        "output_bounds": [],
        "timeout_ms": 10000,
        "input_shape": null
    }"#;
    let result: Result<VerificationSpec, _> = serde_json::from_str(spec_json);
    assert!(
        result.is_err(),
        "Expected deserialization to reject empty output_bounds"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("output_bounds cannot be empty"),
        "Expected 'output_bounds cannot be empty' in error, got: {err_msg}"
    );
}
