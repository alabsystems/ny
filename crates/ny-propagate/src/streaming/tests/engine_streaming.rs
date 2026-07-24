// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GemmEngine-aware streaming CROWN tests (#3959).
//!
//! Verifies that `propagate_crown_streaming_with_engine` threads the engine
//! through the backward pass and produces bounds matching the no-engine
//! baseline. Split from `crown_streaming.rs` to stay within file size limits.

use super::*;
use crate::streaming::*;
use ny_test_utils::CountingGemmEngine;

/// Streaming CROWN with GemmEngine must produce bounds matching the no-engine
/// baseline, verifying that the engine is threaded through the backward pass.
///
/// Regression for #3959: streaming/tests had zero GemmEngine coverage.
#[ntest::timeout(10000)]
#[test]
fn streaming_crown_with_engine_matches_baseline_3959() {
    // Non-trivial 3-layer linear network with deterministic weights.
    let mut network = Network::new();
    for i in 0..3 {
        let mut weight = Array2::<f32>::zeros((4, 4));
        for r in 0..4 {
            for c in 0..4 {
                let val = ((r * 7 + c * 11 + i * 13) % 10) as f32 * 0.02 - 0.1;
                weight[[r, c]] = val;
            }
        }
        let bias = Some(Array1::<f32>::zeros(4));
        let linear = LinearLayer::new(weight, bias).unwrap();
        network.add_layer(Layer::Linear(linear));
    }

    let lower = ArrayD::from_elem(ndarray::IxDyn(&[4]), -0.5_f32);
    let upper = ArrayD::from_elem(ndarray::IxDyn(&[4]), 0.5_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let verifier = StreamingVerifier::new(StreamingConfig::default());

    // Baseline: no engine
    let result_none = verifier
        .propagate_crown_streaming(&network, &input)
        .expect("streaming CROWN without engine should succeed");

    // With engine
    let engine = CountingGemmEngine::new();
    let result_eng = verifier
        .propagate_crown_streaming_with_engine(&network, &input, Some(&engine))
        .expect("streaming CROWN with engine should succeed");

    // Engine must have been invoked (3 linear layers, each uses GEMM)
    assert!(
        engine.gemm_calls() > 0,
        "GemmEngine should be invoked during streaming CROWN backward, got 0 calls"
    );

    // Bounds must match
    assert_eq!(result_eng.shape(), result_none.shape());
    let tol = 1e-5;
    for (i, (eng, none)) in result_eng
        .lower()
        .iter()
        .zip(result_none.lower().iter())
        .enumerate()
    {
        assert!(
            (eng - none).abs() < tol,
            "streaming CROWN lower mismatch at {i}: engine={eng}, none={none}"
        );
    }
    for (i, (eng, none)) in result_eng
        .upper()
        .iter()
        .zip(result_none.upper().iter())
        .enumerate()
    {
        assert!(
            (eng - none).abs() < tol,
            "streaming CROWN upper mismatch at {i}: engine={eng}, none={none}"
        );
    }
}

/// Streaming CROWN with engine on a network containing ReLU activations:
/// verifies the engine threads through Linear layers while ReLU backward
/// (element-wise slope) is handled by the trait path.
#[ntest::timeout(10000)]
#[test]
fn streaming_crown_with_engine_relu_network_3959() {
    use crate::layers::ReLULayer;

    let w1 = Array2::from_shape_vec((3, 2), vec![0.4, -0.2, -0.15, 0.5, 0.1, 0.3]).unwrap();
    let b1 = Some(Array1::from_vec(vec![0.05, -0.03, 0.02]));
    let linear1 = LinearLayer::new(w1, b1).unwrap();

    let w2 = Array2::from_shape_vec((2, 3), vec![0.3, -0.1, 0.25, -0.2, 0.35, 0.15]).unwrap();
    let b2 = Some(Array1::from_vec(vec![-0.02, 0.04]));
    let linear2 = LinearLayer::new(w2, b2).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));

    let lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![-1.0, -1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0, 1.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let verifier = StreamingVerifier::new(StreamingConfig::default());

    // Baseline
    let result_none = verifier
        .propagate_crown_streaming(&network, &input)
        .expect("streaming CROWN without engine should succeed");

    // With engine
    let engine = CountingGemmEngine::new();
    let result_eng = verifier
        .propagate_crown_streaming_with_engine(&network, &input, Some(&engine))
        .expect("streaming CROWN with engine should succeed on ReLU network");

    // Engine must be called (at least for the two Linear layers)
    assert!(
        engine.gemm_calls() > 0,
        "GemmEngine should be invoked for Linear layers in ReLU network, got 0 calls"
    );

    // Bounds must match
    let tol = 1e-5;
    for (i, (eng, none)) in result_eng
        .lower()
        .iter()
        .zip(result_none.lower().iter())
        .enumerate()
    {
        assert!(
            (eng - none).abs() < tol,
            "ReLU network lower mismatch at {i}: engine={eng}, none={none}"
        );
    }
    for (i, (eng, none)) in result_eng
        .upper()
        .iter()
        .zip(result_none.upper().iter())
        .enumerate()
    {
        assert!(
            (eng - none).abs() < tol,
            "ReLU network upper mismatch at {i}: engine={eng}, none={none}"
        );
    }
}
