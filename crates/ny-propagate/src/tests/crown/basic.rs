// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN and α-CROWN algorithm tests.

use super::super::proptest_soundness::{sample_points, FP_TOLERANCE};
use super::*;
use crate::layers::activations::silu_critical_point;
use ndarray::{arr1, arr2, Array1, Array2, ArrayD, IxDyn};

// ============================================================
// CROWN TESTS
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_crown_single_linear_matches_ibp() {
    // For a single linear layer, CROWN should produce exact same bounds as IBP
    // because both are sound and linear layers preserve linearity.
    let weight = arr2(&[[1.0, 2.0], [3.0, -1.0]]);
    let bias = arr1(&[1.0, -1.0]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    // For pure linear network, CROWN and IBP should produce identical bounds
    for i in 0..2 {
        assert!(
            (ibp_output.lower()[[i]] - crown_output.lower()[[i]]).abs() < 1e-5,
            "lower[{}]: IBP={} CROWN={}",
            i,
            ibp_output.lower()[[i]],
            crown_output.lower()[[i]]
        );
        assert!(
            (ibp_output.upper()[[i]] - crown_output.upper()[[i]]).abs() < 1e-5,
            "upper[{}]: IBP={} CROWN={}",
            i,
            ibp_output.upper()[[i]],
            crown_output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_linear_relu_positive_region() {
    // When ReLU input is entirely positive, CROWN = IBP
    // W = [[1, 1]], b = [5] -> output always >= 5 for input in [0,1]
    let w1 = arr2(&[[1.0, 1.0]]);
    let b1 = arr1(&[5.0]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    // Pre-ReLU bounds: [5, 7], all positive, so ReLU is identity
    // Both should give same bounds
    assert!((ibp_output.lower()[[0]] - 5.0).abs() < 1e-5);
    assert!((ibp_output.upper()[[0]] - 7.0).abs() < 1e-5);
    assert!((crown_output.lower()[[0]] - 5.0).abs() < 1e-5);
    assert!((crown_output.upper()[[0]] - 7.0).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_linear_relu_negative_region() {
    // When ReLU input is entirely negative, both give zeros
    // W = [[1, 1]], b = [-5] -> output in [-5, -3] for input in [0,1]
    let w1 = arr2(&[[1.0, 1.0]]);
    let b1 = arr1(&[-5.0]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    // Pre-ReLU bounds: [-5, -3], all negative, so ReLU outputs 0
    assert!((ibp_output.lower()[[0]] - 0.0).abs() < 1e-5);
    assert!((ibp_output.upper()[[0]] - 0.0).abs() < 1e-5);
    assert!((crown_output.lower()[[0]] - 0.0).abs() < 1e-5);
    assert!((crown_output.upper()[[0]] - 0.0).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_linear_relu_crossing() {
    // When ReLU crosses zero, CROWN uses linear relaxation
    // W = [[1, 1]], b = [-0.5] -> pre-ReLU in [-0.5, 1.5] for input in [0,1]
    let w1 = arr2(&[[1.0, 1.0]]);
    let b1 = arr1(&[-0.5]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    // Pre-ReLU bounds: [-0.5, 1.5], crossing zero
    // IBP: lower = max(0, -0.5) = 0, upper = max(0, 1.5) = 1.5
    assert!((ibp_output.lower()[[0]] - 0.0).abs() < 1e-5);
    assert!((ibp_output.upper()[[0]] - 1.5).abs() < 1e-5);

    // CROWN with linear relaxation for crossing ReLU:
    // The linear approximation may give a looser lower bound than IBP
    // because CROWN uses y >= alpha*x which can be negative when x < 0
    // But CROWN should be SOUND: actual outputs are within bounds

    // Verify CROWN bounds are sound by testing concrete inputs
    let test_inputs = [
        arr1(&[0.0, 0.0]),
        arr1(&[1.0, 1.0]),
        arr1(&[0.5, 0.5]),
        arr1(&[0.0, 1.0]),
        arr1(&[1.0, 0.0]),
    ];

    for x in &test_inputs {
        let z = w1.dot(x) + &b1;
        let y = z.mapv(|v| v.max(0.0));

        assert!(
            y[0] >= crown_output.lower()[[0]] - 1e-5,
            "CROWN lower {} not sound for input {:?}, actual output {}",
            crown_output.lower()[[0]],
            x,
            y[0]
        );
        assert!(
            y[0] <= crown_output.upper()[[0]] + 1e-5,
            "CROWN upper {} not sound for input {:?}, actual output {}",
            crown_output.upper()[[0]],
            x,
            y[0]
        );
    }

    // CROWN upper bound should be at least as tight as IBP
    assert!(
        crown_output.upper()[[0]] <= ibp_output.upper()[[0]] + 1e-5,
        "CROWN upper {} > IBP upper {}",
        crown_output.upper()[[0]],
        ibp_output.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_linear_silu_soundness() {
    // SiLU should be supported by sequential CROWN propagation.
    let w1 = arr2(&[[1.0]]);
    let b1 = arr1(&[0.0]);
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::SiLU(SiLULayer::new()));

    let input = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();

    let crown_output = network.propagate_crown(&input).unwrap();

    let silu = SiLULayer::new();
    let mut samples = sample_points(-2.0, 2.0, 41);
    let critical = silu_critical_point();
    if (-2.0..=2.0).contains(&critical) {
        samples.push(critical);
    }
    for x in samples {
        let y = silu.eval(x);
        assert!(
            y >= crown_output.lower()[[0]] - FP_TOLERANCE,
            "SiLU({})={} below CROWN lower {}",
            x,
            y,
            crown_output.lower()[[0]]
        );
        assert!(
            y <= crown_output.upper()[[0]] + FP_TOLERANCE,
            "SiLU({})={} above CROWN upper {}",
            x,
            y,
            crown_output.upper()[[0]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_silu_critical_interval_soundness() {
    // Validate SiLU CROWN relaxation on an interval spanning the critical point (~-1.28).
    let mut network = Network::new();
    network.add_layer(Layer::SiLU(SiLULayer::new()));

    let lower = -3.0;
    let upper = 1.5;
    let input = BoundedTensor::new(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn()).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    let silu = SiLULayer::new();
    let mut samples = sample_points(lower, upper, 41);
    let critical = silu_critical_point();
    if (lower..=upper).contains(&critical) {
        samples.push(critical);
    }
    for x in samples {
        let y = silu.eval(x);
        assert!(
            y >= crown_output.lower()[[0]] - FP_TOLERANCE,
            "SiLU({})={} below CROWN lower {}",
            x,
            y,
            crown_output.lower()[[0]]
        );
        assert!(
            y <= crown_output.upper()[[0]] + FP_TOLERANCE,
            "SiLU({})={} above CROWN upper {}",
            x,
            y,
            crown_output.upper()[[0]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_negative_affine_scale_mulconstant_then_relu() {
    // Regression test for negative affine scaling in CROWN backward substitution (#306, #307).
    //
    // Network: x -> (-1 * x) -> ReLU
    // Input: x ∈ [-1, 1]
    // Exact range: ReLU(-x) ∈ [0, 1]
    let mut network = Network::new();
    network.add_layer(Layer::MulConstant(MulConstantLayer::scalar(-1.0)));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    assert!(
        (crown_output.lower()[[0]] - 0.0).abs() < 1e-5,
        "expected lower ≈ 0 for ReLU(-x), got {}",
        crown_output.lower()[[0]]
    );
    assert!(
        (crown_output.upper()[[0]] - 1.0).abs() < 1e-5,
        "expected upper ≈ 1 for ReLU(-x), got {}",
        crown_output.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_negative_affine_scale_batchnorm_then_relu() {
    // Regression test for negative BatchNorm scale in CROWN backward substitution (#306, #307).
    //
    // BatchNorm is linear: y = scale * x + bias.
    // Choose scale=-1, bias=0 so this reduces to y = -x, then ReLU.
    // Input: x ∈ [-1, 1] -> output ∈ [0, 1].
    let scale = ArrayD::from_elem(IxDyn(&[1]), -1.0_f32);
    let bias = ArrayD::from_elem(IxDyn(&[1]), 0.0_f32);
    let bn = BatchNormLayer::from_scale_bias(scale, bias).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::BatchNorm(bn));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    assert!(
        (crown_output.lower()[[0]] - 0.0).abs() < 1e-5,
        "expected lower ≈ 0 for ReLU(BN(x)) with scale=-1, got {}",
        crown_output.lower()[[0]]
    );
    assert!(
        (crown_output.upper()[[0]] - 1.0).abs() < 1e-5,
        "expected upper ≈ 1 for ReLU(BN(x)) with scale=-1, got {}",
        crown_output.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_mlp_tighter_than_ibp() {
    // 2-layer MLP: Linear(2->2) -> ReLU -> Linear(2->1)
    // For deeper networks with crossing ReLUs, CROWN should give tighter bounds
    //
    // First layer: W1 = [[1, -1], [-1, 1]], b1 = [0, 0]
    // For input in [-1, 1]^2:
    //   z1[0] = x0 - x1 in [-2, 2]
    //   z1[1] = -x0 + x1 in [-2, 2]
    // After ReLU: both outputs in [0, 2]
    //
    // Second layer: W2 = [[1, 1]], b2 = [0]
    // Output: a1[0] + a1[1]
    //
    // IBP: Each ReLU output in [0, 2], so sum in [0, 4]
    // But actually: a1[0] + a1[1] = max(x0-x1, 0) + max(x1-x0, 0) = |x0-x1|
    // For x in [-1,1]^2: |x0-x1| in [0, 2], so tighter bounds are [0, 2]
    //
    // CROWN exploits the linear relationship and should give tighter bounds

    let w1 = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    let w2 = arr2(&[[1.0, 1.0]]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, None).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    // IBP gives [0, 4] due to overestimation
    assert!(
        (ibp_output.lower()[[0]] - 0.0).abs() < 1e-5,
        "IBP lower = {}",
        ibp_output.lower()[[0]]
    );
    assert!(
        (ibp_output.upper()[[0]] - 4.0).abs() < 1e-5,
        "IBP upper = {}",
        ibp_output.upper()[[0]]
    );

    // CROWN should give tighter bounds (at least as tight as IBP)
    assert!(
        crown_output.lower()[[0]] >= ibp_output.lower()[[0]] - 1e-5,
        "CROWN lower {} should be >= IBP lower {}",
        crown_output.lower()[[0]],
        ibp_output.lower()[[0]]
    );
    assert!(
        crown_output.upper()[[0]] <= ibp_output.upper()[[0]] + 1e-5,
        "CROWN upper {} should be <= IBP upper {}",
        crown_output.upper()[[0]],
        ibp_output.upper()[[0]]
    );

    // Note: The exact tightness depends on the ReLU relaxation heuristic
    // With default heuristic, CROWN should still provide some tightening
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_soundness() {
    // Soundness test: concrete outputs must be within CROWN bounds
    let w1 = arr2(&[[2.0, -1.0], [1.0, 3.0]]);
    let b1 = arr1(&[1.0, -1.0]);
    let w2 = arr2(&[[1.0, -1.0]]);
    let b2 = arr1(&[0.5]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap(),
    ));

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let crown_output = network.propagate_crown(&input_bounds).unwrap();

    // Test several concrete inputs
    let test_inputs = [
        arr1(&[-1.0, -1.0]),
        arr1(&[1.0, 1.0]),
        arr1(&[0.0, 0.0]),
        arr1(&[-1.0, 1.0]),
        arr1(&[1.0, -1.0]),
        arr1(&[0.5, -0.5]),
        arr1(&[-0.5, 0.5]),
    ];

    for x in &test_inputs {
        // Compute concrete output: Linear -> ReLU -> Linear
        let z1 = w1.dot(x) + &b1;
        let a1 = z1.mapv(|v| v.max(0.0));
        let y = w2.dot(&a1) + &b2;

        // Verify soundness
        assert!(
            y[0] >= crown_output.lower()[[0]] - 1e-5,
            "Soundness violation: concrete {} < CROWN lower {} for input {:?}",
            y[0],
            crown_output.lower()[[0]],
            x
        );
        assert!(
            y[0] <= crown_output.upper()[[0]] + 1e-5,
            "Soundness violation: concrete {} > CROWN upper {} for input {:?}",
            y[0],
            crown_output.upper()[[0]],
            x
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_empty_network() {
    let network = Network::new();

    let input =
        BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[2.0, 3.0]).into_dyn()).unwrap();

    let output = network.propagate_crown(&input).unwrap();

    // Empty network: output = input
    assert!((output.lower()[[0]] - 0.0).abs() < 1e-5);
    assert!((output.upper()[[0]] - 2.0).abs() < 1e-5);
    assert!((output.lower()[[1]] - 1.0).abs() < 1e-5);
    assert!((output.upper()[[1]] - 3.0).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_soundness_multiple_networks() {
    // Test that CROWN bounds are SOUND for various networks
    // Note: CROWN may not always be tighter than IBP for all networks
    // due to the linear relaxation approximation, but it MUST be sound.

    // Network definitions with their weight matrices for concrete evaluation
    struct TestNetwork {
        network: Network,
        w1: Array2<f32>,
        b1: Option<Array1<f32>>,
        w2: Array2<f32>,
        b2: Option<Array1<f32>>,
        w3: Option<Array2<f32>>,
        b3: Option<Array1<f32>>,
    }

    let networks = [
        // Network 1: Simple MLP (Linear -> ReLU -> Linear)
        TestNetwork {
            network: {
                let mut net = Network::new();
                net.add_layer(Layer::Linear(
                    LinearLayer::new(arr2(&[[1.0, 1.0], [-1.0, 1.0]]), Some(arr1(&[0.0, 0.0])))
                        .unwrap(),
                ));
                net.add_layer(Layer::ReLU(ReLULayer));
                net.add_layer(Layer::Linear(
                    LinearLayer::new(arr2(&[[1.0, -1.0]]), None).unwrap(),
                ));
                net
            },
            w1: arr2(&[[1.0, 1.0], [-1.0, 1.0]]),
            b1: Some(arr1(&[0.0, 0.0])),
            w2: arr2(&[[1.0, -1.0]]),
            b2: None,
            w3: None,
            b3: None,
        },
        // Network 2: MLP with negative weights (Linear -> ReLU -> Linear)
        TestNetwork {
            network: {
                let mut net = Network::new();
                net.add_layer(Layer::Linear(
                    LinearLayer::new(arr2(&[[-0.5, 0.5], [0.5, 0.5]]), Some(arr1(&[0.1, -0.1])))
                        .unwrap(),
                ));
                net.add_layer(Layer::ReLU(ReLULayer));
                net.add_layer(Layer::Linear(
                    LinearLayer::new(arr2(&[[0.5, 0.5]]), Some(arr1(&[0.0]))).unwrap(),
                ));
                net
            },
            w1: arr2(&[[-0.5, 0.5], [0.5, 0.5]]),
            b1: Some(arr1(&[0.1, -0.1])),
            w2: arr2(&[[0.5, 0.5]]),
            b2: Some(arr1(&[0.0])),
            w3: None,
            b3: None,
        },
    ];

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Test inputs within bounds
    let test_inputs = [
        arr1(&[-1.0, -1.0]),
        arr1(&[1.0, 1.0]),
        arr1(&[0.0, 0.0]),
        arr1(&[-1.0, 1.0]),
        arr1(&[1.0, -1.0]),
        arr1(&[0.5, -0.5]),
        arr1(&[-0.5, 0.5]),
    ];

    for (net_idx, test_net) in networks.iter().enumerate() {
        let crown_output = test_net.network.propagate_crown(&input_bounds).unwrap();

        for x in &test_inputs {
            // Compute concrete output
            let z1 = test_net.w1.dot(x) + test_net.b1.as_ref().unwrap_or(&arr1(&[0.0, 0.0]));
            let a1 = z1.mapv(|v| v.max(0.0));
            let z2 = test_net.w2.dot(&a1) + test_net.b2.as_ref().unwrap_or(&arr1(&[0.0]));
            let y = if test_net.w3.is_some() {
                let a2 = z2.mapv(|v| v.max(0.0));
                let z3 = test_net.w3.as_ref().unwrap().dot(&a2)
                    + test_net.b3.as_ref().unwrap_or(&arr1(&[0.0]));
                z3
            } else {
                z2
            };

            // Verify soundness
            assert!(
                y[0] >= crown_output.lower()[[0]] - 1e-5,
                "Network {}: CROWN lower {} not sound for input {:?}, actual {}",
                net_idx,
                crown_output.lower()[[0]],
                x,
                y[0]
            );
            assert!(
                y[0] <= crown_output.upper()[[0]] + 1e-5,
                "Network {}: CROWN upper {} not sound for input {:?}, actual {}",
                net_idx,
                crown_output.upper()[[0]],
                x,
                y[0]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_identity() {
    let bounds = LinearBounds::identity(3);
    assert_eq!(bounds.num_outputs(), 3);
    assert_eq!(bounds.num_inputs(), 3);

    // Identity: A = I, b = 0
    assert!((bounds.lower_a[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((bounds.lower_a[[0, 1]] - 0.0).abs() < 1e-6);
    assert!((bounds.lower_a[[1, 1]] - 1.0).abs() < 1e-6);
    assert!((bounds.lower_b[[0]] - 0.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_concretize() {
    // Test concretization with simple linear bounds
    // lower: y >= 2*x + 1, upper: y <= 3*x + 2
    let bounds =
        LinearBounds::new(arr2(&[[2.0]]), arr1(&[1.0]), arr2(&[[3.0]]), arr1(&[2.0])).unwrap();

    // Input x in [0, 1]
    let input = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let output = bounds.concretize(&input);

    // lower = 2*0 + 1 = 1 (use lower input for positive coeff)
    // upper = 3*1 + 2 = 5 (use upper input for positive coeff)
    assert!(
        (output.lower()[[0]] - 1.0).abs() < 1e-5,
        "lower = {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 5.0).abs() < 1e-5,
        "upper = {}",
        output.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_bounds_concretize_negative_coeff() {
    // Test concretization with negative coefficients
    // lower: y >= -2*x + 1, upper: y <= -1*x + 2
    let bounds =
        LinearBounds::new(arr2(&[[-2.0]]), arr1(&[1.0]), arr2(&[[-1.0]]), arr1(&[2.0])).unwrap();

    // Input x in [0, 1]
    let input = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let output = bounds.concretize(&input);

    // lower = -2*1 + 1 = -1 (use upper input for negative coeff to minimize)
    // upper = -1*0 + 2 = 2 (use lower input for negative coeff to maximize)
    assert!(
        (output.lower()[[0]] - (-1.0)).abs() < 1e-5,
        "lower = {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 2.0).abs() < 1e-5,
        "upper = {}",
        output.upper()[[0]]
    );
}

/// Regression test for issue #1607: ReLU relaxation divide-by-zero when u ≈ l.
///
/// When pre-activation bounds have near-zero width (u ≈ l), the ReLU relaxation
/// slope/intercept calculation had a division by (u - l) which could produce
/// inf or NaN. The fix clamps the denominator to RELU_RELAX_MIN_WIDTH.
#[ntest::timeout(10000)]
#[test]
fn test_relu_crown_near_zero_width_interval_1607() {
    // Create pre-activation bounds with near-zero width in crossing region
    let epsilon = 1e-12_f32;
    let pre_lower = ArrayD::from_elem(IxDyn(&[2]), -epsilon);
    let pre_upper = ArrayD::from_elem(IxDyn(&[2]), epsilon);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Identity linear bounds for testing
    let linear_bounds = LinearBounds::identity(2);
    let relu = ReLULayer::new();

    // This should NOT panic or produce NaN/inf
    let result = relu.propagate_linear_with_bounds(&linear_bounds, &pre_activation);

    // Verify we get valid bounds (no NaN/inf)
    let result = result.expect("propagate_linear_with_bounds should not fail for near-zero width");
    assert!(
        result.lower_a.iter().all(|&x| x.is_finite()),
        "lower_a contains non-finite values"
    );
    assert!(
        result.upper_a.iter().all(|&x| x.is_finite()),
        "upper_a contains non-finite values"
    );
    assert!(
        result.lower_b.iter().all(|&x| x.is_finite()),
        "lower_b contains non-finite values"
    );
    assert!(
        result.upper_b.iter().all(|&x| x.is_finite()),
        "upper_b contains non-finite values"
    );

    // Test soundness: verify relaxation is valid at a few points
    let test_points = [-epsilon, 0.0_f32, epsilon];
    for &x in &test_points {
        let relu_x = x.max(0.0);
        // For identity bounds, output[i] = input[i].
        // Check both lower and upper soundness using the full row.
        let input_vec = Array1::from_vec(vec![x, x]);
        for i in 0..2 {
            let upper_bound = result.upper_a.row(i).dot(&input_vec) + result.upper_b[i];
            let lower_bound = result.lower_a.row(i).dot(&input_vec) + result.lower_b[i];
            assert!(
                upper_bound >= relu_x - 1e-6,
                "Upper bound violated at x={}: bound={} < ReLU={}",
                x,
                upper_bound,
                relu_x
            );
            assert!(
                lower_bound <= relu_x + 1e-6,
                "Lower bound violated at x={}: bound={} > ReLU={}",
                x,
                lower_bound,
                relu_x
            );
        }
    }
}
