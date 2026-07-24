// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GraphNetwork BatchNorm IBP/CROWN tests.
use crate::*;
use ndarray::{arr1, Array2, ArrayD, IxDyn};

#[ntest::timeout(10000)]
#[test]
fn test_batchnorm_ibp_positive_scale() {
    // BatchNorm with 2 channels, positive scale
    // scale = [2.0, 3.0], bias = [1.0, -1.0]
    let scale = arr1(&[2.0_f32, 3.0]).into_dyn();
    let bias = arr1(&[1.0_f32, -1.0]).into_dyn();
    let bn = BatchNormLayer::from_scale_bias(scale, bias).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("bn", Layer::BatchNorm(bn)));
    graph.set_output("bn");

    // Input shape (C, H, W) = (2, 2, 2), values in [-1, 1]
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![-1.0; 8]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), vec![1.0; 8]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // Channel 0: y = 2*x + 1, x in [-1, 1] => y in [-1, 3]
    // Channel 1: y = 3*x - 1, x in [-1, 1] => y in [-4, 2]
    for i in 0..4 {
        assert!(
            (output.lower()[[0, i / 2, i % 2]] - (-1.0)).abs() < 1e-5,
            "Channel 0 lower should be -1"
        );
        assert!(
            (output.upper()[[0, i / 2, i % 2]] - 3.0).abs() < 1e-5,
            "Channel 0 upper should be 3"
        );
    }
    for i in 0..4 {
        assert!(
            (output.lower()[[1, i / 2, i % 2]] - (-4.0)).abs() < 1e-5,
            "Channel 1 lower should be -4"
        );
        assert!(
            (output.upper()[[1, i / 2, i % 2]] - 2.0).abs() < 1e-5,
            "Channel 1 upper should be 2"
        );
    }
}

/// Test BatchNorm IBP with negative scale (should swap bounds)

#[ntest::timeout(10000)]
#[test]
fn test_batchnorm_ibp_negative_scale() {
    // BatchNorm with negative scale
    // scale = [-2.0, 1.0], bias = [0.0, 0.0]
    let scale = arr1(&[-2.0_f32, 1.0]).into_dyn();
    let bias = arr1(&[0.0_f32, 0.0]).into_dyn();
    let bn = BatchNormLayer::from_scale_bias(scale, bias).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("bn", Layer::BatchNorm(bn)));
    graph.set_output("bn");

    // Input shape (2, 2): For 2D, BatchNorm uses channel at index 1
    // So column 0 is channel 0, column 1 is channel 1
    // Channel 0: values in [1.0, 2.0], Channel 1: values in [3.0, 5.0]
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 3.0, 1.0, 3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2.0, 5.0, 2.0, 5.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let output = graph.propagate_ibp(&input).unwrap();

    // Channel 0 (negative scale): y = -2*x, x in [1, 2] => y in [-4, -2]
    // Need to swap because scale is negative
    assert!(
        (output.lower()[[0, 0]] - (-4.0)).abs() < 1e-5,
        "Negative scale lower: expected -4, got {}",
        output.lower()[[0, 0]]
    );
    assert!(
        (output.upper()[[0, 0]] - (-2.0)).abs() < 1e-5,
        "Negative scale upper: expected -2, got {}",
        output.upper()[[0, 0]]
    );

    // Channel 1 (positive scale): y = 1*x, x in [3, 5] => y in [3, 5]
    assert!(
        (output.lower()[[0, 1]] - 3.0).abs() < 1e-5,
        "Positive scale lower: expected 3, got {}",
        output.lower()[[0, 1]]
    );
    assert!(
        (output.upper()[[0, 1]] - 5.0).abs() < 1e-5,
        "Positive scale upper: expected 5, got {}",
        output.upper()[[0, 1]]
    );
}

/// Test BatchNorm CROWN backward propagation in GraphNetwork

#[ntest::timeout(10000)]
#[test]
fn test_batchnorm_crown_backward() {
    // Linear -> Reshape -> BatchNorm -> Reshape -> Linear network
    // This tests that CROWN backward propagation through BatchNorm works
    let mut graph = GraphNetwork::new();

    // First linear: 4 inputs -> 4 outputs
    let w1 = Array2::eye(4);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));

    // Reshape to (C=2, L=2) for BatchNorm
    let reshape1 = ReshapeLayer::new(vec![2, 2]);
    graph.add_node(GraphNode::new(
        "reshape1",
        Layer::Reshape(reshape1),
        vec!["linear1".to_string()],
    ));

    // BatchNorm: scale = [2, 0.5], bias = [1, -1]
    let scale = arr1(&[2.0_f32, 0.5]).into_dyn();
    let bias = arr1(&[1.0_f32, -1.0]).into_dyn();
    let bn = BatchNormLayer::from_scale_bias(scale, bias).unwrap();
    graph.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bn),
        vec!["reshape1".to_string()],
    ));

    // Reshape to flatten for final linear
    let reshape2 = ReshapeLayer::new(vec![4]);
    graph.add_node(GraphNode::new(
        "reshape2",
        Layer::Reshape(reshape2),
        vec!["bn".to_string()],
    ));

    // Final linear: 4 inputs -> 1 output (sum all)
    let w2 = Array2::ones((1, 4));
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["reshape2".to_string()],
    ));

    graph.set_output("linear2");

    // Input is flat 4D
    let lower = arr1(&[-1.0_f32; 4]).into_dyn();
    let upper = arr1(&[1.0_f32; 4]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Get CROWN bounds
    let crown_output = graph.propagate_crown(&input).unwrap();
    let ibp_output = graph.propagate_ibp(&input).unwrap();

    // CROWN should be at least as tight as IBP
    assert!(
        crown_output.lower()[[0]] >= ibp_output.lower()[[0]] - 1e-5,
        "CROWN lower should be >= IBP lower"
    );
    assert!(
        crown_output.upper()[[0]] <= ibp_output.upper()[[0]] + 1e-5,
        "CROWN upper should be <= IBP upper"
    );

    // Verify soundness with concrete inputs
    let test_inputs = vec![
        arr1(&[-1.0_f32, -1.0, -1.0, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0, 1.0, 1.0]).into_dyn(),
        arr1(&[0.0_f32, 0.0, 0.0, 0.0]).into_dyn(),
        arr1(&[-0.5_f32, 0.5, -0.5, 0.5]).into_dyn(),
    ];

    for test_input in &test_inputs {
        let concrete = BoundedTensor::concrete(test_input.clone()).unwrap();
        let concrete_output = graph.propagate_ibp(&concrete).unwrap();

        assert!(
            concrete_output.lower()[[0]] >= crown_output.lower()[[0]] - 1e-5,
            "Soundness: concrete {} < CROWN lower {}",
            concrete_output.lower()[[0]],
            crown_output.lower()[[0]]
        );
        assert!(
            concrete_output.upper()[[0]] <= crown_output.upper()[[0]] + 1e-5,
            "Soundness: concrete {} > CROWN upper {}",
            concrete_output.upper()[[0]],
            crown_output.upper()[[0]]
        );
    }
}

/// Test BatchNorm CROWN soundness with ReLU

#[ntest::timeout(10000)]
#[test]
fn test_batchnorm_crown_with_relu_soundness() {
    // Simple: Reshape -> BatchNorm -> ReLU -> Reshape -> Linear
    // Tests interaction between BatchNorm and ReLU in CROWN
    let mut graph = GraphNetwork::new();

    // Reshape to (C=2, L=2)
    let reshape1 = ReshapeLayer::new(vec![2, 2]);
    graph.add_node(GraphNode::from_input("reshape1", Layer::Reshape(reshape1)));

    // BatchNorm: scale = [1.0, 1.0], bias = [0.0, 0.0] (identity)
    let scale = arr1(&[1.0_f32, 1.0]).into_dyn();
    let bias = arr1(&[0.0_f32, 0.0]).into_dyn();
    let bn = BatchNormLayer::from_scale_bias(scale, bias).unwrap();
    graph.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bn),
        vec!["reshape1".to_string()],
    ));

    // ReLU
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["bn".to_string()],
    ));

    // Reshape to flatten
    let reshape2 = ReshapeLayer::new(vec![4]);
    graph.add_node(GraphNode::new(
        "reshape2",
        Layer::Reshape(reshape2),
        vec!["relu".to_string()],
    ));

    // Final linear: sum all
    let w2 = Array2::ones((1, 4));
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["reshape2".to_string()],
    ));

    graph.set_output("linear2");

    // Input bounds
    let lower = arr1(&[-1.0_f32, -1.0, -1.0, -1.0]).into_dyn();
    let upper = arr1(&[1.0_f32, 1.0, 1.0, 1.0]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let crown_output = graph.propagate_crown(&input).unwrap();
    let ibp_output = graph.propagate_ibp(&input).unwrap();

    // CROWN bounds should be valid
    assert!(
        crown_output.lower()[[0]] <= crown_output.upper()[[0]],
        "CROWN bounds must be valid"
    );

    // Test soundness with corner inputs
    let test_inputs = vec![
        arr1(&[-1.0_f32, -1.0, -1.0, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0, 1.0, 1.0]).into_dyn(),
        arr1(&[0.0_f32, 0.0, 0.0, 0.0]).into_dyn(),
        arr1(&[-1.0_f32, 1.0, -1.0, 1.0]).into_dyn(),
        arr1(&[1.0_f32, -1.0, 1.0, -1.0]).into_dyn(),
    ];

    for test_input in &test_inputs {
        let concrete = BoundedTensor::concrete(test_input.clone()).unwrap();
        let concrete_output = graph.propagate_ibp(&concrete).unwrap();

        assert!(
            concrete_output.lower()[[0]] >= crown_output.lower()[[0]] - 1e-4,
            "Soundness violation: concrete {} < CROWN lower {} for input {:?}",
            concrete_output.lower()[[0]],
            crown_output.lower()[[0]],
            test_input
        );
        assert!(
            concrete_output.upper()[[0]] <= crown_output.upper()[[0]] + 1e-4,
            "Soundness violation: concrete {} > CROWN upper {} for input {:?}",
            concrete_output.upper()[[0]],
            crown_output.upper()[[0]],
            test_input
        );
    }

    // CROWN should be at least as tight as IBP (or equal for linear net)
    assert!(
        crown_output.lower()[[0]] >= ibp_output.lower()[[0]] - 1e-4,
        "CROWN lower {} should be >= IBP lower {}",
        crown_output.lower()[[0]],
        ibp_output.lower()[[0]]
    );
}

/// Test BatchNorm CROWN with NCHW (4D) input

#[ntest::timeout(10000)]
#[test]
fn test_batchnorm_crown_4d_input() {
    // Test with proper 4D CNN-style input
    let mut graph = GraphNetwork::new();

    // BatchNorm: 3 channels
    let scale = arr1(&[1.0_f32, 2.0, 0.5]).into_dyn();
    let bias = arr1(&[0.0_f32, 1.0, -1.0]).into_dyn();
    let bn = BatchNormLayer::from_scale_bias(scale, bias).unwrap();
    graph.add_node(GraphNode::from_input("bn", Layer::BatchNorm(bn)));

    // Flatten
    let reshape = ReshapeLayer::new(vec![12]); // 3*2*2
    graph.add_node(GraphNode::new(
        "flatten",
        Layer::Reshape(reshape),
        vec!["bn".to_string()],
    ));

    // Linear to single output
    let w = Array2::ones((1, 12));
    let linear = LinearLayer::new(w, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear",
        Layer::Linear(linear),
        vec!["flatten".to_string()],
    ));

    graph.set_output("linear");

    // 4D input: (N=1, C=3, H=2, W=2)
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 3, 2, 2]), vec![-1.0; 12]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 3, 2, 2]), vec![1.0; 12]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let crown_output = graph.propagate_crown(&input).unwrap();
    let ibp_output = graph.propagate_ibp(&input).unwrap();

    // CROWN should give valid bounds
    assert!(
        crown_output.lower()[[0]].is_finite(),
        "CROWN lower must be finite"
    );
    assert!(
        crown_output.upper()[[0]].is_finite(),
        "CROWN upper must be finite"
    );
    assert!(
        crown_output.lower()[[0]] <= crown_output.upper()[[0]],
        "CROWN bounds must be valid"
    );

    // CROWN should be at least as tight as IBP
    assert!(
        crown_output.lower()[[0]] >= ibp_output.lower()[[0]] - 1e-4,
        "CROWN lower {} >= IBP lower {}",
        crown_output.lower()[[0]],
        ibp_output.lower()[[0]]
    );
    assert!(
        crown_output.upper()[[0]] <= ibp_output.upper()[[0]] + 1e-4,
        "CROWN upper {} <= IBP upper {}",
        crown_output.upper()[[0]],
        ibp_output.upper()[[0]]
    );
}

/// Test BatchNorm CROWN with negative scale (requires bound swapping)

#[ntest::timeout(10000)]
#[test]
fn test_batchnorm_crown_negative_scale() {
    // Test that negative scale is handled correctly in backward pass
    let mut graph = GraphNetwork::new();

    // Reshape to (C=2, L=2) for BatchNorm
    let reshape1 = ReshapeLayer::new(vec![2, 2]);
    graph.add_node(GraphNode::from_input("reshape1", Layer::Reshape(reshape1)));

    // BatchNorm with one negative scale
    let scale = arr1(&[-1.0_f32, 1.0]).into_dyn();
    let bias = arr1(&[0.0_f32, 0.0]).into_dyn();
    let bn = BatchNormLayer::from_scale_bias(scale, bias).unwrap();
    graph.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bn),
        vec!["reshape1".to_string()],
    ));

    // Reshape to flatten
    let reshape2 = ReshapeLayer::new(vec![4]);
    graph.add_node(GraphNode::new(
        "reshape2",
        Layer::Reshape(reshape2),
        vec!["bn".to_string()],
    ));

    // Linear to single output (sum all)
    let w = Array2::ones((1, 4));
    let linear = LinearLayer::new(w, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear",
        Layer::Linear(linear),
        vec!["reshape2".to_string()],
    ));

    graph.set_output("linear");

    // Input: flat 4D, values in [0, 2]
    let lower = arr1(&[0.0_f32, 0.0, 0.0, 0.0]).into_dyn();
    let upper = arr1(&[2.0_f32, 2.0, 2.0, 2.0]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let crown_output = graph.propagate_crown(&input).unwrap();

    // After reshape: (2, 2) - Channel 0 has first 2 values, Channel 1 has last 2
    // Channel 0: y = -1 * x, x in [0, 2] => y in [-2, 0]
    // Channel 1: y = 1 * x, x in [0, 2] => y in [0, 2]
    // Sum of 4 elements: 2*[-2, 0] + 2*[0, 2] = [-4, 0] + [0, 4] = [-4, 4]

    // Test soundness
    let test_inputs = vec![
        arr1(&[0.0_f32, 0.0, 0.0, 0.0]).into_dyn(),
        arr1(&[2.0_f32, 2.0, 2.0, 2.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0, 1.0, 1.0]).into_dyn(),
        arr1(&[0.0_f32, 0.0, 2.0, 2.0]).into_dyn(),
        arr1(&[2.0_f32, 2.0, 0.0, 0.0]).into_dyn(),
    ];

    for test_input in &test_inputs {
        let concrete = BoundedTensor::concrete(test_input.clone()).unwrap();
        let concrete_output = graph.propagate_ibp(&concrete).unwrap();

        assert!(
            concrete_output.lower()[[0]] >= crown_output.lower()[[0]] - 1e-4,
            "Soundness: concrete {} < CROWN lower {} for input {:?}",
            concrete_output.lower()[[0]],
            crown_output.lower()[[0]],
            test_input
        );
        assert!(
            concrete_output.upper()[[0]] <= crown_output.upper()[[0]] + 1e-4,
            "Soundness: concrete {} > CROWN upper {} for input {:?}",
            concrete_output.upper()[[0]],
            crown_output.upper()[[0]],
            test_input
        );
    }
}

/// Regression: a BatchNorm channel with var + eps == 0 yields an Inf scale
/// (`ny / sqrt(0) = inf`). The CROWN backward then composes that Inf scale with
/// the spec/identity coefficient matrix. Before the fix, a zero coefficient
/// times the Inf scale produced `0 * inf = NaN`, which poisoned the linear
/// bounds and aborted intermediate-bound construction at `BoundedTensor::new`
/// (NaN/Inf rejection) — the failure that scored cifar100_2024 at 0 in graph
/// beta-CROWN.
///
/// After the fix, the BatchNorm CROWN backward uses `safe_mul_for_bounds`
/// (0 * inf = 0): the zero-coefficient term composes to exactly 0, and any
/// genuine Inf coefficient (nonzero coeff * inf scale) is widened to ±inf by
/// concretize's CROWN_COEFF_MAX/Inf short-circuit. Propagation must therefore
/// (a) not panic, (b) never emit NaN, and (c) produce sound bounds
/// (lower <= upper, with any non-finiteness being a conservative ±inf widening
/// rather than a finite-but-wrong value).
#[ntest::timeout(10000)]
#[test]
fn test_batchnorm_zero_variance_inf_scale_no_nan_abort_4xxx() {
    // var = -eps so var + eps == 0 -> sqrt = 0 -> scale = ny / 0 = +inf for ch 0.
    // The other channels have well-behaved variance.
    let eps = 1e-5_f32;
    let ny = arr1(&[1.0_f32, 1.0, 1.0]).into_dyn();
    let beta = arr1(&[0.0_f32, 0.5, -0.5]).into_dyn();
    let mean = arr1(&[0.0_f32, 0.0, 0.0]).into_dyn();
    let var = arr1(&[-eps, 1.0, 2.0]).into_dyn();

    let bn = BatchNormLayer::new(&ny, &beta, &mean, &var, eps)
        .expect("BatchNorm::new should construct (guard only rejects var+eps < 0)");
    // Confirm we actually exercise the degenerate-Inf-scale path.
    assert!(
        !bn.scale[[0]].is_finite(),
        "expected channel-0 scale to be non-finite (inf), got {}",
        bn.scale[[0]]
    );

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("bn", Layer::BatchNorm(bn)));
    // ReLU after BatchNorm (CNN-style Conv/BN/ReLU pattern).
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["bn".to_string()],
    ));
    // Flatten (C=3, H=2, W=2) -> 12 then Linear to a single output.
    let reshape = ReshapeLayer::new(vec![12]);
    graph.add_node(GraphNode::new(
        "flatten",
        Layer::Reshape(reshape),
        vec!["relu".to_string()],
    ));
    let w = Array2::ones((1, 12));
    let linear = LinearLayer::new(w, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear",
        Layer::Linear(linear),
        vec!["flatten".to_string()],
    ));
    graph.set_output("linear");

    // 4D NCHW input (N=1, C=3, H=2, W=2), values in [-1, 1].
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 3, 2, 2]), vec![-1.0; 12]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 3, 2, 2]), vec![1.0; 12]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Sanity check that the helper assertions below see a meaningful output.
    let check_sound = |label: &str, out: &BoundedTensor| {
        for (l, u) in out.lower().iter().zip(out.upper().iter()) {
            // No NaN is ever sound. A non-finite (+/-inf) endpoint is the sound
            // conservative widening we expect from the degenerate channel.
            assert!(
                !l.is_nan(),
                "{label}: lower bound is NaN (unsound abort cause)"
            );
            assert!(
                !u.is_nan(),
                "{label}: upper bound is NaN (unsound abort cause)"
            );
            assert!(l <= u, "{label}: inverted bound lower {l} > upper {u}");
        }
    };

    // Scalar CROWN backward path (crown_scalar.rs).
    let crown = graph
        .propagate_crown(&input)
        .expect("graph CROWN must not abort on Inf-scale BatchNorm");
    check_sound("scalar-CROWN", &crown);

    // Batched CROWN backward path (crown_batched.rs) — the graph beta-CROWN lane.
    let crown_batched = graph
        .propagate_crown_batched(&input)
        .expect("graph batched CROWN must not abort on Inf-scale BatchNorm");
    check_sound("batched-CROWN", &crown_batched);

    // IBP must also remain sound (no NaN) as the fallback path.
    let ibp = graph
        .propagate_ibp(&input)
        .expect("graph IBP must not abort on Inf-scale BatchNorm");
    check_sound("IBP", &ibp);
}
