// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Reference comparisons against Auto-LiRPA.

use super::*;
use ndarray::{arr1, arr2};

// ==================== Auto-LiRPA Comparison Benchmarks ====================
//
// These tests compare ny bounds against reference values computed
// by Auto-LiRPA (Python). The reference values are from:
// benchmarks/auto_lirpa_reference.py
//
// Performance timing comparisons live in:
// crates/ny-propagate/examples/benchmark_autolirpa_performance.rs
//
// Run with: cargo test -p ny-propagate benchmark_auto_lirpa -- --nocapture

#[ntest::timeout(10000)]
#[test]
fn benchmark_auto_lirpa_toy_model() {
    // Toy model from Auto-LiRPA examples/simple/toy.py:
    // - Linear: 2 -> 2 (w1=[[1, -1], [2, -1]], no bias)
    // - ReLU
    // - Linear: 2 -> 1 (w2=[[1, -1]], no bias)
    //
    // Input bounds: lower=[-1, -2], upper=[2, 1]
    //
    // Auto-LiRPA reference results (PyTorch 2.9.1):
    // - IBP: lower=-6.0, upper=4.0
    // - CROWN: lower=-3.0, upper=3.0
    // - alpha-CROWN: lower=-3.0, upper=3.0

    let w1 = arr2(&[[1.0f32, -1.0], [2.0, -1.0]]);
    let w2 = arr2(&[[1.0f32, -1.0]]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1.clone(), None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2.clone(), None).unwrap()));

    let input_bounds = BoundedTensor::new(
        arr1(&[-1.0f32, -2.0]).into_dyn(),
        arr1(&[2.0f32, 1.0]).into_dyn(),
    )
    .unwrap();

    // Auto-LiRPA reference values
    let ref_ibp_lower = -6.0f32;
    let ref_ibp_upper = 4.0f32;
    let ref_crown_lower = -3.0f32;
    let ref_crown_upper = 3.0f32;

    // IBP
    let ibp_result = network.propagate_ibp(&input_bounds).unwrap();
    let ibp_lower = ibp_result.lower()[[0]];
    let ibp_upper = ibp_result.upper()[[0]];

    println!("\n=== Auto-LiRPA Toy Model Comparison ===");
    println!("IBP:");
    println!("  ny: lower={:.6}, upper={:.6}", ibp_lower, ibp_upper);
    println!(
        "  Auto-LiRPA:  lower={:.6}, upper={:.6}",
        ref_ibp_lower, ref_ibp_upper
    );

    assert!(
        (ibp_lower - ref_ibp_lower).abs() < 1e-4,
        "IBP lower mismatch: ny={}, Auto-LiRPA={}",
        ibp_lower,
        ref_ibp_lower
    );
    assert!(
        (ibp_upper - ref_ibp_upper).abs() < 1e-4,
        "IBP upper mismatch: ny={}, Auto-LiRPA={}",
        ibp_upper,
        ref_ibp_upper
    );

    // CROWN
    let crown_result = network.propagate_crown(&input_bounds).unwrap();
    let crown_lower = crown_result.lower()[[0]];
    let crown_upper = crown_result.upper()[[0]];

    println!("CROWN:");
    println!("  ny: lower={:.6}, upper={:.6}", crown_lower, crown_upper);
    println!(
        "  Auto-LiRPA:  lower={:.6}, upper={:.6}",
        ref_crown_lower, ref_crown_upper
    );

    assert!(
        (crown_lower - ref_crown_lower).abs() < 1e-4,
        "CROWN lower mismatch: ny={}, Auto-LiRPA={}",
        crown_lower,
        ref_crown_lower
    );
    assert!(
        (crown_upper - ref_crown_upper).abs() < 1e-4,
        "CROWN upper mismatch: ny={}, Auto-LiRPA={}",
        crown_upper,
        ref_crown_upper
    );

    // alpha-CROWN
    let alpha_result = network.propagate_alpha_crown(&input_bounds).unwrap();
    let alpha_lower = alpha_result.lower()[[0]];
    let alpha_upper = alpha_result.upper()[[0]];

    println!("alpha-CROWN:");
    println!("  ny: lower={:.6}, upper={:.6}", alpha_lower, alpha_upper);
    println!(
        "  Auto-LiRPA:  lower={:.6}, upper={:.6}",
        ref_crown_lower, ref_crown_upper
    );

    // alpha-CROWN should be at least as tight as CROWN
    assert!(
        alpha_lower >= crown_lower - 1e-5,
        "alpha-CROWN lower should be >= CROWN lower"
    );
    assert!(
        alpha_upper <= crown_upper + 1e-5,
        "alpha-CROWN upper should be <= CROWN upper"
    );

    // Verify soundness with concrete points
    println!("Soundness check:");
    let test_inputs = [
        arr1(&[-1.0f32, -2.0]),
        arr1(&[2.0f32, 1.0]),
        arr1(&[0.5f32, -0.5]),
    ];

    for x in &test_inputs {
        let z1 = w1.dot(x);
        let a1 = z1.mapv(|v| v.max(0.0));
        let y = w2.dot(&a1);
        let output = y[0];

        println!("  x={:?} -> y={:.4}", x.to_vec(), output);

        assert!(
            output >= ibp_lower - 1e-5 && output <= ibp_upper + 1e-5,
            "Soundness violation for IBP"
        );
        assert!(
            output >= crown_lower - 1e-5 && output <= crown_upper + 1e-5,
            "Soundness violation for CROWN"
        );
        assert!(
            output >= alpha_lower - 1e-5 && output <= alpha_upper + 1e-5,
            "Soundness violation for alpha-CROWN"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
// Justification: Reference constants from Auto-LiRPA torch output with full f64 precision;
// truncating would reduce comparison accuracy against the reference implementation.
#[allow(clippy::excessive_precision)]
fn benchmark_auto_lirpa_deep_model() {
    // Deep model: 3-layer MLP with specific weights from Auto-LiRPA
    // Input: 3 -> 4 -> 4 -> 2
    // Weights are from torch.manual_seed(42) with randn * 0.5
    //
    // Auto-LiRPA reference results (epsilon=0.1 around [0.5, 0.5, 0.5]):
    // - IBP: lower=[0.00688, -0.01418], upper=[0.01246, 0.00072]
    // - CROWN: lower=[0.00821, -0.01064], upper=[0.01113, -0.00283]

    // Exact weights from Auto-LiRPA benchmark
    let fc1_weight = arr2(&[
        [0.16834518f32, 0.06440470, 0.11723118],
        [0.11516652, -0.56142819, -0.09316415],
        [1.10410070, -0.31899852, 0.23082861],
        [0.13367544, 0.26745233, 0.40467861],
    ]);
    let fc1_bias = arr1(&[0.11102903f32, -0.16897990, -0.09889599, 0.09579718]);

    let fc2_weight = arr2(&[
        [-0.69233710f32, -0.43561807, -0.11168297, 0.85868055],
        [0.15943986, -0.21225949, 0.15286016, -0.38729626],
        [-0.77878612, 0.49781805, -0.43989292, -0.30057147],
        [-0.63707572, 1.06139255, -0.61732668, -0.24395694],
    ]);
    let fc2_bias = arr1(&[0.02815196f32, 0.00561635, 0.05227160, -0.02383569]);

    let fc3_weight = arr2(&[
        [-0.02495167f32, 0.26316848, -0.00424941, 0.36453030],
        [0.06657098, 0.43198884, -0.50783736, -0.44437426],
    ]);
    let fc3_bias = arr1(&[0.01497797f32, -0.02088939]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(fc1_weight.clone(), Some(fc1_bias.clone())).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(fc2_weight.clone(), Some(fc2_bias.clone())).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(fc3_weight.clone(), Some(fc3_bias.clone())).unwrap(),
    ));

    // Input bounds: epsilon=0.1 around [0.5, 0.5, 0.5]
    let input_bounds = BoundedTensor::new(
        arr1(&[0.4f32, 0.4, 0.4]).into_dyn(),
        arr1(&[0.6f32, 0.6, 0.6]).into_dyn(),
    )
    .unwrap();

    // Auto-LiRPA reference values
    let ref_ibp_lower = arr1(&[0.00687957f32, -0.01418082]);
    let ref_ibp_upper = arr1(&[0.01246351f32, 0.00071711]);
    let ref_crown_lower = arr1(&[0.00820859f32, -0.01063502]);
    let ref_crown_upper = arr1(&[0.01113450f32, -0.00282869]);

    // IBP
    let ibp_result = network.propagate_ibp(&input_bounds).unwrap();

    println!("\n=== Auto-LiRPA Deep Model Comparison ===");
    println!("IBP:");
    println!(
        "  ny: lower=[{:.6}, {:.6}], upper=[{:.6}, {:.6}]",
        ibp_result.lower()[[0]],
        ibp_result.lower()[[1]],
        ibp_result.upper()[[0]],
        ibp_result.upper()[[1]]
    );
    println!(
        "  Auto-LiRPA:  lower=[{:.6}, {:.6}], upper=[{:.6}, {:.6}]",
        ref_ibp_lower[0], ref_ibp_lower[1], ref_ibp_upper[0], ref_ibp_upper[1]
    );

    // Check IBP bounds (allow small tolerance for floating point)
    for i in 0..2 {
        assert!(
            (ibp_result.lower()[[i]] - ref_ibp_lower[i]).abs() < 1e-4,
            "IBP lower[{}] mismatch: ny={}, Auto-LiRPA={}",
            i,
            ibp_result.lower()[[i]],
            ref_ibp_lower[i]
        );
        assert!(
            (ibp_result.upper()[[i]] - ref_ibp_upper[i]).abs() < 1e-4,
            "IBP upper[{}] mismatch: ny={}, Auto-LiRPA={}",
            i,
            ibp_result.upper()[[i]],
            ref_ibp_upper[i]
        );
    }

    // CROWN
    let crown_result = network.propagate_crown(&input_bounds).unwrap();

    println!("CROWN:");
    println!(
        "  ny: lower=[{:.6}, {:.6}], upper=[{:.6}, {:.6}]",
        crown_result.lower()[[0]],
        crown_result.lower()[[1]],
        crown_result.upper()[[0]],
        crown_result.upper()[[1]]
    );
    println!(
        "  Auto-LiRPA:  lower=[{:.6}, {:.6}], upper=[{:.6}, {:.6}]",
        ref_crown_lower[0], ref_crown_lower[1], ref_crown_upper[0], ref_crown_upper[1]
    );

    // CROWN bounds should be close to reference
    for i in 0..2 {
        assert!(
            (crown_result.lower()[[i]] - ref_crown_lower[i]).abs() < 1e-4,
            "CROWN lower[{}] mismatch: ny={}, Auto-LiRPA={}",
            i,
            crown_result.lower()[[i]],
            ref_crown_lower[i]
        );
        assert!(
            (crown_result.upper()[[i]] - ref_crown_upper[i]).abs() < 1e-4,
            "CROWN upper[{}] mismatch: ny={}, Auto-LiRPA={}",
            i,
            crown_result.upper()[[i]],
            ref_crown_upper[i]
        );
    }

    // Verify CROWN is tighter than IBP
    let ibp_width_0 = ibp_result.upper()[[0]] - ibp_result.lower()[[0]];
    let ibp_width_1 = ibp_result.upper()[[1]] - ibp_result.lower()[[1]];
    let crown_width_0 = crown_result.upper()[[0]] - crown_result.lower()[[0]];
    let crown_width_1 = crown_result.upper()[[1]] - crown_result.lower()[[1]];

    println!("Width comparison:");
    println!("  IBP width:   [{:.6}, {:.6}]", ibp_width_0, ibp_width_1);
    println!(
        "  CROWN width: [{:.6}, {:.6}]",
        crown_width_0, crown_width_1
    );
    println!(
        "  CROWN tightening: [{:.1}%, {:.1}%]",
        100.0 * (1.0 - crown_width_0 / ibp_width_0),
        100.0 * (1.0 - crown_width_1 / ibp_width_1)
    );

    assert!(
        crown_width_0 <= ibp_width_0 + 1e-5,
        "CROWN should be at least as tight as IBP for output 0"
    );
    assert!(
        crown_width_1 <= ibp_width_1 + 1e-5,
        "CROWN should be at least as tight as IBP for output 1"
    );

    // alpha-CROWN
    let alpha_result = network.propagate_alpha_crown(&input_bounds).unwrap();

    println!("alpha-CROWN:");
    println!(
        "  ny: lower=[{:.6}, {:.6}], upper=[{:.6}, {:.6}]",
        alpha_result.lower()[[0]],
        alpha_result.lower()[[1]],
        alpha_result.upper()[[0]],
        alpha_result.upper()[[1]]
    );

    let alpha_width_0 = alpha_result.upper()[[0]] - alpha_result.lower()[[0]];
    let alpha_width_1 = alpha_result.upper()[[1]] - alpha_result.lower()[[1]];

    println!(
        "  alpha-CROWN width: [{:.6}, {:.6}]",
        alpha_width_0, alpha_width_1
    );

    // alpha-CROWN should be at least as tight as CROWN
    assert!(
        alpha_width_0 <= crown_width_0 + 1e-5,
        "alpha-CROWN should be at least as tight as CROWN for output 0"
    );
    assert!(
        alpha_width_1 <= crown_width_1 + 1e-5,
        "alpha-CROWN should be at least as tight as CROWN for output 1"
    );

    // Verify soundness by checking center point
    let center = arr1(&[0.5f32, 0.5, 0.5]);
    let z1 = fc1_weight.dot(&center) + &fc1_bias;
    let a1 = z1.mapv(|v| v.max(0.0));
    let z2 = fc2_weight.dot(&a1) + &fc2_bias;
    let a2 = z2.mapv(|v| v.max(0.0));
    let output = fc3_weight.dot(&a2) + &fc3_bias;

    println!("Soundness check at center [0.5, 0.5, 0.5]:");
    println!("  Concrete output: [{:.6}, {:.6}]", output[0], output[1]);

    for i in 0..2 {
        assert!(
            output[i] >= ibp_result.lower()[[i]] - 1e-5
                && output[i] <= ibp_result.upper()[[i]] + 1e-5,
            "Soundness violation for IBP at output {}",
            i
        );
        assert!(
            output[i] >= crown_result.lower()[[i]] - 1e-5
                && output[i] <= crown_result.upper()[[i]] + 1e-5,
            "Soundness violation for CROWN at output {}",
            i
        );
        assert!(
            output[i] >= alpha_result.lower()[[i]] - 1e-5
                && output[i] <= alpha_result.upper()[[i]] + 1e-5,
            "Soundness violation for alpha-CROWN at output {}",
            i
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_vs_autolirpa_tiny() {
    // Exact network from minimal test:
    // Linear(2->2) -> ReLU -> Linear(2->2)
    // W1 = [[1, 1], [1, -1]], b1 = [-0.5, 0]
    // W2 = [[1, 0], [0, 1]], b2 = [0, 0]

    let w1 = arr2(&[[1.0, 1.0], [1.0, -1.0]]);
    let b1 = arr1(&[-0.5, 0.0]);
    let w2 = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let b2 = arr1(&[0.0, 0.0]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Expected results (from Auto-LiRPA):
    // IBP: lower=[0, 0], upper=[1.5, 1]
    // Raw CROWN: lower=[-0.5, 0], upper=[1.5, 1]
    // After IBP intersection (#2990): lower=[max(-0.5, 0), max(0, 0)] = [0, 0]
    // The intersection tightens CROWN lower[0] from -0.5 to 0.0 because IBP is tighter.

    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    println!(
        "IBP: lower={:?}, upper={:?}",
        ibp_output.lower().as_slice().unwrap(),
        ibp_output.upper().as_slice().unwrap()
    );
    println!(
        "CROWN: lower={:?}, upper={:?}",
        crown_output.lower().as_slice().unwrap(),
        crown_output.upper().as_slice().unwrap()
    );

    // Check IBP
    assert!(
        (ibp_output.lower()[[0]] - 0.0).abs() < 1e-5,
        "IBP lower[0] should be 0, got {}",
        ibp_output.lower()[[0]]
    );
    assert!(
        (ibp_output.lower()[[1]] - 0.0).abs() < 1e-5,
        "IBP lower[1] should be 0, got {}",
        ibp_output.lower()[[1]]
    );
    assert!(
        (ibp_output.upper()[[0]] - 1.5).abs() < 1e-5,
        "IBP upper[0] should be 1.5, got {}",
        ibp_output.upper()[[0]]
    );
    assert!(
        (ibp_output.upper()[[1]] - 1.0).abs() < 1e-5,
        "IBP upper[1] should be 1.0, got {}",
        ibp_output.upper()[[1]]
    );

    // Check CROWN after IBP intersection (#2990).
    // Raw CROWN lower[0] = -0.5, but IBP lower[0] = 0.0, so intersection gives 0.0.
    assert!(
        (crown_output.lower()[[0]] - 0.0).abs() < 1e-5,
        "CROWN lower[0] should be 0.0 after IBP intersection (#2990), got {}",
        crown_output.lower()[[0]]
    );
    assert!(
        (crown_output.lower()[[1]] - 0.0).abs() < 1e-5,
        "CROWN lower[1] should be 0, got {}",
        crown_output.lower()[[1]]
    );
    assert!(
        (crown_output.upper()[[0]] - 1.5).abs() < 1e-5,
        "CROWN upper[0] should be 1.5, got {}",
        crown_output.upper()[[0]]
    );
    assert!(
        (crown_output.upper()[[1]] - 1.0).abs() < 1e-5,
        "CROWN upper[1] should be 1.0, got {}",
        crown_output.upper()[[1]]
    );
}
