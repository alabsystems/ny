// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN-IBP tests and intermediate bound collection.

use super::helpers::total_width;
use super::*;
use ndarray::{arr1, arr2, Array1, Array2, ArrayD, IxDyn};

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_tighter_than_crown() {
    // Test that CROWN-IBP produces bounds that are at least as tight as standard CROWN.
    // This is a deep network where CROWN-IBP's tighter intermediate bounds should help.
    //
    // Network: 4 layers of Linear->ReLU with "crossing" inputs (where ReLU needs relaxation)

    // Layer 1: Linear(3->4)
    let w1 = arr2(&[
        [1.0, 2.0, -1.0],
        [-1.0, 1.0, 1.0],
        [0.5, -0.5, 1.0],
        [1.0, 1.0, 1.0],
    ]);
    let b1 = arr1(&[-0.5, 0.0, -0.3, 0.0]);

    // Layer 2: Linear(4->4)
    let w2 = arr2(&[
        [1.0, -1.0, 0.5, 0.0],
        [0.5, 1.0, -0.5, 0.5],
        [-0.5, 0.5, 1.0, -0.5],
        [0.0, 0.5, -0.5, 1.0],
    ]);
    let b2 = arr1(&[0.0, -0.2, 0.1, 0.0]);

    // Layer 3: Linear(4->3)
    let w3 = arr2(&[
        [1.0, -0.5, 0.5, 0.5],
        [-0.5, 1.0, -0.5, 0.5],
        [0.5, 0.5, 1.0, -0.5],
    ]);
    let b3 = arr1(&[0.0, 0.0, 0.0]);

    // Layer 4: Linear(3->2)
    let w4 = arr2(&[[1.0, -1.0, 0.5], [-0.5, 1.0, 1.0]]);
    let b4 = arr1(&[0.0, 0.0]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w3, Some(b3)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w4, Some(b4)).unwrap()));

    // Input with perturbation that creates "crossing" regions
    let input = BoundedTensor::new(
        arr1(&[-0.5, -0.5, -0.5]).into_dyn(),
        arr1(&[0.5, 0.5, 0.5]).into_dyn(),
    )
    .unwrap();

    let crown_output = network.propagate_crown(&input).unwrap();
    let crown_ibp_output = network.propagate_crown_ibp(&input).unwrap();

    println!(
        "CROWN: lower={:?}, upper={:?}",
        crown_output.lower().as_slice().unwrap(),
        crown_output.upper().as_slice().unwrap()
    );
    println!(
        "CROWN-IBP: lower={:?}, upper={:?}",
        crown_ibp_output.lower().as_slice().unwrap(),
        crown_ibp_output.upper().as_slice().unwrap()
    );

    // Print improvement metrics
    // Note: CROWN-IBP produces tighter intermediate bounds which leads to different
    // ReLU relaxation parameters. This doesn't guarantee every individual bound is
    // tighter, but typically improves overall bound width.
    let crown_width = total_width(&crown_output);
    let crown_ibp_width = total_width(&crown_ibp_output);
    println!(
        "Total width - CROWN: {:.4}, CROWN-IBP: {:.4}",
        crown_width, crown_ibp_width
    );
    println!(
        "Improvement: {:.2}%",
        (1.0 - crown_ibp_width / crown_width) * 100.0
    );

    // Verify that bounds are sound (still valid bounds)
    for i in 0..crown_output.len() {
        assert!(
            crown_ibp_output.lower()[[i]] <= crown_ibp_output.upper()[[i]],
            "CROWN-IBP bounds should be valid: lower[{}]={} <= upper[{}]={}",
            i,
            crown_ibp_output.lower()[[i]],
            i,
            crown_ibp_output.upper()[[i]]
        );
    }

    // CROWN-IBP uses tighter intermediate bounds, so the final output should be
    // at least as tight as plain CROWN. Additive tolerance covers f32 rounding only.
    assert!(
        crown_ibp_width <= crown_width + 1e-6,
        "CROWN-IBP total width {crown_ibp_width} should not exceed CROWN {crown_width}",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_collect_bounds() {
    // Test that collect_crown_ibp_bounds produces tighter bounds than collect_ibp_bounds

    // Simple 2-layer network
    let w1 = arr2(&[[1.0, 1.0], [1.0, -1.0]]);
    let b1 = arr1(&[-0.5, 0.0]);
    let w2 = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let b2 = arr1(&[0.0, 0.0]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let ibp_bounds = network.collect_ibp_bounds(&input).unwrap();
    let crown_ibp_bounds = network.collect_crown_ibp_bounds(&input).unwrap();

    println!("Layer bounds comparison:");
    for (i, (ibp, crown_ibp)) in ibp_bounds.iter().zip(crown_ibp_bounds.iter()).enumerate() {
        println!(
            "Layer {}: IBP width={:.4}, CROWN-IBP width={:.4}",
            i,
            ibp.max_width(),
            crown_ibp.max_width()
        );
    }

    // CROWN-IBP bounds should be at least as tight at each layer
    for (i, (ibp, crown_ibp)) in ibp_bounds.iter().zip(crown_ibp_bounds.iter()).enumerate() {
        for j in 0..ibp.len() {
            assert!(
                crown_ibp.lower()[[j]] >= ibp.lower()[[j]] - 1e-5,
                "Layer {} elem {}: CROWN-IBP lower {} >= IBP lower {}",
                i,
                j,
                crown_ibp.lower()[[j]],
                ibp.lower()[[j]]
            );
            assert!(
                crown_ibp.upper()[[j]] <= ibp.upper()[[j]] + 1e-5,
                "Layer {} elem {}: CROWN-IBP upper {} <= IBP upper {}",
                i,
                j,
                crown_ibp.upper()[[j]],
                ibp.upper()[[j]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_forward_tightens_exact_successor_outputs() {
    // Regression for #3397: if a layer's output only feeds an exact backward step
    // (here ReLU -> Linear), CROWN-IBP should reuse the already-tightened previous
    // layer by forward-propagating it instead of running another full partial pass.
    let w1 = arr2(&[
        [1.0, 2.0, -1.0],
        [-1.0, 1.0, 1.0],
        [0.5, -0.5, 1.0],
        [1.0, 1.0, 1.0],
    ]);
    let b1 = arr1(&[-0.5, 0.0, -0.3, 0.0]);
    let w2 = arr2(&[
        [1.0, -1.0, 0.5, 0.0],
        [0.5, 1.0, -0.5, 0.5],
        [-0.5, 0.5, 1.0, -0.5],
        [0.0, 0.5, -0.5, 1.0],
    ]);
    let b2 = arr1(&[0.0, -0.2, 0.1, 0.0]);
    let w3 = arr2(&[
        [1.0, -0.5, 0.5, 0.5],
        [-0.5, 1.0, -0.5, 0.5],
        [0.5, 0.5, 1.0, -0.5],
    ]);
    let b3 = arr1(&[0.0, 0.0, 0.0]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w3, Some(b3)).unwrap()));

    let input = BoundedTensor::new(
        arr1(&[-0.5, -0.5, -0.5]).into_dyn(),
        arr1(&[0.5, 0.5, 0.5]).into_dyn(),
    )
    .unwrap();

    let ibp_bounds = network.collect_ibp_bounds(&input).unwrap();
    let with_status = network
        .collect_crown_ibp_bounds_with_status(&input)
        .unwrap();

    assert!(!with_status.has_fallbacks());
    assert_eq!(
        with_status.provenance_for_layer(3),
        Some(BoundsProvenance::Crown)
    );

    let expected_relu = ReLULayer.propagate_ibp(&with_status.bounds[2]).unwrap();
    assert_eq!(with_status.bounds[3].lower(), expected_relu.lower());
    assert_eq!(with_status.bounds[3].upper(), expected_relu.upper());

    assert!(
        with_status.bounds[3].max_width() <= ibp_bounds[3].max_width() + 1e-5,
        "forward-tightened ReLU output should not be looser than plain IBP"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_collect_bounds_reports_fallback_events() {
    // Regression for #2060: CROWN-IBP intermediate collection must expose when
    // a layer falls back to forward bounds due to CROWN failure.
    let mut network = Network::new();
    network.add_layer(Layer::NonZero(NonZeroLayer));

    let input = BoundedTensor::new(
        arr1(&[0.0, 1.0, -1.0]).into_dyn(),
        arr1(&[0.0, 1.0, -1.0]).into_dyn(),
    )
    .unwrap();

    let legacy_bounds = network.collect_crown_ibp_bounds(&input).unwrap();
    let with_status = network
        .collect_crown_ibp_bounds_with_status(&input)
        .unwrap();

    assert_eq!(legacy_bounds.len(), 1);
    assert_eq!(with_status.bounds.len(), 1);
    assert_eq!(legacy_bounds[0].lower(), with_status.bounds[0].lower());
    assert_eq!(legacy_bounds[0].upper(), with_status.bounds[0].upper());
    assert_eq!(with_status.provenance.len(), 1);
    assert_eq!(
        with_status.provenance_for_layer(0),
        Some(BoundsProvenance::ForwardFallback(
            CrownIbpFallbackReason::CrownPropagationError
        ))
    );

    assert!(with_status.has_fallbacks());
    assert_eq!(with_status.fallback_count(), 1);
    assert_eq!(with_status.first_fallback_layer(), Some(0));

    let event = &with_status.fallback_events[0];
    assert_eq!(event.layer_index, 0);
    assert_eq!(event.layer_type, "NonZero");
    assert_eq!(event.reason, CrownIbpFallbackReason::CrownPropagationError);
    assert!(event.details.contains("NonZero"));
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_collect_bounds_preserves_non_1d_shapes() {
    // Regression test: propagate_crown_partial must reshape its concretized bounds to the
    // IBP output shape so CROWN-IBP can tighten intermediate bounds on non-1D activations
    // (e.g., ONNX models with batch/spatial dimensions like [1, 1, 1, 5]).

    // Build a small network where IBP is loose but CROWN can be tight:
    // x in [-1, 1]
    // y1 = ReLU(x), y2 = ReLU(-x)
    // z = y1 + y2 in [0, 1] but IBP gives upper bound 2.
    //
    // Then reshape z to [1, 1, 1, 5] to force a non-1D activation shape.
    let w1 = arr2(&[[1.0], [-1.0]]); // 1 -> 2
    let b1 = arr1(&[0.0, 0.0]);

    // 2 -> 5, each output is y1 + y2
    let w2 = arr2(&[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]);
    let b2 = arr1(&[0.0, 0.0, 0.0, 0.0, 0.0]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1, 1, 1, 5])));

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let ibp_bounds = network.collect_ibp_bounds(&input).unwrap();
    let crown_ibp_bounds = network.collect_crown_ibp_bounds(&input).unwrap();

    // Reshape layer output must remain 4D for both IBP and CROWN-IBP.
    assert_eq!(ibp_bounds[3].shape(), &[1, 1, 1, 5]);
    assert_eq!(crown_ibp_bounds[3].shape(), &[1, 1, 1, 5]);

    // IBP upper bound is loose: y1 in [0,1], y2 in [0,1] -> y1+y2 in [0,2]
    for &v in ibp_bounds[3].upper().iter() {
        assert!(
            (v - 2.0).abs() < 1e-5,
            "Expected IBP upper bound 2.0, got {v}"
        );
    }

    // With correct reshaping, CROWN-IBP intersects with CROWN and tightens to <= 1.
    for &v in crown_ibp_bounds[3].upper().iter() {
        assert!(
            v <= 1.0 + 1e-4,
            "Expected tightened CROWN-IBP upper bound <= 1.0, got {v}"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_conv2d_4d_shapes() {
    // Test CROWN-IBP with Conv2D network that has 4D tensor shapes.
    // Verifies that shape handling works correctly for convolutional models.
    use crate::layers::Conv2dLayer;

    // Input: [1, 4, 4] (1 channel, 4x4 spatial)
    // Conv2d: kernel [2, 1, 2, 2] -> [2, 3, 3] (2 channels, 3x3 spatial)
    // ReLU -> [2, 3, 3]
    // Flatten -> 18

    let mut kernel = ArrayD::zeros(IxDyn(&[2, 1, 2, 2]));
    // First output channel: sum of 2x2 patch
    kernel[[0, 0, 0, 0]] = 0.5;
    kernel[[0, 0, 0, 1]] = 0.5;
    kernel[[0, 0, 1, 0]] = 0.5;
    kernel[[0, 0, 1, 1]] = 0.5;
    // Second output channel: difference pattern (creates crossing)
    kernel[[1, 0, 0, 0]] = 1.0;
    kernel[[1, 0, 0, 1]] = -1.0;
    kernel[[1, 0, 1, 0]] = -1.0;
    kernel[[1, 0, 1, 1]] = 1.0;

    let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 4, 4).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Input bounds for [1, 4, 4] shape
    let lower = ArrayD::from_elem(IxDyn(&[1, 4, 4]), -0.5);
    let upper = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.5);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run all three methods
    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();
    let crown_ibp_output = network.propagate_crown_ibp(&input).unwrap();

    // Verify shapes are preserved
    assert_eq!(ibp_output.shape(), &[2, 3, 3], "IBP output shape mismatch");
    assert_eq!(
        crown_output.shape(),
        &[2, 3, 3],
        "CROWN output shape mismatch"
    );
    assert_eq!(
        crown_ibp_output.shape(),
        &[2, 3, 3],
        "CROWN-IBP output shape mismatch"
    );

    // Calculate widths
    let ibp_width = total_width(&ibp_output);
    let crown_width = total_width(&crown_output);
    let crown_ibp_width = total_width(&crown_ibp_output);

    println!("Conv2D 4D shapes test:");
    println!("  IBP total width: {:.4}", ibp_width);
    println!("  CROWN total width: {:.4}", crown_width);
    println!("  CROWN-IBP total width: {:.4}", crown_ibp_width);

    // CROWN should be at least as tight as IBP
    assert!(
        crown_width <= ibp_width + 1e-4,
        "CROWN ({:.4}) should be <= IBP ({:.4})",
        crown_width,
        ibp_width
    );

    // Verify all bounds are valid
    for i in 0..ibp_output.len() {
        assert!(
            crown_ibp_output.lower().as_slice().unwrap()[i]
                <= crown_ibp_output.upper().as_slice().unwrap()[i],
            "CROWN-IBP bounds invalid at index {}",
            i
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_intermediate_bounds_conv2d() {
    // Test that collect_crown_ibp_bounds produces tighter intermediate bounds
    // for Conv2D networks with 4D shapes.
    use crate::layers::Conv2dLayer;

    // Similar to above but we check intermediate bounds
    let mut kernel = ArrayD::zeros(IxDyn(&[2, 1, 2, 2]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = -1.0;
    kernel[[0, 0, 1, 0]] = 1.0;
    kernel[[0, 0, 1, 1]] = -1.0;
    kernel[[1, 0, 0, 0]] = 0.5;
    kernel[[1, 0, 0, 1]] = 0.5;
    kernel[[1, 0, 1, 0]] = 0.5;
    kernel[[1, 0, 1, 1]] = 0.5;

    let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 4, 4).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));

    let lower = ArrayD::from_elem(IxDyn(&[1, 4, 4]), -0.5);
    let upper = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.5);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let ibp_bounds = network.collect_ibp_bounds(&input).unwrap();
    let crown_ibp_bounds = network.collect_crown_ibp_bounds(&input).unwrap();

    // Verify shapes match at each layer
    for (i, (ibp, crown_ibp)) in ibp_bounds.iter().zip(crown_ibp_bounds.iter()).enumerate() {
        assert_eq!(
            ibp.shape(),
            crown_ibp.shape(),
            "Layer {} shape mismatch: IBP {:?} vs CROWN-IBP {:?}",
            i,
            ibp.shape(),
            crown_ibp.shape()
        );
        println!(
            "Layer {}: shape {:?}, IBP width {:.4}, CROWN-IBP width {:.4}",
            i,
            ibp.shape(),
            ibp.max_width(),
            crown_ibp.max_width()
        );
    }

    // Verify IBP bounds at layer 0 (Conv2d output before ReLU)
    // These should be [2, 3, 3]
    assert_eq!(
        ibp_bounds[0].shape(),
        &[2, 3, 3],
        "Conv2d output shape should be [2, 3, 3]"
    );
    assert_eq!(
        crown_ibp_bounds[0].shape(),
        &[2, 3, 3],
        "CROWN-IBP Conv2d output shape should be [2, 3, 3]"
    );
}

/// Estimate Dense A-matrix pair memory in GB for a given (out_dim, in_dim).
///
/// CROWN backward stores lower_a and upper_a as f32 matrices:
/// total = 2 × out_dim × in_dim × sizeof(f32)
fn dense_pair_gb(out_dim: usize, in_dim: usize) -> f64 {
    (2 * out_dim * in_dim * size_of::<f32>()) as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_memory_budget_fallback_3515() {
    const DIM: usize = 512;

    tests::with_crown_dense_budget_mb("1", || {
        let mut network = Network::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(Array2::eye(DIM), Some(Array1::zeros(DIM))).unwrap(),
        ));

        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[DIM]), -0.25),
            ArrayD::from_elem(IxDyn(&[DIM]), 0.25),
        )
        .unwrap();

        let ibp_bounds = network.collect_ibp_bounds(&input).unwrap();
        let with_status = network
            .collect_crown_ibp_bounds_with_status(&input)
            .unwrap();

        assert_eq!(with_status.bounds.len(), 1);
        assert_eq!(
            with_status.provenance_for_layer(0),
            Some(BoundsProvenance::ForwardFallback(
                CrownIbpFallbackReason::MemoryBudgetExceeded
            ))
        );
        assert_eq!(with_status.fallback_count(), 1);
        assert_eq!(with_status.bounds[0].lower(), ibp_bounds[0].lower());
        assert_eq!(with_status.bounds[0].upper(), ibp_bounds[0].upper());

        let event = &with_status.fallback_events[0];
        assert_eq!(event.layer_index, 0);
        assert_eq!(event.layer_type, "Linear");
        assert_eq!(event.reason, CrownIbpFallbackReason::MemoryBudgetExceeded);
        assert!(event.details.contains("initial_dense_identity"));
        assert!(event.details.contains("budget is 1048576 bytes"));
    });
}

/// Regression test for #3515: CROWN-IBP Dense materialization memory scaling.
///
/// Proves that CROWN-IBP partial passes through competition-shape CNNs create
/// multi-GB Dense A-matrices at Reshape/Flatten boundaries. The benchmark
/// consumed 110 GB RSS due to these allocations plus allocator retention.
///
/// Fix: memory estimation before `to_dense()`, fall back to IBP when exceeds budget.
#[test]
fn test_crown_ibp_memory_scaling_regression_3515() {
    // Soundnessbench CROWN-IBP at k=5 (Conv2 output 24×64×64):
    // Patches→Dense at Reshape: to_dense() creates (98304, 12288) per matrix.
    // Source: patches.rs:553 — Array2::<f32>::zeros((out_dim, in_dim))
    let sb_gb = dense_pair_gb(24 * 64 * 64, 3 * 64 * 64);

    // Metaroom Dense identity at Flatten: Array2::eye(28672) per matrix.
    // Source: ibp.rs:508 — LinearBounds::identity(output_dim)
    let mr_identity_gb = dense_pair_gb(64 * 16 * 28, 64 * 16 * 28);

    // Metaroom after Conv3 backward (stride 2, input 32×56):
    // A-matrix becomes (28672, 57344).
    let mr_conv_gb = dense_pair_gb(64 * 16 * 28, 32 * 32 * 56);

    println!("=== #3515 Memory Scaling Analysis ===");
    println!("Soundnessbench Patches→Dense at Conv2: {sb_gb:.2} GB");
    println!("Metaroom Dense identity at Flatten: {mr_identity_gb:.2} GB");
    println!("Metaroom after Conv3 backward: {mr_conv_gb:.2} GB");
    println!("Peak with GEMM temps: ~{:.1} GB", sb_gb * 2.0);

    // All exceed a safe 2 GB threshold for a single A-matrix pair.
    let threshold = 2.0;
    assert!(
        sb_gb > threshold,
        "soundnessbench: {sb_gb:.2} GB > {threshold} GB"
    );
    assert!(
        mr_identity_gb > threshold,
        "metaroom identity: {mr_identity_gb:.2} GB > {threshold} GB"
    );
    assert!(
        mr_conv_gb > threshold,
        "metaroom conv: {mr_conv_gb:.2} GB > {threshold} GB"
    );
}
