// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GraphNetwork domain clipping integration tests.
use crate::*;
use ndarray::{arr1, arr2};

#[ntest::timeout(10000)]
#[test]
fn test_domain_clipping_collect_statistics() {
    use crate::domain_clip::DomainClipper;

    // Build a simple network: input -> linear -> relu
    let mut graph = GraphNetwork::new();

    let weights = arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]);
    let bias = arr1(&[0.1_f32, 0.2]);
    let linear_layer = LinearLayer::new(weights, Some(bias)).unwrap();
    let linear_node = GraphNode::from_input("linear", Layer::Linear(linear_layer));
    graph.add_node(linear_node);

    let relu_node = GraphNode::new("relu", Layer::ReLU(ReLULayer), vec!["linear".to_string()]);
    graph.add_node(relu_node);
    graph.set_output("relu");

    // Collect statistics from concrete forward pass
    let mut clipper = DomainClipper::default();
    let concrete_input = BoundedTensor::concrete(arr1(&[1.0_f32, 2.0]).into_dyn()).unwrap();

    graph
        .collect_activation_statistics(&concrete_input, &mut clipper)
        .unwrap();

    // Verify statistics were collected
    let summary = clipper.summary();
    assert_eq!(summary.total_layers, 2); // linear and relu
    assert_eq!(summary.total_samples, 2); // one sample per layer
}

#[ntest::timeout(10000)]
#[test]
fn test_domain_clipping_tightens_bounds() {
    use crate::domain_clip::{ClipStrategy, DomainClipConfig, DomainClipper};

    // Build a simple network: input -> linear
    let mut graph = GraphNetwork::new();

    let weights = arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]); // Identity-like
    let bias = arr1(&[0.0_f32, 0.0]);
    let linear_layer = LinearLayer::new(weights, Some(bias)).unwrap();
    let linear_node = GraphNode::from_input("linear", Layer::Linear(linear_layer));
    graph.add_node(linear_node);
    graph.set_output("linear");

    // Collect statistics from multiple concrete samples
    let mut clipper = DomainClipper::new(DomainClipConfig {
        strategy: ClipStrategy::Empirical { margin_factor: 0.1 },
        min_samples: 1,
        enabled: true,
        exclude_patterns: vec![],
        max_tightening_factor: 100.0,
    });

    // Samples around [1, 2]
    for _ in 0..10 {
        let sample = BoundedTensor::concrete(arr1(&[1.0_f32, 2.0]).into_dyn()).unwrap();
        graph
            .collect_activation_statistics(&sample, &mut clipper)
            .unwrap();
    }

    // Now propagate with very wide input bounds
    let wide_input = BoundedTensor::new(
        arr1(&[-100.0_f32, -100.0]).into_dyn(),
        arr1(&[100.0_f32, 100.0]).into_dyn(),
    )
    .unwrap();

    // Without clipping
    let bounds_no_clip = graph.propagate_ibp(&wide_input).unwrap();

    // With clipping
    let bounds_clipped = graph
        .propagate_ibp_with_clipper(&wide_input, &mut clipper)
        .unwrap();

    // Clipped bounds should be tighter
    let width_no_clip = bounds_no_clip.max_width();
    let width_clipped = bounds_clipped.max_width();

    assert!(
        width_clipped < width_no_clip,
        "Clipped bounds ({:.2}) should be tighter than unclipped ({:.2})",
        width_clipped,
        width_no_clip
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_domain_clipping_soundness() {
    use crate::domain_clip::{ClipStrategy, DomainClipConfig, DomainClipper};

    // Build a network: input -> linear -> relu
    let mut graph = GraphNetwork::new();

    let weights = arr2(&[[2.0_f32, -1.0], [1.0, 3.0]]);
    let bias = arr1(&[0.5_f32, -0.5]);
    let linear_layer = LinearLayer::new(weights, Some(bias)).unwrap();
    let linear_node = GraphNode::from_input("linear", Layer::Linear(linear_layer));
    graph.add_node(linear_node);

    let relu_node = GraphNode::new("relu", Layer::ReLU(ReLULayer), vec!["linear".to_string()]);
    graph.add_node(relu_node);
    graph.set_output("relu");

    // Collect statistics from samples in a specific range
    let mut clipper = DomainClipper::new(DomainClipConfig {
        strategy: ClipStrategy::Statistical { k: 6.0 }, // 6-sigma bounds
        min_samples: 5,
        enabled: true,
        exclude_patterns: vec![],
        max_tightening_factor: 1000.0,
    });

    // Collect stats from samples in [-1, 1] range
    for i in 0..20 {
        let x = (i as f32 - 10.0) / 10.0; // -1.0 to 0.9
        let sample = BoundedTensor::concrete(arr1(&[x, x * 0.5]).into_dyn()).unwrap();
        graph
            .collect_activation_statistics(&sample, &mut clipper)
            .unwrap();
    }

    // Propagate with bounds that include our sample range
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -0.5]).into_dyn(),
        arr1(&[1.0_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let bounds_clipped = graph
        .propagate_ibp_with_clipper(&input, &mut clipper)
        .unwrap();

    // Verify clipped bounds are valid (lower <= upper)
    for (l, u) in bounds_clipped
        .lower()
        .iter()
        .zip(bounds_clipped.upper().iter())
    {
        assert!(
            l <= u,
            "Clipped bounds must be valid: lower={} <= upper={}",
            l,
            u
        );
    }

    // Clipping only ever tightens: the clipped output must lie within the
    // unclipped output elementwise.
    let bounds_no_clip = graph.propagate_ibp(&input).unwrap();
    for i in 0..bounds_clipped.len() {
        let clip_l = bounds_clipped.lower().as_slice().unwrap()[i];
        let clip_u = bounds_clipped.upper().as_slice().unwrap()[i];
        let orig_l = bounds_no_clip.lower().as_slice().unwrap()[i];
        let orig_u = bounds_no_clip.upper().as_slice().unwrap()[i];
        assert!(
            clip_l >= orig_l - 1e-5 && clip_u <= orig_u + 1e-5,
            "Clipped bounds [{}, {}] at index {} must be within unclipped bounds [{}, {}]",
            clip_l,
            clip_u,
            i,
            orig_l,
            orig_u
        );
    }

    // Verify bounds contain concrete outputs for inputs in range
    let test_inputs = vec![
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[-0.5_f32, -0.25]).into_dyn(),
        arr1(&[0.5_f32, 0.25]).into_dyn(),
    ];

    for test_input in test_inputs {
        let concrete = BoundedTensor::concrete(test_input.clone()).unwrap();
        let concrete_output = graph.propagate_ibp(&concrete).unwrap();

        // Concrete output should be within clipped bounds
        for i in 0..concrete_output.len() {
            let val = concrete_output.lower().as_slice().unwrap()[i];
            let clip_l = bounds_clipped.lower().as_slice().unwrap()[i];
            let clip_u = bounds_clipped.upper().as_slice().unwrap()[i];

            assert!(
                val >= clip_l - 1e-5 && val <= clip_u + 1e-5,
                "Concrete output {} at index {} should be within clipped bounds [{}, {}]",
                val,
                i,
                clip_l,
                clip_u
            );
        }
    }
}

// =========================================================================
// BatchNorm CROWN Tests
// =========================================================================

// Test BatchNorm IBP propagation with positive scale.
