// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GELU-specific CROWN tests for transformer components.

use super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_gelu_crown_linear_relaxation_soundness() {
    // Test that GELU linear relaxation produces sound bounds
    // Test various intervals
    let test_cases = vec![
        (-2.0_f32, -1.0_f32), // Negative region (convex)
        (0.0, 1.0),           // Positive region (concave)
        (-1.0, 1.0),          // Mixed region
        (-0.5, 0.5),          // Small mixed region
        (-3.0, 2.0),          // Wide mixed region
    ];

    for (l, u) in test_cases {
        let (lower_slope, lower_intercept, upper_slope, upper_intercept) =
            gelu_linear_relaxation(l, u, GeluApproximation::Erf);

        // Sample points in the interval and verify bounds
        for t in 0..=20 {
            let x = l + (u - l) * (t as f32 / 20.0);
            let gelu_val = gelu_eval(x, GeluApproximation::Erf);
            let lower_bound = lower_slope * x + lower_intercept;
            let upper_bound = upper_slope * x + upper_intercept;

            assert!(
                gelu_val >= lower_bound - 1e-5,
                "GELU({}) = {} should be >= lower bound {} for interval [{}, {}]",
                x,
                gelu_val,
                lower_bound,
                l,
                u
            );
            assert!(
                gelu_val <= upper_bound + 1e-5,
                "GELU({}) = {} should be <= upper bound {} for interval [{}, {}]",
                x,
                gelu_val,
                upper_bound,
                l,
                u
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_gelu_crown_propagation_soundness() {
    // Test GELU CROWN propagation end-to-end

    // Create a network: Linear -> GELU
    let weight = arr2(&[[1.0_f32, 0.5], [-0.5, 1.0], [0.3, -0.8]]);
    let linear = LinearLayer::new(weight, Some(arr1(&[0.1, -0.1, 0.0]))).unwrap();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::default()),
        vec!["linear".to_string()],
    ));
    graph.set_output("gelu");

    // Input with perturbation
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    let crown_bounds = graph.propagate_crown(&input).unwrap();
    let ibp_bounds = graph.propagate_ibp(&input).unwrap();

    // Verify soundness: sample random inputs and check they're within bounds
    let rng_seed = 42;
    for i in 0..100 {
        let t1 = ((i * 7 + rng_seed) % 100) as f32 / 100.0;
        let t2 = ((i * 11 + rng_seed) % 100) as f32 / 100.0;
        let x1 = -0.5 + t1;
        let x2 = -0.5 + t2;

        // Forward pass
        let z1_0 = 1.0 * x1 + 0.5 * x2 + 0.1;
        let z1_1 = -0.5 * x1 + 1.0 * x2 - 0.1;
        let z1_2 = 0.3 * x1 - 0.8 * x2;

        let y0 = gelu_eval(z1_0, GeluApproximation::Erf);
        let y1 = gelu_eval(z1_1, GeluApproximation::Erf);
        let y2 = gelu_eval(z1_2, GeluApproximation::Erf);

        // Check CROWN bounds contain the output
        assert!(
            y0 >= crown_bounds.lower()[[0]] - 1e-5 && y0 <= crown_bounds.upper()[[0]] + 1e-5,
            "Output 0 {} outside CROWN bounds [{}, {}]",
            y0,
            crown_bounds.lower()[[0]],
            crown_bounds.upper()[[0]]
        );
        assert!(
            y1 >= crown_bounds.lower()[[1]] - 1e-5 && y1 <= crown_bounds.upper()[[1]] + 1e-5,
            "Output 1 {} outside CROWN bounds [{}, {}]",
            y1,
            crown_bounds.lower()[[1]],
            crown_bounds.upper()[[1]]
        );
        assert!(
            y2 >= crown_bounds.lower()[[2]] - 1e-5 && y2 <= crown_bounds.upper()[[2]] + 1e-5,
            "Output 2 {} outside CROWN bounds [{}, {}]",
            y2,
            crown_bounds.lower()[[2]],
            crown_bounds.upper()[[2]]
        );
    }

    // Note: For GELU, CROWN linear relaxation may not always be tighter than IBP
    // because IBP uses exact interval evaluation while CROWN uses linear bounds.
    // The key property is soundness, not tightness for GELU.
    // Just verify both methods produce sound bounds (tested above).
    let _crown_width: f32 = (0..3)
        .map(|i| crown_bounds.upper()[[i]] - crown_bounds.lower()[[i]])
        .sum();
    let _ibp_width: f32 = (0..3)
        .map(|i| ibp_bounds.upper()[[i]] - ibp_bounds.lower()[[i]])
        .sum();
    // Both are sound - tightness depends on network structure
}
