// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Discontinuous layer CROWN/IBP consistency ====================

#[ntest::timeout(10000)]
#[test]
fn test_discontinuous_layers_crown_equals_ibp() {
    // For discontinuous layers with constant bounds (slope=0),
    // CROWN should produce the same bounds as IBP
    let input_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.5, 0.3, 2.1, -0.8]).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.2, 1.7, 3.9, 0.4]).unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    // Test each discontinuous layer type
    let layers_to_test: Vec<(&str, Layer)> = vec![
        ("Floor", Layer::Floor(FloorLayer)),
        ("Ceil", Layer::Ceil(CeilLayer)),
        ("Round", Layer::Round(RoundLayer)),
        ("Sign", Layer::Sign(SignLayer)),
    ];

    for (name, layer) in layers_to_test {
        let mut network = Network::new();
        network.add_layer(layer);

        let crown_result = network.propagate_crown(&input).unwrap();
        let ibp_result = network.propagate_ibp(&input).unwrap();

        for i in 0..4 {
            assert!(
                (crown_result.lower()[[i]] - ibp_result.lower()[[i]]).abs() < 1e-4,
                "{} CROWN lower should match IBP at {}: {} vs {}",
                name,
                i,
                crown_result.lower()[[i]],
                ibp_result.lower()[[i]]
            );
            assert!(
                (crown_result.upper()[[i]] - ibp_result.upper()[[i]]).abs() < 1e-4,
                "{} CROWN upper should match IBP at {}: {} vs {}",
                name,
                i,
                crown_result.upper()[[i]],
                ibp_result.upper()[[i]]
            );
        }
    }
}
