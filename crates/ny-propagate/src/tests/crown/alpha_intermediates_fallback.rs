// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for alpha-CROWN intermediates fallback behavior.

use crate::network::NetworkAlphaCrownExt;
use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

fn build_network_with_unsupported_where() -> Network {
    let mut network = Network::new();
    let w1 = arr2(&[[1.0_f32, 2.0], [-1.0, 1.0], [0.5, -0.5]]);
    let b1 = arr1(&[0.1, -0.2, 0.3]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let const_true = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0_f32, 2.0, 3.0]).unwrap();
    let const_false = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0_f32, 0.0, 0.5]).unwrap();
    network.add_layer(Layer::Where(WhereLayer {
        const_true: Some(const_true),
        const_false: Some(const_false),
    }));

    let w2 = arr2(&[[1.0_f32, -1.0, 0.5], [0.5, 1.0, -0.5]]);
    let b2 = arr1(&[0.0, 0.1]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));
    network
}

fn assert_zero_slopes(intermediate: &AlphaCrownIntermediate, outputs: usize, inputs: usize) {
    assert_eq!(intermediate.final_bounds.lower_a.nrows(), outputs);
    assert_eq!(intermediate.final_bounds.lower_a.ncols(), inputs);
    assert_eq!(intermediate.final_bounds.upper_a.nrows(), outputs);
    assert_eq!(intermediate.final_bounds.upper_a.ncols(), inputs);
    assert!(intermediate
        .final_bounds
        .lower_a
        .iter()
        .all(|v| v.abs() <= 1e-8));
    assert!(intermediate
        .final_bounds
        .upper_a
        .iter()
        .all(|v| v.abs() <= 1e-8));
}

fn assert_bias_and_concrete_match_crown(
    intermediate: &AlphaCrownIntermediate,
    crown_bounds: &BoundedTensor,
    input: &BoundedTensor,
) {
    let crown_flat = crown_bounds.flatten();
    for i in 0..crown_flat.len() {
        assert!((intermediate.final_bounds.lower_b[i] - crown_flat.lower()[[i]]).abs() <= 1e-5);
        assert!((intermediate.final_bounds.upper_b[i] - crown_flat.upper()[[i]]).abs() <= 1e-5);
    }

    let fallback_concrete = intermediate.final_bounds.concretize(input);
    for i in 0..crown_flat.len() {
        assert!((fallback_concrete.lower()[[i]] - crown_flat.lower()[[i]]).abs() <= 1e-5);
        assert!((fallback_concrete.upper()[[i]] - crown_flat.upper()[[i]]).abs() <= 1e-5);
    }
}

/// Regression test for #2189: AnalyticChain intermediates fallback on UnsupportedOp
/// must return constant CROWN bounds (A=0, b=CROWN concrete bounds), not identity.
#[ntest::timeout(10000)]
#[test]
fn test_analytic_chain_intermediates_unsupported_op_returns_constant_bounds_2189() {
    let network = build_network_with_unsupported_where();
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    let layer_bounds = network.collect_crown_ibp_bounds(&input).unwrap();
    let pre_activation_bounds = vec![layer_bounds[0].clone()];
    let alpha_state = AlphaState::from_preactivation_bounds(&pre_activation_bounds, &[0]).unwrap();

    let intermediate = network
        .propagate_alpha_crown_with_intermediates_impl(&input, &layer_bounds, &alpha_state, None)
        .expect("UnsupportedOp intermediates path should fallback to CROWN and succeed");

    assert!(intermediate.a_at_relu.is_empty());
    assert!(intermediate.pre_relu_bounds.is_empty());

    let crown_bounds = network.propagate_crown(&input).unwrap();
    assert_zero_slopes(
        &intermediate,
        crown_bounds.flatten().len(),
        input.flatten().len(),
    );
    assert_bias_and_concrete_match_crown(&intermediate, &crown_bounds, &input);
}
