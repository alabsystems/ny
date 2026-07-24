// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Where (conditional) tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_where_ibp_ternary_basic() {
    // Test Where: output = x if condition else y
    // For interval bounds, result is union of x and y bounds
    let condition_lower = ArrayD::from_elem(IxDyn(&[3]), 0.0f32);
    let condition_upper = ArrayD::from_elem(IxDyn(&[3]), 1.0f32);
    let condition = BoundedTensor::new(condition_lower, condition_upper).unwrap();

    // x bounds: [2, 5]
    let x_lower = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let x_upper = ArrayD::from_elem(IxDyn(&[3]), 5.0f32);
    let x = BoundedTensor::new(x_lower, x_upper).unwrap();

    // y bounds: [-1, 3]
    let y_lower = ArrayD::from_elem(IxDyn(&[3]), -1.0f32);
    let y_upper = ArrayD::from_elem(IxDyn(&[3]), 3.0f32);
    let y = BoundedTensor::new(y_lower, y_upper).unwrap();

    let where_layer = WhereLayer::new();
    let output = where_layer
        .propagate_ibp_ternary(&condition, &x, &y)
        .unwrap();

    // Union of [2, 5] and [-1, 3] = [-1, 5]
    for i in 0..3 {
        assert!(
            (output.lower()[[i]] - (-1.0)).abs() < 1e-6,
            "lower should be min(-1, 2) = -1, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 5.0).abs() < 1e-6,
            "upper should be max(5, 3) = 5, got {}",
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_where_ibp_ternary_non_overlapping() {
    // Test Where with non-overlapping intervals
    let condition_lower = ArrayD::from_elem(IxDyn(&[2]), 0.0f32);
    let condition_upper = ArrayD::from_elem(IxDyn(&[2]), 1.0f32);
    let condition = BoundedTensor::new(condition_lower, condition_upper).unwrap();

    // x bounds: [10, 20]
    let x_lower = ArrayD::from_elem(IxDyn(&[2]), 10.0f32);
    let x_upper = ArrayD::from_elem(IxDyn(&[2]), 20.0f32);
    let x = BoundedTensor::new(x_lower, x_upper).unwrap();

    // y bounds: [-5, 5]
    let y_lower = ArrayD::from_elem(IxDyn(&[2]), -5.0f32);
    let y_upper = ArrayD::from_elem(IxDyn(&[2]), 5.0f32);
    let y = BoundedTensor::new(y_lower, y_upper).unwrap();

    let where_layer = WhereLayer::new();
    let output = where_layer
        .propagate_ibp_ternary(&condition, &x, &y)
        .unwrap();

    // Union of [10, 20] and [-5, 5] = [-5, 20]
    for i in 0..2 {
        assert!(
            (output.lower()[[i]] - (-5.0)).abs() < 1e-6,
            "lower should be -5, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 20.0).abs() < 1e-6,
            "upper should be 20, got {}",
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_where_ibp_error_not_supported() {
    // Single-input IBP should fail
    let lower = ArrayD::from_elem(IxDyn(&[3]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let where_layer = WhereLayer::new();
    assert!(where_layer.propagate_ibp(&input).is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_where_linear_not_supported() {
    let bounds = LinearBounds::identity(4);
    let where_layer = WhereLayer::new();
    assert!(where_layer.propagate_linear(&bounds).is_err());
}

/// Sequential CROWN through an embedded-constant Where layer (single `cond`
/// input flowing down the chain; both branches embedded constants). The output
/// is a constant vector w.r.t. the chain input, so the new sequential
/// `crown_backward_step` embedded-constant handler must produce the EXACT
/// per-element select when `cond` is constant — tighter than (and contained
/// within) the IBP union — instead of dropping to whole-network IBP fallback.
///
/// Chain: input z -> MulConstant(0) -> AddConstant(mask) -> Where(true,false).
/// cond = mask (a constant 0/1 vector regardless of z), out[i] = true[i] if
/// mask[i] else false[i].
#[ntest::timeout(10000)]
#[test]
fn test_sequential_crown_embedded_constant_where_exact_and_sound() {
    let mask = vec![1.0_f32, 0.0, 1.0];
    let const_true = ArrayD::from_shape_vec(IxDyn(&[3]), vec![7.0_f32, 8.0, 9.0]).unwrap();
    let const_false = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-7.0_f32, -8.0, -9.0]).unwrap();

    let mut net = Network::new();
    net.add_layer(Layer::MulConstant(MulConstantLayer::scalar(0.0)));
    net.add_layer(Layer::AddConstant(AddConstantLayer::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), mask.clone()).unwrap(),
    )));
    net.add_layer(Layer::Where(WhereLayer::with_constants(
        Some(const_true.clone()),
        Some(const_false.clone()),
    )));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0_f32, -2.0, -2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0_f32, 2.0, 2.0]).unwrap(),
    )
    .unwrap();

    let crown = net.propagate_crown(&input).unwrap();
    let ibp = net.propagate_ibp(&input).unwrap();

    let expected: Vec<f32> = (0..3)
        .map(|i| {
            if mask[i] >= 0.5 {
                const_true[[i]]
            } else {
                const_false[[i]]
            }
        })
        .collect();

    for i in 0..3 {
        // Soundness: CROWN bound contains the exact constant output.
        assert!(
            expected[i] >= crown.lower()[[i]] - 1e-4 && expected[i] <= crown.upper()[[i]] + 1e-4,
            "seq CROWN: out[{i}]={} not in [{}, {}]",
            expected[i],
            crown.lower()[[i]],
            crown.upper()[[i]],
        );
        // Exactness: constant cond => zero-width interval at the selected branch.
        assert!(
            (crown.lower()[[i]] - expected[i]).abs() < 1e-4,
            "seq CROWN lower[{i}]={} != {}",
            crown.lower()[[i]],
            expected[i]
        );
        assert!(
            (crown.upper()[[i]] - expected[i]).abs() < 1e-4,
            "seq CROWN upper[{i}]={} != {}",
            crown.upper()[[i]],
            expected[i]
        );
        // No looser than IBP union.
        assert!(crown.lower()[[i]] >= ibp.lower()[[i]] - 1e-4);
        assert!(crown.upper()[[i]] <= ibp.upper()[[i]] + 1e-4);
    }
    // And strictly tighter than the IBP union (which would be [-|c|, |c|] per elt).
    let strictly_tighter = (0..3).any(|i| {
        crown.lower()[[i]] > ibp.lower()[[i]] + 1e-4 || crown.upper()[[i]] < ibp.upper()[[i]] - 1e-4
    });
    assert!(
        strictly_tighter,
        "seq CROWN embedded-constant Where should tighten over the IBP union"
    );
}
