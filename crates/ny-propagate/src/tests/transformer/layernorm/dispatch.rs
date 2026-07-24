// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_unsupported_op_still_falls_back_to_ibp() {
    // Regression test: UnsupportedOp errors should still trigger the network-level
    // fallback to IBP. Only SoundnessRefusal should propagate as an error.
    // Reference: designs/2026-02-08-crown-fallback-error-semantics.md
    use crate::layers::FloorLayer;

    let mut network = Network::new();
    let dim = 4;

    // Linear: 4 -> 4
    let weight1 =
        Array2::from_shape_fn((dim, dim), |(i, j)| 0.3 * ((i * 17 + j * 31) as f32).sin());
    network.add_layer(Layer::Linear(LinearLayer::new(weight1, None).unwrap()));

    // Floor: not in the batched CROWN dispatch, returns UnsupportedOp
    network.add_layer(Layer::Floor(FloorLayer));

    // Linear: 4 -> 4
    let weight2 =
        Array2::from_shape_fn((dim, dim), |(i, j)| 0.3 * ((i * 23 + j * 41) as f32).cos());
    network.add_layer(Layer::Linear(LinearLayer::new(weight2, None).unwrap()));

    // Create input bounds
    let lower = ArrayD::from_elem(IxDyn(&[1, dim]), -0.5_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, dim]), 0.5_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Run batched CROWN — Floor returns UnsupportedOp from the batched dispatch
    // catch-all, so the network should silently fall back to IBP and return Ok bounds.
    let result = network.propagate_crown_batched(&input);
    assert!(
        result.is_ok(),
        "UnsupportedOp should trigger IBP fallback, not propagate as error: {:?}",
        result.err()
    );

    let bounds = result.unwrap();
    let ibp_bounds = network.propagate_ibp(&input).unwrap();

    assert_eq!(bounds.shape(), &[1, dim]);
    for (l, u) in bounds.lower().iter().zip(bounds.upper().iter()) {
        assert!(l.is_finite(), "Lower bound should be finite");
        assert!(u.is_finite(), "Upper bound should be finite");
        assert!(l <= u, "Lower bound should <= upper bound");
    }

    // UnsupportedOp fallback should produce the same result as direct IBP.
    // This guards against silently switching this test to a non-fallback path.
    for (idx, (lhs, rhs)) in bounds
        .lower()
        .iter()
        .zip(ibp_bounds.lower().iter())
        .enumerate()
    {
        assert!(
            (lhs - rhs).abs() <= 1e-6,
            "lower mismatch vs IBP at {}: fallback={} ibp={}",
            idx,
            lhs,
            rhs
        );
    }
    for (idx, (lhs, rhs)) in bounds
        .upper()
        .iter()
        .zip(ibp_bounds.upper().iter())
        .enumerate()
    {
        assert!(
            (lhs - rhs).abs() <= 1e-6,
            "upper mismatch vs IBP at {}: fallback={} ibp={}",
            idx,
            lhs,
            rhs
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_constant_arithmetic_layer_dispatch_batched_matches_unbatched() {
    let linear_bounds = LinearBounds::new(
        arr2(&[[1.0, -0.5, 0.2], [0.3, 0.0, -1.1]]),
        arr1(&[0.4, -0.8]),
        arr2(&[[-0.2, 0.9, 0.5], [1.2, -0.4, 0.7]]),
        arr1(&[-0.1, 0.6]),
    )
    .unwrap();
    let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
        linear_bounds.lower_a.clone().into_dyn(),
        linear_bounds.lower_b.clone().into_dyn(),
        linear_bounds.upper_a.clone().into_dyn(),
        linear_bounds.upper_b.clone().into_dyn(),
        vec![linear_bounds.num_inputs()],
        vec![linear_bounds.num_outputs()],
    );

    let layers = [
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[]), 0.25))),
        Layer::SubConstant(SubConstantLayer::scalar(0.10)),
        Layer::SubConstant(SubConstantLayer::new_reverse(ArrayD::from_elem(
            IxDyn(&[]),
            -0.35,
        ))),
        Layer::MulConstant(MulConstantLayer::scalar(-1.25)),
        Layer::DivConstant(DivConstantLayer::scalar(2.0)),
    ];

    for layer in layers {
        let name = layer.layer_type();

        let expected = layer
            .propagate_crown_backward(&linear_bounds, None)
            .unwrap_or_else(|e| panic!("{name} unbatched CROWN backward failed: {e:?}"));
        let actual = layer
            .propagate_crown_backward_batched(&batched_bounds, None, None)
            .unwrap_or_else(|e| panic!("{name} batched CROWN backward failed: {e:?}"));

        tests::assert_all_close(
            &actual.lower_a,
            &expected.lower_a.into_dyn(),
            1e-6,
            &format!("{name} lower_a"),
        );
        tests::assert_all_close(
            &actual.lower_b,
            &expected.lower_b.into_dyn(),
            1e-6,
            &format!("{name} lower_b"),
        );
        tests::assert_all_close(
            &actual.upper_a,
            &expected.upper_a.into_dyn(),
            1e-6,
            &format!("{name} upper_a"),
        );
        tests::assert_all_close(
            &actual.upper_b,
            &expected.upper_b.into_dyn(),
            1e-6,
            &format!("{name} upper_b"),
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_constant_arithmetic_network_batched_crown_soundness() {
    // Network-level soundness test for the 4 constant arithmetic layers wired
    // into batched CROWN dispatch (W1 commit e4cbc87, #1708).
    // Verify that batched CROWN bounds contain the true output at concrete points.

    let dim = 4;
    let mut network = Network::new();

    // Linear -> AddConstant -> MulConstant -> SubConstant -> DivConstant -> ReLU -> Linear
    let weight1 =
        Array2::from_shape_fn((dim, dim), |(i, j)| 0.3 * ((i * 17 + j * 31) as f32).sin());
    network.add_layer(Layer::Linear(LinearLayer::new(weight1, None).unwrap()));

    network.add_layer(Layer::AddConstant(AddConstantLayer::new(
        ArrayD::from_elem(IxDyn(&[]), 0.25),
    )));
    network.add_layer(Layer::MulConstant(MulConstantLayer::scalar(-1.5)));
    network.add_layer(Layer::SubConstant(SubConstantLayer::scalar(0.1)));
    network.add_layer(Layer::DivConstant(DivConstantLayer::scalar(2.0)));
    network.add_layer(Layer::ReLU(ReLULayer));

    let weight2 =
        Array2::from_shape_fn((dim, dim), |(i, j)| 0.2 * ((i * 23 + j * 41) as f32).cos());
    network.add_layer(Layer::Linear(LinearLayer::new(weight2, None).unwrap()));

    let lower = ArrayD::from_elem(IxDyn(&[1, dim]), -0.5_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, dim]), 0.5_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let bounds = network
        .propagate_crown_batched(&input)
        .expect("Batched CROWN with constant arithmetic layers should succeed");

    // Check bounds are finite and non-inverted
    for (l, u) in bounds.lower().iter().zip(bounds.upper().iter()) {
        assert!(l.is_finite() && u.is_finite(), "Bounds must be finite");
        assert!(l <= u, "Bounds must not be inverted: {} > {}", l, u);
    }

    // Soundness check: evaluate network at concrete points within input bounds
    // by using propagate_ibp with lower==upper (gives exact forward pass output).
    let test_points: Vec<Vec<f32>> = vec![
        vec![-0.5, -0.5, -0.5, -0.5],
        vec![0.5, 0.5, 0.5, 0.5],
        vec![0.0, 0.0, 0.0, 0.0],
        vec![-0.5, 0.5, -0.5, 0.5],
        vec![0.25, -0.3, 0.1, -0.4],
    ];

    for (idx, point) in test_points.iter().enumerate() {
        let pt = ArrayD::from_shape_vec(IxDyn(&[1, dim]), point.clone()).unwrap();
        let concrete_input = BoundedTensor::new(pt.clone(), pt).unwrap();
        let concrete_output = network
            .propagate_ibp(&concrete_input)
            .expect("Concrete forward pass should succeed");
        for (j, (&val_l, &val_u)) in concrete_output
            .lower()
            .iter()
            .zip(concrete_output.upper().iter())
            .enumerate()
        {
            // With lower==upper, IBP gives exact output, so val_l ≈ val_u
            assert!(
                (val_l - val_u).abs() < 1e-5,
                "Concrete output should be a point, not an interval"
            );
            let val = val_l;
            let lb = bounds.lower()[[0, j]];
            let ub = bounds.upper()[[0, j]];
            assert!(
                val >= lb - 1e-5 && val <= ub + 1e-5,
                "Soundness violation at point {}, output {}: val={} not in [{}, {}]",
                idx,
                j,
                val,
                lb,
                ub
            );
        }
    }

    // Verify IBP bounds are also sound (for comparison)
    let ibp_bounds = network.propagate_ibp(&input).unwrap();
    for (l, u) in ibp_bounds.lower().iter().zip(ibp_bounds.upper().iter()) {
        assert!(l.is_finite() && u.is_finite(), "IBP bounds must be finite");
        assert!(l <= u, "IBP bounds must not be inverted: {} > {}", l, u);
    }

    // Note: CROWN is not guaranteed to be tighter than IBP in all network
    // configurations. With negative MulConstant and ReLU, the linear relaxation
    // can sometimes be looser. Both must be sound (contain all true outputs).
}
