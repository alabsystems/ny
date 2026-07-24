// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::MatMulLayer;
use crate::BatchedLinearBounds;

#[ntest::timeout(10000)]
#[test]
fn batched_crown_bias_halving_uses_directed_rounding_2173() {
    let matmul = MatMulLayer::new(false, None);

    // Concrete 1x1 inputs keep McCormick terms inactive when incoming A is zero.
    let input_a = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1]), 0.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1]), 0.0_f32),
    )
    .expect("input_a");
    let input_b = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1]), 0.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1]), 0.0_f32),
    )
    .expect("input_b");

    let mut bounds = BatchedLinearBounds::identity(&[1]).expect("identity bounds");
    bounds.lower_a.fill(0.0);
    bounds.upper_a.fill(0.0);

    // Subnormal values where "round then halve" differs by 1 ULP from
    // "halve in f64 then directed-round" (the sound formulation).
    let lower_seed = f32::from_bits(0x004a_e545);
    let upper_seed = f32::from_bits(0x0063_e42f);
    bounds.lower_b[[0]] = lower_seed;
    bounds.upper_b[[0]] = upper_seed;

    let (bounds_a, bounds_b) = matmul
        .propagate_linear_batched_binary(&bounds, &input_a, &input_b)
        .expect("batched crown");

    let expected_lower_half = next_down_f32(((lower_seed as f64) * 0.5) as f32);
    let expected_upper_half = next_up_f32(((upper_seed as f64) * 0.5) as f32);
    let buggy_lower_half = next_down_f32(lower_seed) * 0.5;
    let buggy_upper_half = next_up_f32(upper_seed) * 0.5;

    assert_eq!(
        bounds_a.lower_b[[0]].to_bits(),
        expected_lower_half.to_bits(),
        "lower bias must halve in f64 before directed rounding",
    );
    assert_eq!(
        bounds_b.lower_b[[0]].to_bits(),
        expected_lower_half.to_bits(),
        "lower bias split must match for both operand branches",
    );
    assert_eq!(
        bounds_a.upper_b[[0]].to_bits(),
        expected_upper_half.to_bits(),
        "upper bias must halve in f64 before directed rounding",
    );
    assert_eq!(
        bounds_b.upper_b[[0]].to_bits(),
        expected_upper_half.to_bits(),
        "upper bias split must match for both operand branches",
    );
    assert_ne!(
        expected_lower_half.to_bits(),
        buggy_lower_half.to_bits(),
        "regression seed must differ from the buggy round-then-halve path",
    );
    assert_ne!(
        expected_upper_half.to_bits(),
        buggy_upper_half.to_bits(),
        "regression seed must differ from the buggy round-then-halve path",
    );
}
