// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use ndarray::{Array1, Array2};
use ny_propagate::LinearBounds;
use ny_tensor::{BoundedTensor, RepairStrategy};

const MAX_DIM: u8 = 8;

#[derive(Debug, Arbitrary)]
struct FuzzLinearCase {
    outputs: u8,
    inputs: u8,
    coeffs: Vec<f32>,
    lower_bias: Vec<f32>,
    upper_bias: Vec<f32>,
    lower_input: Vec<f32>,
    upper_input: Vec<f32>,
    widen_input: bool,
}

fn clamp_dim(value: u8) -> usize {
    usize::from(value.min(MAX_DIM))
}

fn resize_values(mut values: Vec<f32>, len: usize) -> Vec<f32> {
    values.resize(len, 0.0);
    values
}

fn vector(values: Vec<f32>, len: usize) -> Array1<f32> {
    Array1::from_vec(resize_values(values, len))
}

fn matrix(values: Vec<f32>, rows: usize, cols: usize) -> Array2<f32> {
    Array2::from_shape_vec(
        (rows, cols),
        resize_values(values, rows.saturating_mul(cols)),
    )
    .expect("matrix shape is derived from the resized vector length")
}

fuzz_target!(|case: FuzzLinearCase| {
    let outputs = clamp_dim(case.outputs);
    let inputs = clamp_dim(case.inputs);

    let lower_a = matrix(case.coeffs.clone(), outputs, inputs);
    let upper_a = matrix(case.coeffs, outputs, inputs);
    let lower_b = vector(case.lower_bias, outputs);
    let upper_b = vector(case.upper_bias, outputs);
    let linear_bounds = LinearBounds::new_repaired(
        lower_a,
        lower_b,
        upper_a,
        upper_b,
        RepairStrategy::Conservative,
    )
    .expect("new_repaired only fails on shape mismatches");

    let input_strategy = if case.widen_input {
        RepairStrategy::Widen
    } else {
        RepairStrategy::Conservative
    };
    let input_bounds = BoundedTensor::new_repaired(
        vector(case.lower_input, inputs).into_dyn(),
        vector(case.upper_input, inputs).into_dyn(),
        input_strategy,
    )
    .expect("new_repaired only fails on shape mismatches");

    let _ = linear_bounds.concretize(&input_bounds);
    let concretized_checked = linear_bounds.concretize_checked(&input_bounds);
    let concretized_sound = linear_bounds.concretize_sound(&input_bounds);

    if let Ok(bounds) = concretized_checked {
        assert_eq!(bounds.shape(), &[outputs]);
    }
    assert_eq!(concretized_sound.shape(), &[outputs]);
});
