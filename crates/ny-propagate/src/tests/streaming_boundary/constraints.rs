// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{Array1, ArrayD};

#[ntest::timeout(10000)]
#[test]
fn test_boundary_output_constraints_construction_3500() {
    let output_dim = 12;
    let boundary_size = BOUNDARY_SAMPLES;
    let amplitude_bound = 1.0;

    let constraints = boundary_output_constraints(output_dim, boundary_size, amplitude_bound);

    assert_eq!(constraints.num_constraints(), 4 * boundary_size);
    assert_eq!(constraints.output_dim(), output_dim);

    let within = Array1::from_elem(output_dim, 0.5);
    assert!(
        constraints.is_satisfied(&within),
        "Output within bounds should satisfy constraints"
    );

    let mut outside = Array1::from_elem(output_dim, 0.5);
    outside[0] = 2.0;
    assert!(
        !constraints.is_satisfied(&outside),
        "Output outside bounds should not satisfy constraints"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_boundary_constraints_with_crown_bounds_3500() {
    let input_length = 16;
    let epsilon = 0.05;
    let (network, output_length) = build_synthetic_vocoder(input_length);
    let input = mel_input(input_length, epsilon);

    let crown_output = network
        .propagate_crown(&input)
        .expect("CROWN should succeed");
    let (first_l, first_u, last_l, last_u) =
        extract_boundary_bounds(&crown_output, output_length, BOUNDARY_SAMPLES);

    let max_abs = first_l
        .iter()
        .chain(first_u.iter())
        .chain(last_l.iter())
        .chain(last_u.iter())
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);

    let amplitude_bound = max_abs + 0.1;
    let constraints = boundary_output_constraints(output_length, BOUNDARY_SAMPLES, amplitude_bound);

    let center_input =
        BoundedTensor::concrete(ArrayD::from_elem(ndarray::IxDyn(&[1, input_length]), 0.5))
            .expect("valid concrete input");
    let center_output = network
        .propagate_ibp(&center_input)
        .expect("concrete eval should succeed");

    let flat_out = center_output.flatten();
    let out_slice = flat_out.lower().as_slice().expect("contiguous");
    let out_vec = Array1::from_vec(out_slice[..output_length].to_vec());

    assert!(
        constraints.is_satisfied(&out_vec),
        "Center point should satisfy boundary constraints (max_abs={max_abs}, bound={amplitude_bound})"
    );
}
