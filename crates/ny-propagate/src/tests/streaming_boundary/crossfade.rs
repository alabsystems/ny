// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_crossfade_energy_bounded_3500() {
    // Acceptance criterion 2: Prove crossfade energy stays in [0.5, 1.5]
    // of steady-state energy.
    let input_length = 16;
    let epsilon = 0.05;
    let (network, output_length) = build_synthetic_vocoder(input_length);

    let input_a = mel_input(input_length, epsilon);
    let input_b = mel_input(input_length, epsilon);

    let crown_a = network
        .propagate_crown(&input_a)
        .expect("CROWN chunk A should succeed");
    let crown_b = network
        .propagate_crown(&input_b)
        .expect("CROWN chunk B should succeed");

    let (_a_first_l, _a_first_u, a_last_l, a_last_u) =
        extract_boundary_bounds(&crown_a, output_length, BOUNDARY_SAMPLES);
    let (b_first_l, b_first_u, _b_last_l, _b_last_u) =
        extract_boundary_bounds(&crown_b, output_length, BOUNDARY_SAMPLES);

    let (xfade_lower, xfade_upper) =
        crossfade_overlap_add_bounds(&a_last_l, &a_last_u, &b_first_l, &b_first_u);
    let (xfade_e_lower, xfade_e_upper) = energy_bounds(&xfade_lower, &xfade_upper);
    let (steady_a_lower, steady_a_upper) = energy_bounds(&a_last_l, &a_last_u);
    let (steady_b_lower, steady_b_upper) = energy_bounds(&b_first_l, &b_first_u);

    let energy_threshold = 1e-6;
    for i in 0..BOUNDARY_SAMPLES {
        let steady_e_upper = steady_a_upper[i].max(steady_b_upper[i]);
        if steady_e_upper > energy_threshold {
            assert!(
                xfade_e_upper[i] <= 1.5 * steady_e_upper + 1e-6,
                "Crossfade energy spike at sample {i}: xfade={} > 1.5 * steady={}",
                xfade_e_upper[i],
                steady_e_upper
            );
        }
        let steady_e_lower = steady_a_lower[i].min(steady_b_lower[i]);
        if steady_e_lower > energy_threshold && xfade_e_lower[i] > energy_threshold {
            assert!(
                xfade_e_lower[i] >= 0.5 * steady_e_lower - 1e-6,
                "Crossfade energy dip at sample {i}: xfade={} < 0.5 * steady={}",
                xfade_e_lower[i],
                steady_e_lower
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_crossfade_phase_alignment_3500() {
    // Acceptance criterion 3: Prove phase alignment of overlapping regions.
    let input_length = 16;
    let epsilon = 0.05;
    let (network, output_length) = build_synthetic_vocoder(input_length);

    let input = mel_input(input_length, epsilon);
    let crown_output = network
        .propagate_crown(&input)
        .expect("CROWN should succeed");

    let (_first_l, _first_u, last_l, last_u) =
        extract_boundary_bounds(&crown_output, output_length, BOUNDARY_SAMPLES);
    let (first_l_b, first_u_b, _last_l_b, _last_u_b) =
        extract_boundary_bounds(&crown_output, output_length, BOUNDARY_SAMPLES);

    let (xfade_lower, xfade_upper) =
        crossfade_overlap_add_bounds(&last_l, &last_u, &first_l_b, &first_u_b);

    let mid = BOUNDARY_SAMPLES / 2;
    let xfade_width_mid = xfade_upper[mid] - xfade_lower[mid];
    let chunk_a_width_mid = last_u[mid] - last_l[mid];
    let chunk_b_width_mid = first_u_b[mid] - first_l_b[mid];
    let max_chunk_width = chunk_a_width_mid.max(chunk_b_width_mid);

    assert!(
        xfade_width_mid <= max_chunk_width + 1e-5,
        "Phase alignment: crossfade width at midpoint ({}) exceeds max chunk width ({})",
        xfade_width_mid,
        max_chunk_width
    );

    let xfade_width_0 = xfade_upper[0] - xfade_lower[0];
    let chunk_a_width_0 = last_u[0] - last_l[0];
    assert!(
        (xfade_width_0 - chunk_a_width_0).abs() < 1e-5,
        "At i=0, crossfade should equal chunk A: width {} vs {}",
        xfade_width_0,
        chunk_a_width_0
    );

    let last = BOUNDARY_SAMPLES - 1;
    let xfade_width_last = xfade_upper[last] - xfade_lower[last];
    let chunk_a_width_last = last_u[last] - last_l[last];
    let chunk_b_width_last = first_u_b[last] - first_l_b[last];
    let expected_max = chunk_a_width_last / BOUNDARY_SAMPLES as f32 + chunk_b_width_last;
    assert!(
        xfade_width_last <= expected_max + 1e-5,
        "At i=N-1, crossfade width {} should be bounded by {}",
        xfade_width_last,
        expected_max
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_crossfade_overlap_add_convex_combination_3500() {
    let input_length = 16;
    let epsilon = 0.05;
    let (network, output_length) = build_synthetic_vocoder(input_length);

    let input = mel_input(input_length, epsilon);
    let crown_output = network
        .propagate_crown(&input)
        .expect("CROWN should succeed");

    let (_first_l, _first_u, a_last_l, a_last_u) =
        extract_boundary_bounds(&crown_output, output_length, BOUNDARY_SAMPLES);
    let (b_first_l, b_first_u, _b_last_l, _b_last_u) =
        extract_boundary_bounds(&crown_output, output_length, BOUNDARY_SAMPLES);

    let (xfade_lower, xfade_upper) =
        crossfade_overlap_add_bounds(&a_last_l, &a_last_u, &b_first_l, &b_first_u);

    let tol = 1e-5;
    for i in 0..BOUNDARY_SAMPLES {
        let min_lower = a_last_l[i].min(b_first_l[i]);
        let max_upper = a_last_u[i].max(b_first_u[i]);

        assert!(
            xfade_lower[i] >= min_lower - tol,
            "Convex combination lower violated at {i}: {} < {}",
            xfade_lower[i],
            min_lower
        );
        assert!(
            xfade_upper[i] <= max_upper + tol,
            "Convex combination upper violated at {i}: {} > {}",
            xfade_upper[i],
            max_upper
        );
    }
}
