// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::ArrayD;

#[ntest::timeout(10000)]
#[test]
fn test_vocoder_boundary_ibp_bounds_3500() {
    // Acceptance criterion 1: CROWN bounds on vocoder boundary samples.
    // Start with IBP as a baseline, then verify CROWN tightens.
    let input_length = 16;
    let epsilon = 0.05;
    let (network, output_length) = build_synthetic_vocoder(input_length);
    let input = mel_input(input_length, epsilon);

    let ibp_output = network.propagate_ibp(&input).expect("IBP should succeed");
    let (first_l, first_u, last_l, last_u) =
        extract_boundary_bounds(&ibp_output, output_length, BOUNDARY_SAMPLES);

    for i in 0..BOUNDARY_SAMPLES {
        assert!(
            first_l[i].is_finite() && first_u[i].is_finite(),
            "IBP boundary sample {i} is non-finite: [{}, {}]",
            first_l[i],
            first_u[i]
        );
        assert!(
            first_l[i] <= first_u[i],
            "IBP first boundary sample {i} inverted: {} > {}",
            first_l[i],
            first_u[i]
        );
        assert!(
            last_l[i].is_finite() && last_u[i].is_finite(),
            "IBP last boundary sample {i} is non-finite: [{}, {}]",
            last_l[i],
            last_u[i]
        );
        assert!(
            last_l[i] <= last_u[i],
            "IBP last boundary sample {i} inverted: {} > {}",
            last_l[i],
            last_u[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_vocoder_boundary_crown_tighter_than_ibp_3500() {
    let input_length = 16;
    let epsilon = 0.05;
    let (network, output_length) = build_synthetic_vocoder(input_length);
    let input = mel_input(input_length, epsilon);

    let ibp_output = network.propagate_ibp(&input).expect("IBP should succeed");
    let crown_output = network
        .propagate_crown(&input)
        .expect("CROWN should succeed");

    let (ibp_first_l, ibp_first_u, ibp_last_l, ibp_last_u) =
        extract_boundary_bounds(&ibp_output, output_length, BOUNDARY_SAMPLES);
    let (crown_first_l, crown_first_u, crown_last_l, crown_last_u) =
        extract_boundary_bounds(&crown_output, output_length, BOUNDARY_SAMPLES);

    let tol = 1e-4;
    for i in 0..BOUNDARY_SAMPLES {
        assert!(
            crown_first_l[i] >= ibp_first_l[i] - tol,
            "First boundary {i}: CROWN lower {} < IBP lower {}",
            crown_first_l[i],
            ibp_first_l[i]
        );
        assert!(
            crown_first_u[i] <= ibp_first_u[i] + tol,
            "First boundary {i}: CROWN upper {} > IBP upper {}",
            crown_first_u[i],
            ibp_first_u[i]
        );
        assert!(
            crown_last_l[i] >= ibp_last_l[i] - tol,
            "Last boundary {i}: CROWN lower {} < IBP lower {}",
            crown_last_l[i],
            ibp_last_l[i]
        );
        assert!(
            crown_last_u[i] <= ibp_last_u[i] + tol,
            "Last boundary {i}: CROWN upper {} > IBP upper {}",
            crown_last_u[i],
            ibp_last_u[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_vocoder_boundary_soundness_sampling_3500() {
    let input_length = 16;
    let epsilon = 0.05;
    let (network, output_length) = build_synthetic_vocoder(input_length);
    let input = mel_input(input_length, epsilon);

    let crown_output = network
        .propagate_crown(&input)
        .expect("CROWN should succeed");
    let (first_l, first_u, last_l, last_u) =
        extract_boundary_bounds(&crown_output, output_length, BOUNDARY_SAMPLES);

    let lower_arr = input.lower().as_slice().expect("contiguous").to_vec();
    let upper_arr = input.upper().as_slice().expect("contiguous").to_vec();
    let input_dim = lower_arr.len();
    let tol = 1e-4;

    for sample_idx in 0..20 {
        let t = sample_idx as f32 / 19.0;
        let mut concrete_input = ArrayD::zeros(input.lower().raw_dim());
        for j in 0..input_dim {
            let t_j = ((t + j as f32 * 0.1) % 1.0).clamp(0.0, 1.0);
            concrete_input.as_slice_mut().expect("contiguous")[j] =
                lower_arr[j] + t_j * (upper_arr[j] - lower_arr[j]);
        }

        let concrete_bt = BoundedTensor::concrete(concrete_input).expect("valid concrete input");
        let concrete_output = network
            .propagate_ibp(&concrete_bt)
            .expect("concrete eval should succeed");

        let flat = concrete_output.flatten();
        let out = flat.lower().as_slice().expect("contiguous");

        for i in 0..BOUNDARY_SAMPLES {
            assert!(
                out[i] >= first_l[i] - tol && out[i] <= first_u[i] + tol,
                "Sample {sample_idx}, first boundary {i}: output {} not in [{}, {}]",
                out[i],
                first_l[i],
                first_u[i]
            );
        }
        for i in 0..BOUNDARY_SAMPLES {
            let idx = output_length - BOUNDARY_SAMPLES + i;
            assert!(
                out[idx] >= last_l[i] - tol && out[idx] <= last_u[i] + tol,
                "Sample {sample_idx}, last boundary {i}: output {} not in [{}, {}]",
                out[idx],
                last_l[i],
                last_u[i]
            );
        }
    }
}
