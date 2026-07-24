// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use proptest::prelude::{prop_assert, proptest, ProptestConfig};

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    #[ntest::timeout(10000)]
    #[test]
    fn proptest_vocoder_boundary_crown_soundness_3500(
        concrete in concrete_input_strategy(16, 0.05),
    ) {
        let input_length = 16;
        let epsilon = 0.05;
        let (network, output_length) = build_synthetic_vocoder(input_length);
        let bounded_input = mel_input(input_length, epsilon);

        let crown_output = network
            .propagate_crown(&bounded_input)
            .map_err(|e| proptest::test_runner::TestCaseError::fail(
                format!("CROWN propagation failed: {e}")
            ))?;

        let (first_l, first_u, last_l, last_u) =
            extract_boundary_bounds(&crown_output, output_length, BOUNDARY_SAMPLES);

        let concrete_bt = BoundedTensor::concrete(concrete)
            .map_err(|e| proptest::test_runner::TestCaseError::fail(
                format!("concrete BoundedTensor failed: {e}")
            ))?;
        let concrete_output = network
            .propagate_ibp(&concrete_bt)
            .map_err(|e| proptest::test_runner::TestCaseError::fail(
                format!("concrete propagation failed: {e}")
            ))?;

        let flat = concrete_output.flatten();
        let out = flat.lower().as_slice().expect("contiguous");
        let tol = 1e-4;

        for i in 0..BOUNDARY_SAMPLES {
            prop_assert!(
                out[i] >= first_l[i] - tol && out[i] <= first_u[i] + tol,
                "First boundary {i}: output {} not in [{}, {}]",
                out[i], first_l[i], first_u[i]
            );
        }
        for i in 0..BOUNDARY_SAMPLES {
            let idx = output_length - BOUNDARY_SAMPLES + i;
            prop_assert!(
                out[idx] >= last_l[i] - tol && out[idx] <= last_u[i] + tol,
                "Last boundary {i}: output {} not in [{}, {}]",
                out[idx], last_l[i], last_u[i]
            );
        }
    }

    #[test]
    fn proptest_crossfade_convex_combination_3500(
        a_lo in proptest::collection::vec(-2.0f32..2.0, BOUNDARY_SAMPLES),
        a_delta in proptest::collection::vec(0.01f32..1.0, BOUNDARY_SAMPLES),
        b_lo in proptest::collection::vec(-2.0f32..2.0, BOUNDARY_SAMPLES),
        b_delta in proptest::collection::vec(0.01f32..1.0, BOUNDARY_SAMPLES),
    ) {
        let a_hi: Vec<f32> = a_lo.iter().zip(&a_delta).map(|(l, d)| l + d).collect();
        let b_hi: Vec<f32> = b_lo.iter().zip(&b_delta).map(|(l, d)| l + d).collect();

        let (xfade_lower, xfade_upper) =
            crossfade_overlap_add_bounds(&a_lo, &a_hi, &b_lo, &b_hi);

        let tol = 1e-5;
        for i in 0..BOUNDARY_SAMPLES {
            let min_lower = a_lo[i].min(b_lo[i]);
            let max_upper = a_hi[i].max(b_hi[i]);

            prop_assert!(
                xfade_lower[i] >= min_lower - tol,
                "Convex lower at {i}: xfade_lower={} < min_lower={}",
                xfade_lower[i], min_lower
            );
            prop_assert!(
                xfade_upper[i] <= max_upper + tol,
                "Convex upper at {i}: xfade_upper={} > max_upper={}",
                xfade_upper[i], max_upper
            );
        }
    }

    #[test]
    fn proptest_crossfade_width_non_amplifying_3500(
        a_lo in proptest::collection::vec(-2.0f32..2.0, BOUNDARY_SAMPLES),
        a_delta in proptest::collection::vec(0.01f32..1.0, BOUNDARY_SAMPLES),
        b_lo in proptest::collection::vec(-2.0f32..2.0, BOUNDARY_SAMPLES),
        b_delta in proptest::collection::vec(0.01f32..1.0, BOUNDARY_SAMPLES),
    ) {
        let a_hi: Vec<f32> = a_lo.iter().zip(&a_delta).map(|(l, d)| l + d).collect();
        let b_hi: Vec<f32> = b_lo.iter().zip(&b_delta).map(|(l, d)| l + d).collect();

        let (xfade_lower, xfade_upper) =
            crossfade_overlap_add_bounds(&a_lo, &a_hi, &b_lo, &b_hi);

        for i in 0..BOUNDARY_SAMPLES {
            let fade_out = (BOUNDARY_SAMPLES - i) as f32 / BOUNDARY_SAMPLES as f32;
            let fade_in = i as f32 / BOUNDARY_SAMPLES as f32;
            let width_a = a_hi[i] - a_lo[i];
            let width_b = b_hi[i] - b_lo[i];
            let xfade_width = xfade_upper[i] - xfade_lower[i];
            let expected_width = fade_out * width_a + fade_in * width_b;

            prop_assert!(
                (xfade_width - expected_width).abs() <= 1e-5,
                "Width formula at {i}: xfade_width={} != expected_width={}",
                xfade_width, expected_width
            );
            prop_assert!(
                xfade_width <= width_a.max(width_b) + 1e-5,
                "Width blowup at {i}: xfade_width={} > max_chunk_width={}",
                xfade_width, width_a.max(width_b)
            );
        }
    }

    #[test]
    fn proptest_crossfade_energy_bounded_3500(
        a_lo in proptest::collection::vec(-1.0f32..1.0, BOUNDARY_SAMPLES),
        a_delta in proptest::collection::vec(0.01f32..0.5, BOUNDARY_SAMPLES),
        b_lo in proptest::collection::vec(-1.0f32..1.0, BOUNDARY_SAMPLES),
        b_delta in proptest::collection::vec(0.01f32..0.5, BOUNDARY_SAMPLES),
    ) {
        let a_hi: Vec<f32> = a_lo.iter().zip(&a_delta).map(|(l, d)| l + d).collect();
        let b_hi: Vec<f32> = b_lo.iter().zip(&b_delta).map(|(l, d)| l + d).collect();

        let (xfade_lower, xfade_upper) =
            crossfade_overlap_add_bounds(&a_lo, &a_hi, &b_lo, &b_hi);
        let (xfade_e_lo, xfade_e_hi) = energy_bounds(&xfade_lower, &xfade_upper);
        let (_, a_e_hi) = energy_bounds(&a_lo, &a_hi);
        let (_, b_e_hi) = energy_bounds(&b_lo, &b_hi);

        let threshold = 1e-6;
        for i in 0..BOUNDARY_SAMPLES {
            let max_e_hi = a_e_hi[i].max(b_e_hi[i]);
            if max_e_hi > threshold {
                prop_assert!(
                    xfade_e_hi[i] <= 1.5 * max_e_hi + 1e-6,
                    "Energy spike at {i}: xfade_e={} > 1.5 * max_chunk_e={}",
                    xfade_e_hi[i], max_e_hi
                );
            }
            prop_assert!(
                xfade_e_lo[i] >= -1e-6,
                "Negative energy lower at {i}: {}",
                xfade_e_lo[i]
            );
        }
    }
}
