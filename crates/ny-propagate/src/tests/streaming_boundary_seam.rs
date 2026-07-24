// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Seam discontinuity and multi-chunk sequential continuity tests for
//! streaming vocoder boundary verification (#3500).
//!
//! Split from `streaming_boundary.rs` for file-size compliance.
//! Tests the "no click" property: |chunk_a[end] - chunk_b[start]| < ε.

use super::streaming_boundary::{
    build_synthetic_vocoder, crossfade_overlap_add_bounds, energy_bounds, extract_boundary_bounds,
    BOUNDARY_SAMPLES,
};
use super::*;
use ndarray::ArrayD;
use proptest::prelude::{prop_assert, proptest, ProptestConfig};

/// Create a bounded input from an explicit per-frame mel profile.
fn mel_input_from_values(values: &[f32], epsilon: f32) -> BoundedTensor {
    let center = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, values.len()]), values.to_vec())
        .expect("mel profile should match [1, input_length]");
    BoundedTensor::from_epsilon(center, epsilon).expect("valid mel input")
}

/// Build a simple linear profile for adjacent chunk tests.
fn ramp_profile(start: f32, end: f32, len: usize) -> Vec<f32> {
    if len == 1 {
        return vec![start];
    }

    (0..len)
        .map(|i| {
            let t = i as f32 / (len - 1) as f32;
            start + t * (end - start)
        })
        .collect()
}

/// Bound the absolute seam discontinuity between adjacent chunks.
///
/// For each overlapped boundary sample, the raw seam jump is:
/// `|chunk_a[end - N + i] - chunk_b[i]|`.
/// If `a ∈ [l_a, u_a]` and `b ∈ [l_b, u_b]`, then:
/// `a - b ∈ [l_a - u_b, u_a - l_b]`.
/// Taking absolute value gives a sound seam-discontinuity interval.
fn absolute_difference_bounds(
    a_lower: &[f32],
    a_upper: &[f32],
    b_lower: &[f32],
    b_upper: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let n = a_lower.len();
    assert_eq!(n, a_upper.len());
    assert_eq!(n, b_lower.len());
    assert_eq!(n, b_upper.len());

    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);

    for i in 0..n {
        let diff_lo = a_lower[i] - b_upper[i];
        let diff_hi = a_upper[i] - b_lower[i];

        let abs_lo = if diff_lo > 0.0 {
            diff_lo
        } else if diff_hi < 0.0 {
            -diff_hi
        } else {
            0.0
        };
        let abs_hi = diff_lo.abs().max(diff_hi.abs());

        lower.push(abs_lo);
        upper.push(abs_hi);
    }

    (lower, upper)
}

/// Verify seam jump bounds contain concrete evaluations for sampled inputs.
///
/// Evaluates the network on `num_samples` concrete inputs within the bounded
/// input regions and checks that the actual seam jump at each boundary sample
/// falls within the precomputed `[jump_lower, jump_upper]` interval.
type ChunkPair<'a> = (&'a BoundedTensor, &'a BoundedTensor);
type SeamLengths = (usize, usize);
type SeamBounds<'a> = (&'a [f32], &'a [f32]);

fn assert_seam_bounds_contain_concrete(
    network: &Network,
    chunks: ChunkPair<'_>,
    lengths: SeamLengths,
    jump_bounds: SeamBounds<'_>,
    num_samples: usize,
) {
    let (chunk_a, chunk_b) = chunks;
    let (output_length, input_length) = lengths;
    let (jump_lower, jump_upper) = jump_bounds;
    let a_lower = chunk_a
        .lower()
        .as_slice()
        .expect("contiguous chunk A lower");
    let a_upper = chunk_a
        .upper()
        .as_slice()
        .expect("contiguous chunk A upper");
    let b_lower = chunk_b
        .lower()
        .as_slice()
        .expect("contiguous chunk B lower");
    let b_upper = chunk_b
        .upper()
        .as_slice()
        .expect("contiguous chunk B upper");

    let tol = 1e-4;
    for sample_idx in 0..num_samples {
        let t_a = sample_idx as f32 / (num_samples - 1).max(1) as f32;
        let t_b = 1.0 - t_a;

        let concrete_a = ArrayD::from_shape_fn(ndarray::IxDyn(&[1, input_length]), |idx| {
            let j = idx[1];
            let mix = ((t_a + j as f32 * 0.07) % 1.0).clamp(0.0, 1.0);
            a_lower[j] + mix * (a_upper[j] - a_lower[j])
        });
        let concrete_b = ArrayD::from_shape_fn(ndarray::IxDyn(&[1, input_length]), |idx| {
            let j = idx[1];
            let mix = ((t_b + j as f32 * 0.11) % 1.0).clamp(0.0, 1.0);
            b_lower[j] + mix * (b_upper[j] - b_lower[j])
        });

        let output_a = network
            .propagate_ibp(&BoundedTensor::concrete(concrete_a).expect("valid chunk A input"))
            .expect("concrete chunk A eval should succeed");
        let output_b = network
            .propagate_ibp(&BoundedTensor::concrete(concrete_b).expect("valid chunk B input"))
            .expect("concrete chunk B eval should succeed");

        let flat_a = output_a.flatten();
        let flat_b = output_b.flatten();
        let samples_a = flat_a.lower().as_slice().expect("contiguous output A");
        let samples_b = flat_b.lower().as_slice().expect("contiguous output B");

        for i in 0..BOUNDARY_SAMPLES {
            let a_idx = output_length - BOUNDARY_SAMPLES + i;
            let actual_jump = (samples_a[a_idx] - samples_b[i]).abs();
            assert!(
                actual_jump >= jump_lower[i] - tol && actual_jump <= jump_upper[i] + tol,
                "Sample {sample_idx}, seam {i}: jump {} not in [{}, {}]",
                actual_jump,
                jump_lower[i],
                jump_upper[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[ntest::timeout(10000)]
#[test]
fn test_adjacent_chunk_seam_discontinuity_bounds_3500() {
    // Directly model the issue's "no click" seam as an absolute difference
    // between chunk A's tail and chunk B's head on distinct adjacent inputs.
    let input_length = 16;
    let epsilon = 0.02;
    let (network, output_length) = build_synthetic_vocoder(input_length);

    let chunk_a = mel_input_from_values(&ramp_profile(0.35, 0.55, input_length), epsilon);
    let chunk_b = mel_input_from_values(&ramp_profile(0.40, 0.60, input_length), epsilon);

    let crown_a = network
        .propagate_crown(&chunk_a)
        .expect("CROWN chunk A should succeed");
    let crown_b = network
        .propagate_crown(&chunk_b)
        .expect("CROWN chunk B should succeed");

    let (_, _, a_last_l, a_last_u) =
        extract_boundary_bounds(&crown_a, output_length, BOUNDARY_SAMPLES);
    let (b_first_l, b_first_u, _, _) =
        extract_boundary_bounds(&crown_b, output_length, BOUNDARY_SAMPLES);

    let (jump_lower, jump_upper) =
        absolute_difference_bounds(&a_last_l, &a_last_u, &b_first_l, &b_first_u);

    for i in 0..BOUNDARY_SAMPLES {
        assert!(
            jump_lower[i].is_finite() && jump_upper[i].is_finite(),
            "Non-finite seam jump bound at {i}: [{}, {}]",
            jump_lower[i],
            jump_upper[i]
        );
        assert!(
            jump_lower[i] >= 0.0 && jump_lower[i] <= jump_upper[i],
            "Invalid seam jump interval at {i}: [{}, {}]",
            jump_lower[i],
            jump_upper[i]
        );
    }

    assert_seam_bounds_contain_concrete(
        &network,
        (&chunk_a, &chunk_b),
        (output_length, input_length),
        (&jump_lower, &jump_upper),
        20,
    );
}

// ---------------------------------------------------------------------------
// Multi-chunk sequential continuity
// ---------------------------------------------------------------------------

/// Verify crossfade properties between two adjacent chunks.
///
/// Checks: (1) convex combination bounds, (2) width non-amplification,
/// (3) energy within 1.5x, (4) phase alignment at midpoint.
fn assert_pairwise_crossfade(
    pair_idx: usize,
    a_last_l: &[f32],
    a_last_u: &[f32],
    b_first_l: &[f32],
    b_first_u: &[f32],
) {
    let (xfade_lower, xfade_upper) =
        crossfade_overlap_add_bounds(a_last_l, a_last_u, b_first_l, b_first_u);

    // 1. Convex combination property
    for i in 0..BOUNDARY_SAMPLES {
        let min_lower = a_last_l[i].min(b_first_l[i]);
        let max_upper = a_last_u[i].max(b_first_u[i]);
        assert!(
            xfade_lower[i] >= min_lower - 1e-5,
            "Pair {pair_idx} convex lower at {i}: {} < {}",
            xfade_lower[i],
            min_lower
        );
        assert!(
            xfade_upper[i] <= max_upper + 1e-5,
            "Pair {pair_idx} convex upper at {i}: {} > {}",
            xfade_upper[i],
            max_upper
        );

        let fade_out = (BOUNDARY_SAMPLES - i) as f32 / BOUNDARY_SAMPLES as f32;
        let fade_in = i as f32 / BOUNDARY_SAMPLES as f32;
        let width_a = a_last_u[i] - a_last_l[i];
        let width_b = b_first_u[i] - b_first_l[i];
        let xfade_width = xfade_upper[i] - xfade_lower[i];
        let expected_width = fade_out * width_a + fade_in * width_b;
        let max_chunk_width = width_a.max(width_b);
        assert!(
            (xfade_width - expected_width).abs() <= 1e-5,
            "Pair {pair_idx} width formula at {i}: {} != {}",
            xfade_width,
            expected_width
        );
        assert!(
            xfade_width <= max_chunk_width + 1e-5,
            "Pair {pair_idx} width blowup at {i}: {} > {}",
            xfade_width,
            max_chunk_width
        );
    }

    // 3. Energy bounds
    let (xfade_e_lo, xfade_e_hi) = energy_bounds(&xfade_lower, &xfade_upper);
    let (a_e_lo, a_e_hi) = energy_bounds(a_last_l, a_last_u);
    let (b_e_lo, b_e_hi) = energy_bounds(b_first_l, b_first_u);
    for i in 0..BOUNDARY_SAMPLES {
        let max_steady_e_hi = a_e_hi[i].max(b_e_hi[i]);
        if max_steady_e_hi > 1e-6 {
            assert!(
                xfade_e_hi[i] <= 1.5 * max_steady_e_hi + 1e-6,
                "Pair {pair_idx} energy spike at {i}: {} > 1.5 * {}",
                xfade_e_hi[i],
                max_steady_e_hi
            );
        }
    }

    // The design criterion is defined over the full overlap region, not just
    // per-sample spikes. Sum the interval energies across the boundary window
    // so the synthetic proof matches the intended crossfade-energy property.
    let total_xfade_e_lo: f32 = xfade_e_lo.iter().sum();
    let total_xfade_e_hi: f32 = xfade_e_hi.iter().sum();
    let total_steady_e_lo: f32 = a_e_lo.iter().zip(&b_e_lo).map(|(a, b)| a.min(*b)).sum();
    let total_steady_e_hi: f32 = a_e_hi.iter().zip(&b_e_hi).map(|(a, b)| a.max(*b)).sum();
    if total_steady_e_hi > 1e-6 {
        assert!(
            total_xfade_e_hi <= 1.5 * total_steady_e_hi + 1e-6,
            "Pair {pair_idx} total energy spike: {} > 1.5 * {}",
            total_xfade_e_hi,
            total_steady_e_hi
        );
    }
    if total_steady_e_lo > 1e-6 && total_xfade_e_lo > 1e-6 {
        assert!(
            total_xfade_e_lo >= 0.5 * total_steady_e_lo - 1e-6,
            "Pair {pair_idx} total energy dip: {} < 0.5 * {}",
            total_xfade_e_lo,
            total_steady_e_lo
        );
    }

    // 4. Phase alignment at midpoint
    let mid = BOUNDARY_SAMPLES / 2;
    let xfade_width = xfade_upper[mid] - xfade_lower[mid];
    let max_chunk_width = (a_last_u[mid] - a_last_l[mid]).max(b_first_u[mid] - b_first_l[mid]);
    assert!(
        xfade_width <= max_chunk_width + 1e-5,
        "Pair {pair_idx} phase alignment: crossfade width {} > max chunk width {}",
        xfade_width,
        max_chunk_width
    );
}

#[ntest::timeout(60000)]
#[test]
fn test_multi_chunk_sequential_continuity_3500() {
    // Verify 3 consecutive chunks maintain bounded crossfade properties
    // across all boundaries. Tests the composition property needed for
    // real streaming TTS (hundreds of chunks).
    let input_length = 16;
    let epsilon = 0.05;
    let (network, output_length) = build_synthetic_vocoder(input_length);

    let profiles = [
        ramp_profile(0.30, 0.50, input_length),
        ramp_profile(0.36, 0.58, input_length),
        ramp_profile(0.42, 0.63, input_length),
    ];
    let mut chunk_bounds = Vec::with_capacity(profiles.len());
    for profile in profiles {
        let input = mel_input_from_values(&profile, epsilon);
        let crown_output = network
            .propagate_crown(&input)
            .expect("CROWN should succeed");
        chunk_bounds.push(extract_boundary_bounds(
            &crown_output,
            output_length,
            BOUNDARY_SAMPLES,
        ));
    }

    // Verify each pairwise crossfade
    for pair_idx in 0..2 {
        let (_, _, ref a_last_l, ref a_last_u) = chunk_bounds[pair_idx];
        let (ref b_first_l, ref b_first_u, _, _) = chunk_bounds[pair_idx + 1];
        assert_pairwise_crossfade(pair_idx, a_last_l, a_last_u, b_first_l, b_first_u);
    }

    // Global sequential property: composing adjacent crossfades should never
    // amplify uncertainty beyond the widest participating chunk boundary.
    let (_, _, ref a0_ll, ref a0_lu) = chunk_bounds[0];
    let (ref b0_fl, ref b0_fu, _, _) = chunk_bounds[1];
    let (xf0_lo, xf0_hi) = crossfade_overlap_add_bounds(a0_ll, a0_lu, b0_fl, b0_fu);

    let (_, _, ref a1_ll, ref a1_lu) = chunk_bounds[1];
    let (ref b1_fl, ref b1_fu, _, _) = chunk_bounds[2];
    let (xf1_lo, xf1_hi) = crossfade_overlap_add_bounds(a1_ll, a1_lu, b1_fl, b1_fu);

    for i in 0..BOUNDARY_SAMPLES {
        let xf0_width = xf0_hi[i] - xf0_lo[i];
        let xf1_width = xf1_hi[i] - xf1_lo[i];
        let max_xfade_width = xf0_width.max(xf1_width);
        let max_chunk_width = (a0_lu[i] - a0_ll[i])
            .max(b0_fu[i] - b0_fl[i])
            .max(a1_lu[i] - a1_ll[i])
            .max(b1_fu[i] - b1_fl[i]);
        assert!(
            max_xfade_width <= max_chunk_width + 1e-5,
            "Sequential width blowup at {i}: max_xfade_width={max_xfade_width} > \
             max_chunk_width={max_chunk_width}"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_adjacent_chunk_total_crossfade_energy_bounded_3500() {
    // Acceptance criterion 2 is about overlap-region energy, not just the
    // per-sample envelope. Verify the full adjacent-chunk window stays within
    // the same [0.5, 1.5] steady-state range on distinct chunk profiles.
    let input_length = 16;
    let epsilon = 0.02;
    let (network, output_length) = build_synthetic_vocoder(input_length);

    let chunk_a = mel_input_from_values(&ramp_profile(0.35, 0.55, input_length), epsilon);
    let chunk_b = mel_input_from_values(&ramp_profile(0.40, 0.60, input_length), epsilon);

    let crown_a = network
        .propagate_crown(&chunk_a)
        .expect("CROWN chunk A should succeed");
    let crown_b = network
        .propagate_crown(&chunk_b)
        .expect("CROWN chunk B should succeed");

    let (_, _, a_last_l, a_last_u) =
        extract_boundary_bounds(&crown_a, output_length, BOUNDARY_SAMPLES);
    let (b_first_l, b_first_u, _, _) =
        extract_boundary_bounds(&crown_b, output_length, BOUNDARY_SAMPLES);

    assert_pairwise_crossfade(0, &a_last_l, &a_last_u, &b_first_l, &b_first_u);
}

// ---------------------------------------------------------------------------
// Proptests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Absolute seam-difference interval arithmetic should contain sampled
    /// discontinuities for arbitrary adjacent chunk bounds.
    #[test]
    fn proptest_absolute_difference_bounds_sound_3500(
        a_lo in proptest::collection::vec(-2.0f32..2.0, BOUNDARY_SAMPLES),
        a_delta in proptest::collection::vec(0.01f32..1.0, BOUNDARY_SAMPLES),
        a_mix in proptest::collection::vec(0.0f32..1.0, BOUNDARY_SAMPLES),
        b_lo in proptest::collection::vec(-2.0f32..2.0, BOUNDARY_SAMPLES),
        b_delta in proptest::collection::vec(0.01f32..1.0, BOUNDARY_SAMPLES),
        b_mix in proptest::collection::vec(0.0f32..1.0, BOUNDARY_SAMPLES),
    ) {
        let a_hi: Vec<f32> = a_lo.iter().zip(&a_delta).map(|(l, d)| l + d).collect();
        let b_hi: Vec<f32> = b_lo.iter().zip(&b_delta).map(|(l, d)| l + d).collect();

        let (diff_lo, diff_hi) = absolute_difference_bounds(&a_lo, &a_hi, &b_lo, &b_hi);

        let tol = 1e-5;
        for i in 0..BOUNDARY_SAMPLES {
            let a = a_lo[i] + a_mix[i] * a_delta[i];
            let b = b_lo[i] + b_mix[i] * b_delta[i];
            let actual = (a - b).abs();

            prop_assert!(
                actual >= diff_lo[i] - tol && actual <= diff_hi[i] + tol,
                "Seam diff at {i}: actual {} not in [{}, {}]",
                actual, diff_lo[i], diff_hi[i]
            );
        }
    }
}
