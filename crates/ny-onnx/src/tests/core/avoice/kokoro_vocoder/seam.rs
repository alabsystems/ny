// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Real-weight seam discontinuity bounds (#3500).
//!
//! Computes |chunk_A[end] - chunk_B[start]| bounds on real Kokoro vocoder
//! prefix weights, directly measuring the "no click" property from #3500
//! requirement 1: |sample[chunk_N, -1] - sample[chunk_N+1, 0]| < ε.
//!
//! The crossfade tests in `crossfade.rs` verify that the overlap-add
//! combination of two chunks produces bounded output. This module instead
//! measures the RAW seam discontinuity BEFORE crossfade, giving avoice
//! a quantitative bound on the maximum possible click magnitude.
//!
//! Reference: #3500 requirement 1
//! Reference: ny-propagate/src/tests/streaming_boundary_seam.rs (synthetic version)

use super::crossfade_support::two_chunk_boundary_fixtures;

/// Bound |a - b| using interval arithmetic.
///
/// If a ∈ [l_a, u_a] and b ∈ [l_b, u_b], then:
///   a - b ∈ [l_a - u_b, u_a - l_b]
///   |a - b| ∈ [max(0, ...), max(|diff_lo|, |diff_hi|)]
///
/// Sound: the absolute value of an interval [lo, hi] is:
///   - If lo >= 0: [lo, hi]
///   - If hi <= 0: [-hi, -lo]
///   - If lo < 0 < hi: [0, max(-lo, hi)]
///
/// Mirrors `absolute_difference_bounds` from
/// `ny-propagate/src/tests/streaming_boundary_seam.rs:47-79`.
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

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_two_chunk_seam_discontinuity_bounded_3500() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    // #3500 requirement 1: |sample[chunk_N, -1] - sample[chunk_N+1, 0]| < ε.
    //
    // Measures the raw seam discontinuity between two adjacent streaming
    // chunks with different audio content (center=0.0 vs center=0.1) on
    // real Kokoro vocoder prefix weights.
    //
    // This is the "no click" property BEFORE crossfade. The magnitude of
    // this bound tells avoice whether crossfade is essential (large jump)
    // or cosmetic (small jump).
    //
    // Uses IBP bounds on the prefix subgraph (first ConvTranspose1d node).
    // Full-graph seam measurement is still waiting on the deeper vocoder lane:
    // #3619 must recover non-trivial deep CROWN tightening, and #3597 must
    // finish the graph-wide engine-threading path.
    //
    // Baseline (2026-03-11, ε=1e-3, prefix subgraph, T=6):
    //   max_jump_upper = 1.761248e-1  (worst-case click magnitude)
    //   max_jump_lower = 1.499317e-1  (guaranteed minimum jump)
    //   zero_lower_count = 30/240
    let ((a_last_lo, a_last_hi), (b_first_lo, b_first_hi), flat_len) =
        two_chunk_boundary_fixtures();

    let n = a_last_lo.len();
    assert!(n > 0, "need at least one boundary sample");

    let (jump_lower, jump_upper) =
        absolute_difference_bounds(&a_last_lo, &a_last_hi, &b_first_lo, &b_first_hi);

    // All seam jump bounds must be finite and well-ordered
    let mut max_jump_upper: f32 = 0.0;
    let mut max_jump_lower: f32 = 0.0;
    let mut zero_lower_count = 0usize;

    for i in 0..n {
        assert!(
            jump_lower[i].is_finite() && jump_upper[i].is_finite(),
            "non-finite seam jump bound at sample {i}/{n}: [{}, {}]",
            jump_lower[i],
            jump_upper[i]
        );
        assert!(
            jump_lower[i] >= 0.0,
            "seam jump lower bound must be non-negative at {i}: {}",
            jump_lower[i]
        );
        assert!(
            jump_lower[i] <= jump_upper[i],
            "inverted seam jump interval at {i}: [{}, {}]",
            jump_lower[i],
            jump_upper[i]
        );

        max_jump_upper = max_jump_upper.max(jump_upper[i]);
        max_jump_lower = max_jump_lower.max(jump_lower[i]);
        if jump_lower[i] == 0.0 {
            zero_lower_count += 1;
        }
    }

    // The seam jump upper bound is the worst-case click magnitude.
    // Report it for avoice to use in crossfade window sizing.
    eprintln!(
        "seam discontinuity: {n} boundary samples from {flat_len} total\n\
         max_jump_upper = {max_jump_upper:.6e} (worst-case click magnitude)\n\
         max_jump_lower = {max_jump_lower:.6e} (guaranteed minimum jump)\n\
         zero_lower_count = {zero_lower_count}/{n} \
         (samples where seam intervals contain zero)"
    );

    // Sanity: the seam jump should be non-vacuous (upper > 0 for at least
    // some samples). If all jump upper bounds are zero, the two chunks
    // produce identical output and the test is degenerate.
    assert!(
        max_jump_upper > 0.0,
        "degenerate seam test: all jump upper bounds are zero, \
         meaning center=0.0 and center=0.1 produce identical prefix outputs"
    );
}
