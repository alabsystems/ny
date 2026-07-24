// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Waveform boundary extraction and validation helpers for kokoro vocoder tests.

use super::*;

/// Real avoice crossfade boundary: 240 samples = 10ms at 24kHz.
pub(super) const KOKORO_BOUNDARY_SAMPLES: usize = 240;

/// Boundary window pair: (lower_bounds, upper_bounds) for a contiguous sample range.
pub(super) type BoundaryWindow = (Vec<f32>, Vec<f32>);

/// Crossfade fixture: (first_boundary, last_boundary, flat_output_len).
pub(super) type BoundaryFixture = (BoundaryWindow, BoundaryWindow, usize);

/// Extract boundary sample bounds from a vocoder output tensor.
///
/// Flattens the output and returns (first_N_lower, first_N_upper,
/// last_N_lower, last_N_upper) where N = min(boundary_size, flat_len).
///
/// This mirrors the synthetic `extract_boundary_bounds` in
/// `crates/ny-propagate/src/tests/streaming_boundary.rs:98-127`
/// but operates directly on the flattened output without requiring
/// a pre-known output length, and clamps boundary_size to the actual
/// output length so prefix subgraph outputs (which may be shorter
/// than full-graph waveform outputs) are handled safely.
pub(super) fn extract_waveform_boundary_bounds(
    output: &BoundedTensor,
    boundary_size: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let flat = output.flatten();
    let lower = flat.lower().as_slice().expect("contiguous lower");
    let upper = flat.upper().as_slice().expect("contiguous upper");

    let flat_len = lower.len();
    let n = boundary_size.min(flat_len);
    assert!(
        n > 0,
        "output must have at least one element for boundary extraction"
    );

    let first_lower = lower[..n].to_vec();
    let first_upper = upper[..n].to_vec();
    let last_lower = lower[flat_len - n..].to_vec();
    let last_upper = upper[flat_len - n..].to_vec();

    (first_lower, first_upper, last_lower, last_upper)
}

/// Assert that extracted boundary windows are finite and ordered, and
/// log a summary. Panics on any non-finite or inverted sample.
pub(super) fn assert_boundary_windows_valid(
    first_lower: &[f32],
    first_upper: &[f32],
    last_lower: &[f32],
    last_upper: &[f32],
    flat_len: usize,
    label: &str,
) {
    let n = first_lower.len();
    for (i, ((&fl, &fu), (&ll, &lu))) in first_lower
        .iter()
        .zip(first_upper.iter())
        .zip(last_lower.iter().zip(last_upper.iter()))
        .enumerate()
    {
        assert!(
            fl.is_finite() && fu.is_finite(),
            "{label} first boundary sample {i}: non-finite ({fl}, {fu})"
        );
        assert!(
            fl <= fu,
            "{label} first boundary sample {i}: inverted ({fl} > {fu})"
        );
        assert!(
            ll.is_finite() && lu.is_finite(),
            "{label} last boundary sample {i}: non-finite ({ll}, {lu})"
        );
        assert!(
            ll <= lu,
            "{label} last boundary sample {i}: inverted ({ll} > {lu})"
        );
    }
    eprintln!(
        "{label}: {} boundary samples from {} total \
         (first: [{:.4e}, {:.4e}], last: [{:.4e}, {:.4e}])",
        n,
        flat_len,
        first_lower[0],
        first_upper[0],
        last_lower[n - 1],
        last_upper[n - 1],
    );
}
