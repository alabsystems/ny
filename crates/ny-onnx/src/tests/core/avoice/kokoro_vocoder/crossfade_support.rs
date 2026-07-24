// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cached boundary fixtures shared by crossfade and seam test lanes (#3500).

use super::super::common::assert_finite_and_ordered;
use super::boundary::{
    extract_waveform_boundary_bounds, BoundaryFixture, BoundaryWindow, KOKORO_BOUNDARY_SAMPLES,
};
use super::graph_support::{first_conv_transpose_node, vocoder_prefix_subgraph};
use super::model::{
    bounded_kokoro_features_input_centered, load_kokoro_vocoder_with_fixed_aux,
    KOKORO_VOCODER_MIN_FIXED_AUX_T,
};
use super::*;

/// Build prefix IBP boundary fixtures centered at a given feature value.
///
/// Different centers simulate different audio content in adjacent streaming
/// chunks. The vocoder's real weights produce different output distributions
/// for different input feature centers.
pub(super) fn prefix_ibp_boundary_fixtures_centered(center_value: f32) -> BoundaryFixture {
    let model = load_kokoro_vocoder_with_fixed_aux(KOKORO_VOCODER_MIN_FIXED_AUX_T);
    let graph = model
        .to_graph_network()
        .expect("graph conversion should succeed");
    let cut_node = first_conv_transpose_node(&graph);
    let prefix = vocoder_prefix_subgraph(&graph, &cut_node);
    let input = bounded_kokoro_features_input_centered(
        &model,
        KOKORO_VOCODER_MIN_FIXED_AUX_T,
        center_value,
        1e-3,
    );

    let output = prefix
        .propagate_ibp(&input)
        .expect("prefix IBP should complete");
    assert_finite_and_ordered(&output, "prefix IBP boundary fixture");

    let flat_len = output.lower().len();
    let (first_lower, first_upper, last_lower, last_upper) =
        extract_waveform_boundary_bounds(&output, KOKORO_BOUNDARY_SAMPLES);

    (
        (first_lower, first_upper),
        (last_lower, last_upper),
        flat_len,
    )
}

/// Build two-chunk boundary fixtures with different feature centers.
///
/// Returns `(chunk_A_last, chunk_B_first, flat_len)` where:
///
/// - chunk_A boundaries (center=0.0): last N samples (outgoing chunk)
/// - chunk_B boundaries (center=0.1): first N samples (incoming chunk)
/// - flat_len from chunk_A for diagnostics
pub(super) fn two_chunk_boundary_fixtures() -> (BoundaryWindow, BoundaryWindow, usize) {
    static FIXTURE: OnceLock<(BoundaryWindow, BoundaryWindow, usize)> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let ((_a_first_lo, _a_first_hi), (a_last_lo, a_last_hi), flat_len) =
                prefix_ibp_boundary_fixtures_centered(0.0);
            let ((b_first_lo, b_first_hi), (_b_last_lo, _b_last_hi), _) =
                prefix_ibp_boundary_fixtures_centered(0.1);

            ((a_last_lo, a_last_hi), (b_first_lo, b_first_hi), flat_len)
        })
        .clone()
}
