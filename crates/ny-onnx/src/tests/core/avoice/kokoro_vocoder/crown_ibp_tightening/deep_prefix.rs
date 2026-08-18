// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::common::assert_finite_and_ordered;
use super::super::boundary::extract_waveform_boundary_bounds;
use super::super::graph_support::vocoder_prefix_subgraph;
use super::super::model::{
    bounded_kokoro_features_input, load_kokoro_vocoder_with_fixed_aux,
    KOKORO_VOCODER_MIN_FIXED_AUX_T,
};
use super::prefix::first_conv1d_after_conv_transpose;
use super::support::{
    assert_crown_no_looser, assert_finite_and_ordered_slices, boundary_spec_matrix,
    with_kokoro_crown_lock, PrefixCrownFixture,
};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Deep-prefix spec-guided CROWN (#3500)
//
// The 2-node (ConvTranspose) prefix produces CROWN == IBP (0 tighter) because
// the backward pass has minimal room to exploit linear relaxation. The 12-node
// prefix (through first Conv1d + ResBlock cycle) amplifies IBP widths by ~10^4x,
// creating meaningful room for CROWN tightening. This smoke test verifies that
// CROWN backward propagates correctly through the deeper prefix and measures
// whether the ResBlock-amplified bounds can be tightened.
//
// Budget: IBP node bounds ~79s + 4 boundary specs backward ~20-60s = ~100-140s.
// ---------------------------------------------------------------------------

const DEEP_PREFIX_BOUNDARY_SPECS: usize = 4;
const DEEP_PREFIX_CROWN_DEADLINE_SECS: u64 = 180;

fn build_deep_prefix_crown_fixture() -> PrefixCrownFixture {
    let model = load_kokoro_vocoder_with_fixed_aux(KOKORO_VOCODER_MIN_FIXED_AUX_T);
    let graph = model
        .to_graph_network()
        .expect("graph conversion should succeed");
    let cut_node = first_conv1d_after_conv_transpose(&graph);
    let prefix = vocoder_prefix_subgraph(&graph, &cut_node);
    let input = bounded_kokoro_features_input(&model, KOKORO_VOCODER_MIN_FIXED_AUX_T, 1e-3);

    let ibp_output = prefix
        .propagate_ibp(&input)
        .expect("deep prefix IBP should succeed");
    assert_finite_and_ordered(&ibp_output, "deep prefix IBP");

    let ibp_node_bounds = prefix
        .collect_node_bounds(&input)
        .expect("deep prefix IBP node-bound collection should succeed");

    PrefixCrownFixture {
        prefix,
        input,
        ibp_output,
        ibp_node_bounds,
    }
}

/// Spec-guided CROWN on the 12-node deep prefix (through first Conv1d after
/// the first ConvTranspose1d). Tests that CROWN backward propagates correctly
/// through the ResBlock cycle (Conv1d + InstanceNorm + SnakeActivation).
///
/// Unlike the 2-node prefix where CROWN == IBP, the deep prefix should show
/// meaningful tightening because the ResBlock amplifies IBP widths by ~10^4x,
/// giving CROWN more room to exploit linear relaxations.
///
/// Budget: ~90-120s graph const-fold (har=[22,61]) + ~80s IBP node bounds +
/// ~20-60s CROWN backward = ~190-260s.  600s accommodates variance.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_spec_guided_crown_deep_prefix_boundary_smoke_3500() {
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    use ny_core::NaiveCpuGemmEngine;

    with_kokoro_crown_lock(|| {
        let t0 = Instant::now();
        let f = build_deep_prefix_crown_fixture();
        let ibp_elapsed = t0.elapsed().as_secs_f64();
        let flat_len = f.ibp_output.lower().len();

        eprintln!(
            "deep prefix ({} nodes, {} output elements): IBP in {:.1}s",
            f.prefix.num_nodes(),
            flat_len,
            ibp_elapsed,
        );

        let spec = boundary_spec_matrix(flat_len, DEEP_PREFIX_BOUNDARY_SPECS);
        let deadline = Instant::now() + Duration::from_secs(DEEP_PREFIX_CROWN_DEADLINE_SECS);

        let t1 = Instant::now();
        let result = f
            .prefix
            .propagate_crown_with_specs_and_engine_with_node_bounds_and_deadline(
                &f.input,
                &spec,
                Some(&NaiveCpuGemmEngine),
                &f.ibp_node_bounds,
                Some(deadline),
            )
            .expect("deep prefix spec-guided CROWN should succeed");
        let crown_elapsed = t1.elapsed().as_secs_f64();

        let flat = result.flatten();
        let crown_lo: Vec<f32> = flat.lower().iter().copied().collect();
        let crown_hi: Vec<f32> = flat.upper().iter().copied().collect();
        assert_finite_and_ordered_slices(&crown_lo, &crown_hi, "deep prefix CROWN");

        let (fl, fu, ll, lu) =
            extract_waveform_boundary_bounds(&f.ibp_output, DEEP_PREFIX_BOUNDARY_SPECS);
        let (tighter, equal) = assert_crown_no_looser(&crown_lo, &crown_hi, &fl, &fu, &ll, &lu);
        eprintln!(
            "deep prefix CROWN: {} specs in {:.1}s ({tighter} tighter, {equal} equal)",
            spec.nrows(),
            crown_elapsed,
        );
    });
}
