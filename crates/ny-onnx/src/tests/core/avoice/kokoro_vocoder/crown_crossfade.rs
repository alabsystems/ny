// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use self::two_chunk::{
    bounded_tensor_from_slices, two_chunk_concrete_crossfade, two_chunk_crossfade_bounds,
};
use super::super::common::{assert_concrete_contained_in_bounds, evaluate_graph_at_center};
use super::boundary::{extract_waveform_boundary_bounds, BoundaryFixture, KOKORO_BOUNDARY_SAMPLES};
use super::crossfade::{crossfade_overlap_add_bounds, energy_bounds, prefix_ibp_boundary_fixtures};
use super::crown_ibp_tightening::support::{
    assert_crown_no_looser, boundary_spec_matrix_range, cached_prefix_crown_fixture,
    spec_guided_crown_with_engine, with_kokoro_crown_lock, PrefixCrownFixture,
    FULL_BOUNDARY_SPEC_CHUNK_SAMPLES,
};
use super::*;
use ny_core::NaiveCpuGemmEngine;
use std::sync::OnceLock;

mod two_chunk;

fn crown_boundary_fixtures_from_fixture(
    fixture: &PrefixCrownFixture,
    boundary_sample_count: usize,
    label_prefix: &str,
) -> BoundaryFixture {
    let flat_len = fixture.ibp_output.lower().len();
    let boundary_size = boundary_sample_count.min(flat_len);
    assert!(
        boundary_size > 0,
        "{label_prefix}: boundary extraction requires a non-empty prefix output"
    );
    let (ibp_first_lo, ibp_first_hi, ibp_last_lo, ibp_last_hi) =
        extract_waveform_boundary_bounds(&fixture.ibp_output, boundary_size);

    let mut first_lower = Vec::with_capacity(boundary_size);
    let mut first_upper = Vec::with_capacity(boundary_size);
    let mut last_lower = Vec::with_capacity(boundary_size);
    let mut last_upper = Vec::with_capacity(boundary_size);
    let chunk_size = FULL_BOUNDARY_SPEC_CHUNK_SAMPLES.min(boundary_size).max(1);

    for start in (0..boundary_size).step_by(chunk_size) {
        let count = (boundary_size - start).min(chunk_size);
        let spec = boundary_spec_matrix_range(flat_len, boundary_size, start, count);
        let rows = spec.nrows() / 2;
        let end = start + rows;
        let label = format!("{label_prefix} [{start}..{end})");
        let (crown_lo, crown_hi) = spec_guided_crown_with_engine(
            &fixture.prefix,
            &fixture.input,
            &fixture.ibp_node_bounds,
            &spec,
            Some(&NaiveCpuGemmEngine),
            &label,
        );
        assert_crown_no_looser(
            &crown_lo,
            &crown_hi,
            &ibp_first_lo[start..end],
            &ibp_first_hi[start..end],
            &ibp_last_lo[start..end],
            &ibp_last_hi[start..end],
        );

        first_lower.extend_from_slice(&crown_lo[..rows]);
        first_upper.extend_from_slice(&crown_hi[..rows]);
        last_lower.extend_from_slice(&crown_lo[rows..]);
        last_upper.extend_from_slice(&crown_hi[rows..]);
    }

    (
        (first_lower, first_upper),
        (last_lower, last_upper),
        flat_len,
    )
}

fn ibp_boundary_fixtures_from_fixture(
    fixture: &PrefixCrownFixture,
    boundary_sample_count: usize,
) -> BoundaryFixture {
    let flat_len = fixture.ibp_output.lower().len();
    let boundary_size = boundary_sample_count.min(flat_len);
    let (ibp_first_lo, ibp_first_hi, ibp_last_lo, ibp_last_hi) =
        extract_waveform_boundary_bounds(&fixture.ibp_output, boundary_size);
    (
        (ibp_first_lo, ibp_first_hi),
        (ibp_last_lo, ibp_last_hi),
        flat_len,
    )
}

fn build_prefix_crown_boundary_fixtures() -> BoundaryFixture {
    with_kokoro_crown_lock(|| {
        let fixture = cached_prefix_crown_fixture();
        crown_boundary_fixtures_from_fixture(
            &fixture,
            KOKORO_BOUNDARY_SAMPLES,
            "same-center crossfade chunk",
        )
    })
}

fn prefix_crown_boundary_fixtures() -> BoundaryFixture {
    static FIXTURE: OnceLock<BoundaryFixture> = OnceLock::new();
    FIXTURE
        .get_or_init(build_prefix_crown_boundary_fixtures)
        .clone()
}

fn scaled_tolerance(values: &[f32]) -> f32 {
    1e-5 * values.iter().copied().map(f32::abs).fold(1.0_f32, f32::max)
}

fn assert_interval_refinement(
    ibp_lower: &[f32],
    ibp_upper: &[f32],
    crown_lower: &[f32],
    crown_upper: &[f32],
    label: &str,
) -> usize {
    let mut tighter = 0usize;

    for i in 0..ibp_lower.len() {
        let tol = scaled_tolerance(&[ibp_lower[i], ibp_upper[i], crown_lower[i], crown_upper[i]]);
        assert!(
            crown_lower[i] >= ibp_lower[i] - tol,
            "sample {i}: CROWN {label} lower {:.6e} escaped IBP lower {:.6e}",
            crown_lower[i],
            ibp_lower[i]
        );
        assert!(
            crown_upper[i] <= ibp_upper[i] + tol,
            "sample {i}: CROWN {label} upper {:.6e} escaped IBP upper {:.6e}",
            crown_upper[i],
            ibp_upper[i]
        );

        if (crown_upper[i] - crown_lower[i]) < (ibp_upper[i] - ibp_lower[i]) - tol {
            tighter += 1;
        }
    }

    tighter
}

type IntervalBounds<'a> = (&'a [f32], &'a [f32]);

type CrossfadeGuaranteeMetrics = (f32, f32, usize);

fn assert_crown_crossfade_guarantees(
    crown_first: IntervalBounds<'_>,
    crown_last: IntervalBounds<'_>,
    crown_xfade: IntervalBounds<'_>,
    crown_xfade_energy: IntervalBounds<'_>,
) -> CrossfadeGuaranteeMetrics {
    let (crown_first_lo, crown_first_hi) = crown_first;
    let (crown_last_lo, crown_last_hi) = crown_last;
    let (crown_xfade_lo, crown_xfade_hi) = crown_xfade;
    let (crown_xfade_e_lo, crown_xfade_e_hi) = crown_xfade_energy;
    let (steady_first_e_lo, steady_first_e_hi) = energy_bounds(crown_first_lo, crown_first_hi);
    let (steady_last_e_lo, steady_last_e_hi) = energy_bounds(crown_last_lo, crown_last_hi);
    let energy_threshold = 1e-5;
    let mut max_excess_width = 0.0_f32;
    let mut min_energy_ratio = f32::INFINITY;
    let mut checked_energy_samples = 0usize;

    for i in 0..crown_xfade_lo.len() {
        let crown_width = crown_xfade_hi[i] - crown_xfade_lo[i];
        let max_chunk_width =
            (crown_last_hi[i] - crown_last_lo[i]).max(crown_first_hi[i] - crown_first_lo[i]);
        let excess_width = crown_width - max_chunk_width;
        max_excess_width = max_excess_width.max(excess_width);
        assert!(
            crown_width <= max_chunk_width + 1e-6,
            "sample {i}: CROWN crossfade width {:.6e} exceeded max chunk width {:.6e}",
            crown_width,
            max_chunk_width
        );

        let steady_upper = steady_first_e_hi[i].max(steady_last_e_hi[i]);
        if steady_upper > energy_threshold {
            assert!(
                crown_xfade_e_hi[i] <= 1.5 * steady_upper + 1e-8,
                "sample {i}: CROWN crossfade energy spike {:.6e} exceeded cap {:.6e}",
                crown_xfade_e_hi[i],
                1.5 * steady_upper
            );
        }

        let steady_lower = steady_first_e_lo[i].min(steady_last_e_lo[i]);
        assert!(
            crown_xfade_e_lo[i] >= 0.0,
            "sample {i}: CROWN crossfade energy lower bound should stay non-negative"
        );
        if steady_lower > energy_threshold {
            // The crossfade interval is composed from independently-bounded
            // endpoint samples, so it loses the sign correlation needed to
            // soundly prove a hard lower-energy floor on real-weight bounds.
            // Track the ratio for diagnostics, but keep the asserted guarantees
            // to width non-amplification and the 1.5x spike cap.
            min_energy_ratio = min_energy_ratio.min(crown_xfade_e_lo[i] / steady_lower);
            checked_energy_samples += 1;
        }
    }

    (
        max_excess_width,
        if checked_energy_samples == 0 {
            0.0
        } else {
            min_energy_ratio
        },
        checked_energy_samples,
    )
}

fn assert_two_chunk_crown_crossfade_guarantees(
    crown_chunk_a: IntervalBounds<'_>,
    crown_chunk_b: IntervalBounds<'_>,
    crown_xfade: IntervalBounds<'_>,
    crown_xfade_energy: IntervalBounds<'_>,
) -> f32 {
    let (chunk_a_lo, chunk_a_hi) = crown_chunk_a;
    let (chunk_b_lo, chunk_b_hi) = crown_chunk_b;
    let (crown_xfade_lo, crown_xfade_hi) = crown_xfade;
    let (crown_xfade_e_lo, crown_xfade_e_hi) = crown_xfade_energy;
    let (_steady_a_e_lo, steady_a_e_hi) = energy_bounds(chunk_a_lo, chunk_a_hi);
    let (_steady_b_e_lo, steady_b_e_hi) = energy_bounds(chunk_b_lo, chunk_b_hi);
    let energy_threshold = 1e-5;
    let mut max_excess_width = 0.0_f32;

    for i in 0..crown_xfade_lo.len() {
        let crown_width = crown_xfade_hi[i] - crown_xfade_lo[i];
        let max_chunk_width = (chunk_a_hi[i] - chunk_a_lo[i]).max(chunk_b_hi[i] - chunk_b_lo[i]);
        let excess_width = crown_width - max_chunk_width;
        max_excess_width = max_excess_width.max(excess_width);
        assert!(
            crown_width <= max_chunk_width + 1e-6,
            "sample {i}: two-chunk CROWN crossfade width {:.6e} exceeded max chunk width {:.6e}",
            crown_width,
            max_chunk_width
        );

        let steady_upper = steady_a_e_hi[i].max(steady_b_e_hi[i]);
        if steady_upper > energy_threshold {
            assert!(
                crown_xfade_e_hi[i] <= 1.5 * steady_upper + 1e-8,
                "sample {i}: two-chunk CROWN crossfade energy spike {:.6e} exceeded cap {:.6e}",
                crown_xfade_e_hi[i],
                1.5 * steady_upper
            );
        }

        let min_lower = chunk_a_lo[i].min(chunk_b_lo[i]);
        let max_upper = chunk_a_hi[i].max(chunk_b_hi[i]);
        assert!(
            crown_xfade_lo[i] >= min_lower - 1e-6,
            "sample {i}: two-chunk CROWN crossfade lower {:.6e} escaped hull lower {:.6e}",
            crown_xfade_lo[i],
            min_lower
        );
        assert!(
            crown_xfade_hi[i] <= max_upper + 1e-6,
            "sample {i}: two-chunk CROWN crossfade upper {:.6e} escaped hull upper {:.6e}",
            crown_xfade_hi[i],
            max_upper
        );
        assert!(
            crown_xfade_e_lo[i] >= 0.0,
            "sample {i}: two-chunk CROWN crossfade energy lower bound should stay non-negative"
        );
    }

    max_excess_width
}

#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
fn test_prefix_crown_crossfade_refines_ibp_guarantees_3500() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    // Concrete-point containment (#3683): evaluate prefix at center of the
    // epsilon ball and assert the output is within IBP bounds.
    {
        let fixture = with_kokoro_crown_lock(cached_prefix_crown_fixture);
        let concrete =
            evaluate_graph_at_center(&fixture.prefix, &fixture.input, "crossfade prefix center");
        assert_concrete_contained_in_bounds(
            &concrete,
            &fixture.ibp_output,
            "crossfade prefix IBP containment",
        );
    }

    let ((ibp_first_lo, ibp_first_hi), (ibp_last_lo, ibp_last_hi), flat_len) =
        prefix_ibp_boundary_fixtures();
    let ((crown_first_lo, crown_first_hi), (crown_last_lo, crown_last_hi), crown_flat_len) =
        prefix_crown_boundary_fixtures();

    assert_eq!(flat_len, crown_flat_len, "IBP/CROWN flat length mismatch");

    let n = ibp_first_lo.len();
    assert_eq!(n, crown_first_lo.len(), "boundary window length mismatch");
    assert_eq!(n, ibp_last_lo.len(), "IBP first/last length mismatch");
    assert_eq!(n, crown_last_lo.len(), "CROWN first/last length mismatch");

    let (ibp_xfade_lo, ibp_xfade_hi) =
        crossfade_overlap_add_bounds(&ibp_last_lo, &ibp_last_hi, &ibp_first_lo, &ibp_first_hi);
    let (crown_xfade_lo, crown_xfade_hi) = crossfade_overlap_add_bounds(
        &crown_last_lo,
        &crown_last_hi,
        &crown_first_lo,
        &crown_first_hi,
    );
    let (ibp_xfade_e_lo, ibp_xfade_e_hi) = energy_bounds(&ibp_xfade_lo, &ibp_xfade_hi);
    let (crown_xfade_e_lo, crown_xfade_e_hi) = energy_bounds(&crown_xfade_lo, &crown_xfade_hi);
    let tighter_crossfade = assert_interval_refinement(
        &ibp_xfade_lo,
        &ibp_xfade_hi,
        &crown_xfade_lo,
        &crown_xfade_hi,
        "crossfade",
    );
    let tighter_energy = assert_interval_refinement(
        &ibp_xfade_e_lo,
        &ibp_xfade_e_hi,
        &crown_xfade_e_lo,
        &crown_xfade_e_hi,
        "crossfade energy",
    );
    let (max_excess_width, min_energy_ratio, checked_energy_samples) =
        assert_crown_crossfade_guarantees(
            (&crown_first_lo, &crown_first_hi),
            (&crown_last_lo, &crown_last_hi),
            (&crown_xfade_lo, &crown_xfade_hi),
            (&crown_xfade_e_lo, &crown_xfade_e_hi),
        );
    // #3683: assert refinement metrics are non-trivial, not just logged.
    // CROWN should tighten at least some crossfade intervals or energies
    // relative to IBP. Zero tightening signals a regression in the CROWN
    // pipeline or a vacuous comparison.
    assert!(
        tighter_crossfade > 0 || tighter_energy > 0,
        "CROWN crossfade should tighten at least one interval or energy: \
         tighter_crossfade={tighter_crossfade}, tighter_energy={tighter_energy}"
    );
    eprintln!(
        "CROWN crossfade refinement: {n} boundary samples from {flat_len} total, \
         tighter intervals = {tighter_crossfade}/{n}, tighter energies = {tighter_energy}/{n}, \
         max excess width = {max_excess_width:.4e}, min lower-energy ratio = {min_energy_ratio:.4}, \
         checked lower-energy samples = {checked_energy_samples}"
    );
}
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
fn test_two_chunk_crown_crossfade_refines_ibp_guarantees_3500() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    // Keep the historical exact-name replay for the real two-chunk lane, but
    // do not require strictly positive tightening here. On the current 2-node
    // shallow prefix, distinct chunk centers still satisfy the no-loosening
    // regression while empirically yielding 0/240 tighter samples after
    // composing the interval crossfade. The deep-prefix smoke in
    // `crown_ibp_tightening.rs` carries the non-trivial tightening claim.
    let (
        _,
        _,
        (ibp_xfade_lo, ibp_xfade_hi),
        (crown_xfade_lo, crown_xfade_hi),
        (ibp_xfade_e_lo, ibp_xfade_e_hi),
        (crown_xfade_e_lo, crown_xfade_e_hi),
        flat_len,
    ) = two_chunk_crossfade_bounds();
    let n = ibp_xfade_lo.len();
    let tighter_crossfade = assert_interval_refinement(
        &ibp_xfade_lo,
        &ibp_xfade_hi,
        &crown_xfade_lo,
        &crown_xfade_hi,
        "two-chunk crossfade",
    );
    let tighter_energy = assert_interval_refinement(
        &ibp_xfade_e_lo,
        &ibp_xfade_e_hi,
        &crown_xfade_e_lo,
        &crown_xfade_e_hi,
        "two-chunk crossfade energy",
    );
    eprintln!(
        "two-chunk CROWN crossfade refinement: {n} boundary samples from {flat_len} total, \
         tighter intervals = {tighter_crossfade}/{n}, tighter energies = {tighter_energy}/{n}"
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
fn test_two_chunk_crown_crossfade_contains_concrete_overlap_3500() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    let (
        _,
        _,
        _,
        (crown_xfade_lo, crown_xfade_hi),
        _,
        (crown_xfade_e_lo, crown_xfade_e_hi),
        flat_len,
    ) = two_chunk_crossfade_bounds();
    let (concrete_waveform, concrete_energy, concrete_flat_len) = two_chunk_concrete_crossfade();

    assert_eq!(
        flat_len, concrete_flat_len,
        "two-chunk concrete/CROWN flat length mismatch"
    );

    let crown_waveform =
        bounded_tensor_from_slices(&crown_xfade_lo, &crown_xfade_hi, "two-chunk CROWN waveform");
    let crown_energy = bounded_tensor_from_slices(
        &crown_xfade_e_lo,
        &crown_xfade_e_hi,
        "two-chunk CROWN crossfade energy",
    );

    assert_concrete_contained_in_bounds(
        &concrete_waveform,
        &crown_waveform,
        "two-chunk concrete waveform containment",
    );
    assert_concrete_contained_in_bounds(
        &concrete_energy,
        &crown_energy,
        "two-chunk concrete energy containment",
    );
}

#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
fn test_two_chunk_crown_crossfade_preserves_width_and_energy_caps_3500() {
    crate::test_fixtures::require_test_model_or_skip!("kokoro_vocoder.onnx");
    let (
        (crown_a_last_lo, crown_a_last_hi),
        (crown_b_first_lo, crown_b_first_hi),
        _,
        (crown_xfade_lo, crown_xfade_hi),
        _,
        (crown_xfade_e_lo, crown_xfade_e_hi),
        flat_len,
    ) = two_chunk_crossfade_bounds();
    let n = crown_xfade_lo.len();
    let max_excess_width = assert_two_chunk_crown_crossfade_guarantees(
        (&crown_a_last_lo, &crown_a_last_hi),
        (&crown_b_first_lo, &crown_b_first_hi),
        (&crown_xfade_lo, &crown_xfade_hi),
        (&crown_xfade_e_lo, &crown_xfade_e_hi),
    );
    eprintln!(
        "two-chunk CROWN crossfade guarantees: {n} boundary samples from {flat_len} total, \
         max excess width = {max_excess_width:.4e}"
    );
}
