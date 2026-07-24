// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::common::evaluate_graph_at_center;
use super::super::boundary::{
    assert_boundary_windows_valid, extract_waveform_boundary_bounds, BoundaryWindow,
    KOKORO_BOUNDARY_SAMPLES,
};
use super::super::crown_ibp_tightening::support::{
    build_prefix_crown_fixture_centered, cached_prefix_crown_fixture, with_kokoro_crown_lock,
    PrefixCrownFixture,
};
use super::*;
use std::sync::OnceLock;

type TwoChunkBoundaryFixture = (BoundaryWindow, BoundaryWindow, usize);
type TwoChunkBoundaryRefinementFixture = (TwoChunkBoundaryFixture, TwoChunkBoundaryFixture);
pub(super) type TwoChunkCrownFixtures = (PrefixCrownFixture, PrefixCrownFixture);
pub(super) type TwoChunkCrossfadeComputation = (
    BoundaryWindow,
    BoundaryWindow,
    BoundaryWindow,
    BoundaryWindow,
    BoundaryWindow,
    BoundaryWindow,
    usize,
);

fn build_two_chunk_crown_fixtures() -> TwoChunkCrownFixtures {
    with_kokoro_crown_lock(|| {
        eprintln!("two-chunk CROWN: chunk-a cached fixture");
        let chunk_a_fixture = cached_prefix_crown_fixture();
        eprintln!("two-chunk CROWN: chunk-b centered fixture");
        let chunk_b_fixture = build_prefix_crown_fixture_centered(0.1);
        (chunk_a_fixture, chunk_b_fixture)
    })
}

pub(super) fn two_chunk_crown_fixtures() -> TwoChunkCrownFixtures {
    static FIXTURE: OnceLock<TwoChunkCrownFixtures> = OnceLock::new();
    FIXTURE.get_or_init(build_two_chunk_crown_fixtures).clone()
}

fn build_two_chunk_boundary_refinement_fixtures() -> TwoChunkBoundaryRefinementFixture {
    let (chunk_a_fixture, chunk_b_fixture) = two_chunk_crown_fixtures();
    let chunk_a_ibp = ibp_boundary_fixtures_from_fixture(&chunk_a_fixture, KOKORO_BOUNDARY_SAMPLES);
    let chunk_b_ibp = ibp_boundary_fixtures_from_fixture(&chunk_b_fixture, KOKORO_BOUNDARY_SAMPLES);
    let (_, (chunk_a_last_lo, chunk_a_last_hi), flat_len_a) = crown_boundary_fixtures_from_fixture(
        &chunk_a_fixture,
        KOKORO_BOUNDARY_SAMPLES,
        "two-chunk chunk-a",
    );
    eprintln!("two-chunk CROWN: chunk-a boundary refinement ready");
    let ((chunk_b_first_lo, chunk_b_first_hi), _, flat_len_b) =
        crown_boundary_fixtures_from_fixture(
            &chunk_b_fixture,
            KOKORO_BOUNDARY_SAMPLES,
            "two-chunk chunk-b",
        );
    eprintln!("two-chunk CROWN: chunk-b boundary refinement ready");
    let (_, (ibp_a_last_lo, ibp_a_last_hi), flat_len_a_ibp) = chunk_a_ibp;
    let ((ibp_b_first_lo, ibp_b_first_hi), _, flat_len_b_ibp) = chunk_b_ibp;
    assert_eq!(
        flat_len_a, flat_len_b,
        "two-chunk CROWN should keep the same prefix output length"
    );
    assert_eq!(
        flat_len_a, flat_len_a_ibp,
        "chunk-a IBP/CROWN should keep the same prefix output length"
    );
    assert_eq!(
        flat_len_a, flat_len_b_ibp,
        "chunk-b IBP/CROWN should keep the same prefix output length"
    );
    (
        (
            (ibp_a_last_lo, ibp_a_last_hi),
            (ibp_b_first_lo, ibp_b_first_hi),
            flat_len_a,
        ),
        (
            (chunk_a_last_lo, chunk_a_last_hi),
            (chunk_b_first_lo, chunk_b_first_hi),
            flat_len_a,
        ),
    )
}

fn two_chunk_boundary_refinement_fixtures() -> TwoChunkBoundaryRefinementFixture {
    static FIXTURE: OnceLock<TwoChunkBoundaryRefinementFixture> = OnceLock::new();
    FIXTURE
        .get_or_init(build_two_chunk_boundary_refinement_fixtures)
        .clone()
}

fn collapse_pointlike_interval(lower: &[f32], upper: &[f32], label: &str) -> Vec<f32> {
    assert_eq!(
        lower.len(),
        upper.len(),
        "{label}: lower/upper length mismatch"
    );
    lower
        .iter()
        .zip(upper.iter())
        .enumerate()
        .map(|(idx, (&lo, &hi))| {
            let tol = scaled_tolerance(&[lo, hi]);
            assert!(
                (hi - lo).abs() <= tol,
                "{label}[{idx}] should stay point-like, got [{lo:.6e}, {hi:.6e}]"
            );
            f32::midpoint(lo, hi)
        })
        .collect()
}

pub(super) fn bounded_tensor_from_slices(
    lower: &[f32],
    upper: &[f32],
    label: &str,
) -> BoundedTensor {
    assert_eq!(
        lower.len(),
        upper.len(),
        "{label}: lower/upper length mismatch"
    );
    let lower = ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec())
        .unwrap_or_else(|err| panic!("{label}: lower shape construction failed: {err}"));
    let upper = ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec())
        .unwrap_or_else(|err| panic!("{label}: upper shape construction failed: {err}"));
    BoundedTensor::new(lower, upper)
        .unwrap_or_else(|err| panic!("{label}: interval tensor construction failed: {err}"))
}

pub(super) fn two_chunk_concrete_crossfade() -> (BoundedTensor, BoundedTensor, usize) {
    let (chunk_a_fixture, chunk_b_fixture) = two_chunk_crown_fixtures();
    let chunk_a_concrete = evaluate_graph_at_center(
        &chunk_a_fixture.prefix,
        &chunk_a_fixture.input,
        "two-chunk chunk-a concrete center",
    );
    let chunk_b_concrete = evaluate_graph_at_center(
        &chunk_b_fixture.prefix,
        &chunk_b_fixture.input,
        "two-chunk chunk-b concrete center",
    );

    let flat_len_a = chunk_a_concrete.flatten().lower().len();
    let flat_len_b = chunk_b_concrete.flatten().lower().len();
    let boundary_size = KOKORO_BOUNDARY_SAMPLES.min(flat_len_a).min(flat_len_b);
    assert!(
        boundary_size > 0,
        "two-chunk concrete crossfade requires non-empty outputs"
    );

    let (a_first_lo, a_first_hi, a_last_lo, a_last_hi) =
        extract_waveform_boundary_bounds(&chunk_a_concrete, boundary_size);
    let (b_first_lo, b_first_hi, b_last_lo, b_last_hi) =
        extract_waveform_boundary_bounds(&chunk_b_concrete, boundary_size);
    assert_boundary_windows_valid(
        &a_first_lo,
        &a_first_hi,
        &a_last_lo,
        &a_last_hi,
        flat_len_a,
        "two-chunk chunk-a concrete",
    );
    assert_boundary_windows_valid(
        &b_first_lo,
        &b_first_hi,
        &b_last_lo,
        &b_last_hi,
        flat_len_b,
        "two-chunk chunk-b concrete",
    );
    let chunk_a_last = collapse_pointlike_interval(
        &a_last_lo,
        &a_last_hi,
        "two-chunk chunk-a concrete last boundary",
    );
    let chunk_b_first = collapse_pointlike_interval(
        &b_first_lo,
        &b_first_hi,
        "two-chunk chunk-b concrete first boundary",
    );
    let (waveform_lo, waveform_hi) =
        crossfade_overlap_add_bounds(&chunk_a_last, &chunk_a_last, &chunk_b_first, &chunk_b_first);
    let waveform = collapse_pointlike_interval(
        &waveform_lo,
        &waveform_hi,
        "two-chunk concrete crossfade waveform",
    );
    let (energy_lo, energy_hi) = energy_bounds(&waveform, &waveform);
    let energy = collapse_pointlike_interval(
        &energy_lo,
        &energy_hi,
        "two-chunk concrete crossfade energy",
    );

    (
        bounded_tensor_from_slices(
            &waveform,
            &waveform,
            "two-chunk concrete crossfade waveform",
        ),
        bounded_tensor_from_slices(&energy, &energy, "two-chunk concrete crossfade energy"),
        flat_len_a,
    )
}

pub(super) fn two_chunk_crossfade_bounds() -> TwoChunkCrossfadeComputation {
    let (
        ((ibp_a_last_lo, ibp_a_last_hi), (ibp_b_first_lo, ibp_b_first_hi), flat_len_a),
        ((crown_a_last_lo, crown_a_last_hi), (crown_b_first_lo, crown_b_first_hi), crown_flat_len),
    ) = two_chunk_boundary_refinement_fixtures();
    assert_eq!(
        flat_len_a, crown_flat_len,
        "two-chunk IBP/CROWN flat length mismatch"
    );
    let n = KOKORO_BOUNDARY_SAMPLES
        .min(ibp_a_last_lo.len())
        .min(ibp_b_first_lo.len());
    assert_eq!(
        n,
        crown_a_last_lo.len(),
        "two-chunk CROWN chunk-a boundary window length mismatch"
    );
    assert_eq!(
        n,
        crown_b_first_lo.len(),
        "two-chunk CROWN chunk-b boundary window length mismatch"
    );
    let ibp_crossfade = crossfade_overlap_add_bounds(
        &ibp_a_last_lo[..n],
        &ibp_a_last_hi[..n],
        &ibp_b_first_lo[..n],
        &ibp_b_first_hi[..n],
    );
    let crown_crossfade = crossfade_overlap_add_bounds(
        &crown_a_last_lo,
        &crown_a_last_hi,
        &crown_b_first_lo,
        &crown_b_first_hi,
    );
    let ibp_crossfade_energy = energy_bounds(&ibp_crossfade.0, &ibp_crossfade.1);
    let crown_crossfade_energy = energy_bounds(&crown_crossfade.0, &crown_crossfade.1);
    (
        (crown_a_last_lo, crown_a_last_hi),
        (crown_b_first_lo, crown_b_first_hi),
        ibp_crossfade,
        crown_crossfade,
        ibp_crossfade_energy,
        crown_crossfade_energy,
        flat_len_a,
    )
}
