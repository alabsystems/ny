// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::boundary::{extract_waveform_boundary_bounds, KOKORO_BOUNDARY_SAMPLES};
use super::support::{
    assert_crown_no_looser, assert_finite_and_ordered_slices, boundary_spec_matrix,
    boundary_spec_matrix_range, cached_prefix_crown_fixture, spec_guided_crown,
    spec_guided_crown_with_engine, spec_guided_crown_with_unbounded_engine, with_kokoro_crown_lock,
    FULL_BOUNDARY_SPEC_CHUNK_SAMPLES, SMOKE_BOUNDARY_SPEC_SAMPLES,
};
use ny_test_utils::assert_slice_close_relative;
use std::time::Instant;

fn run_boundary_chunk_with_engine(
    start: usize,
    count: usize,
    label: &str,
) -> (usize, usize, usize) {
    use ny_core::NaiveCpuGemmEngine;

    let f = cached_prefix_crown_fixture();
    let flat_len = f.ibp_output.lower().len();
    let boundary_size = KOKORO_BOUNDARY_SAMPLES.min(flat_len);
    let spec = boundary_spec_matrix_range(flat_len, boundary_size, start, count);
    let rows = spec.nrows();

    let t0 = Instant::now();
    let (crown_lo, crown_hi) = spec_guided_crown_with_engine(
        &f.prefix,
        &f.input,
        &f.ibp_node_bounds,
        &spec,
        Some(&NaiveCpuGemmEngine),
        label,
    );
    let elapsed = t0.elapsed().as_secs_f64();
    assert_finite_and_ordered_slices(&crown_lo, &crown_hi, label);

    let (fl, fu, ll, lu) = extract_waveform_boundary_bounds(&f.ibp_output, boundary_size);
    let end = start + rows / 2;
    let (tighter, equal) = assert_crown_no_looser(
        &crown_lo,
        &crown_hi,
        &fl[start..end],
        &fu[start..end],
        &ll[start..end],
        &lu[start..end],
    );
    eprintln!(
        "{label}: boundary[{start}..{end}) => {rows} specs in {elapsed:.1}s ({tighter} tighter, {equal} equal)"
    );
    (rows / 2, tighter, equal)
}

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_spec_guided_crown_kokoro_vocoder_prefix_boundary_3500() {
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    with_kokoro_crown_lock(|| {
        let f = cached_prefix_crown_fixture();
        let flat_len = f.ibp_output.lower().len();
        let n = SMOKE_BOUNDARY_SPEC_SAMPLES.min(flat_len);

        let spec = boundary_spec_matrix(flat_len, SMOKE_BOUNDARY_SPEC_SAMPLES);

        let t0 = Instant::now();
        let (crown_lo, crown_hi) =
            spec_guided_crown(&f.prefix, &f.input, &f.ibp_node_bounds, &spec, "IBP");
        eprintln!("spec-guided CROWN: {:.1}s", t0.elapsed().as_secs_f64());
        assert_finite_and_ordered_slices(&crown_lo, &crown_hi, "smoke CROWN");

        // Use SMOKE_BOUNDARY_SPEC_SAMPLES (not KOKORO_BOUNDARY_SAMPLES) so the IBP
        // positions match the spec matrix targets (first/last N of flat output).
        let (fl, fu, ll, lu) =
            extract_waveform_boundary_bounds(&f.ibp_output, SMOKE_BOUNDARY_SPEC_SAMPLES);
        let (t, e) = assert_crown_no_looser(&crown_lo, &crown_hi, &fl, &fu, &ll, &lu);
        eprintln!("CROWN vs IBP: {t} tighter, {e} equal");

        if n > 0 {
            eprintln!(
                "first: IBP=[{:.4e},{:.4e}] CROWN=[{:.4e},{:.4e}]",
                fl[0], fu[0], crown_lo[0], crown_hi[0]
            );
        }
    });
}

/// Regression test proving GemmEngine is actually dispatched through Conv1d
/// CROWN backward on a real Kokoro vocoder prefix subgraph (#3598).
///
/// Uses a CountingGemmEngine to verify that at least one GEMM call flows
/// through the engine during spec-guided CROWN on a ConvTranspose1d-dominated
/// prefix. This validates the dispatch wiring added in the #3598 fix.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_crown_engine_dispatched_conv1d_kokoro_prefix_3598() {
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    use ny_test_utils::CountingGemmEngine;

    // Use fewer boundary specs to keep CROWN fast (~3s CROWN, ~87s fixture).
    const ENGINE_TEST_SPECS: usize = 2;

    with_kokoro_crown_lock(|| {
        let f = cached_prefix_crown_fixture();
        let flat_len = f.ibp_output.lower().len();
        let n = ENGINE_TEST_SPECS.min(flat_len);

        let spec = boundary_spec_matrix(flat_len, ENGINE_TEST_SPECS);
        let engine = CountingGemmEngine::new();

        let t0 = Instant::now();
        let (crown_lo, crown_hi) = spec_guided_crown_with_unbounded_engine(
            &f.prefix,
            &f.input,
            &f.ibp_node_bounds,
            &spec,
            &engine,
            "engine-aware CROWN",
        );
        let elapsed = t0.elapsed().as_secs_f64();
        let gemm_count = engine.gemm_calls();
        eprintln!(
            "engine-aware CROWN ({ENGINE_TEST_SPECS} specs): {elapsed:.1}s, {gemm_count} GEMM calls"
        );

        // Core assertion: engine was actually called during Conv1d/ConvTranspose1d
        // CROWN backward. A count of 0 means the dispatch wiring is broken.
        assert!(
            gemm_count > 0,
            "GemmEngine received 0 GEMM calls — Conv1d CROWN backward dispatch is not \
             threading engine through. Expected >0 calls for {n} boundary specs on a \
             ConvTranspose1d-dominated prefix."
        );
        assert_finite_and_ordered_slices(&crown_lo, &crown_hi, "engine CROWN");

        // The same generic engine must be refused when the real scored
        // deadline is present: GemmEngine cannot cooperatively cancel an
        // in-flight launch. The pollable certified CPU route still returns
        // valid bounds.
        let deadline_engine = CountingGemmEngine::new();
        let (deadline_lo, deadline_hi) = spec_guided_crown_with_engine(
            &f.prefix,
            &f.input,
            &f.ibp_node_bounds,
            &spec,
            Some(&deadline_engine),
            "deadline-scored CROWN",
        );
        assert_eq!(
            deadline_engine.gemm_calls(),
            0,
            "finite-deadline Conv1d CROWN must refuse a generic opaque GEMM engine"
        );
        assert_finite_and_ordered_slices(&deadline_lo, &deadline_hi, "deadline-scored CROWN");

        eprintln!(
            "Conv1d CROWN engine wiring validated: {gemm_count} GEMM dispatches, \
             {n} boundary specs; deadline-scored opaque dispatch correctly refused"
        );
    });
}

/// Graph-level regression for #3598: engine-backed spec-guided CROWN must
/// preserve the baseline `engine=None` bounds on the real Kokoro prefix.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_crown_engine_matches_cpu_baseline_on_kokoro_prefix_3598() {
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    use ny_core::NaiveCpuGemmEngine;

    const EQUIVALENCE_SPECS: usize = 2;

    with_kokoro_crown_lock(|| {
        let f = cached_prefix_crown_fixture();
        let flat_len = f.ibp_output.lower().len();
        let spec = boundary_spec_matrix(flat_len, EQUIVALENCE_SPECS);

        let (baseline_lo, baseline_hi) = spec_guided_crown(
            &f.prefix,
            &f.input,
            &f.ibp_node_bounds,
            &spec,
            "baseline CROWN",
        );
        let (engine_lo, engine_hi) = spec_guided_crown_with_engine(
            &f.prefix,
            &f.input,
            &f.ibp_node_bounds,
            &spec,
            Some(&NaiveCpuGemmEngine),
            "engine CROWN",
        );

        assert_slice_close_relative(&engine_lo, &baseline_lo, 1e-5, "engine lower");
        assert_slice_close_relative(&engine_hi, &baseline_hi, 1e-5, "engine upper");
    });
}

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_spec_guided_crown_kokoro_vocoder_prefix_boundary_chunk0_3500() {
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    with_kokoro_crown_lock(|| {
        let (covered, tighter, equal) = run_boundary_chunk_with_engine(
            0,
            FULL_BOUNDARY_SPEC_CHUNK_SAMPLES,
            "chunk0 engine CROWN",
        );
        eprintln!("chunk0 summary: covered={covered}, tighter={tighter}, equal={equal}");
    });
}

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_spec_guided_crown_kokoro_vocoder_prefix_boundary_chunk1_3500() {
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    with_kokoro_crown_lock(|| {
        let (covered, tighter, equal) = run_boundary_chunk_with_engine(
            FULL_BOUNDARY_SPEC_CHUNK_SAMPLES,
            FULL_BOUNDARY_SPEC_CHUNK_SAMPLES,
            "chunk1 engine CROWN",
        );
        eprintln!("chunk1 summary: covered={covered}, tighter={tighter}, equal={equal}");
    });
}

#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_spec_guided_crown_kokoro_vocoder_prefix_boundary_chunk2_3500() {
    crate::test_fixtures::assert_test_model_available!("kokoro_vocoder.onnx");
    with_kokoro_crown_lock(|| {
        let (covered, tighter, equal) = run_boundary_chunk_with_engine(
            2 * FULL_BOUNDARY_SPEC_CHUNK_SAMPLES,
            FULL_BOUNDARY_SPEC_CHUNK_SAMPLES,
            "chunk2 engine CROWN",
        );
        eprintln!("chunk2 summary: covered={covered}, tighter={tighter}, equal={equal}");
    });
}
