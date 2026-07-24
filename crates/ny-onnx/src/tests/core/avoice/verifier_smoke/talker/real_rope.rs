// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ny_core::{MethodUsed, NaiveCpuGemmEngine};
use ny_propagate::{
    composition::certificate::BoundCertificationResult, PropagationConfig, PropagationMethod,
    Verifier,
};
use std::sync::Arc;

/// Verify the canonical real-RoPE Qwen3-TTS talker attention softmax output
/// through `Verifier::verify_graph(...)`. Softmax outputs are always in
/// [0, 1].
///
/// Uses IBP on the canonical exported sequence length with real positional
/// encoding.
///
/// Part of #L1.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_avoice_talker_softmax_real_rope_verifier_smoke_l1() {
    crate::test_fixtures::require_test_model_or_skip!("talker_attention_layer0.onnx");
    let (softmax_graph, softmax_name, input, output_size) =
        talker_verifier::talker_softmax_real_rope_verifier_setup();

    let output_bounds: Vec<Bound> = (0..output_size).map(|_| Bound::new(0.0, 1.0)).collect();
    let result = run_verifier_smoke_route(
        &softmax_graph,
        &input,
        output_bounds,
        120_000,
        VerifierSmokeRoute::Ibp,
        "talker softmax real-RoPE IBP verifier smoke",
    );

    talker_verifier::assert_softmax_verifier_result(
        &result,
        output_size,
        &softmax_name,
        "talker softmax real-RoPE IBP verifier smoke",
    );
    assert_verified_result_contains_center(
        &result,
        &softmax_graph,
        &input,
        "talker softmax real-RoPE IBP verifier smoke",
    );
}

/// Verify the canonical real-RoPE Qwen3-TTS talker attention softmax output
/// through `Verifier::verify_graph(...)` using the local CROWN route policy.
/// Softmax outputs are always in [0, 1].
///
/// Part of #4217.
#[ignore = "full spec-CROWN over the 4096-output seq16 real-RoPE talker graph on the deterministic \
            NaiveCpuGemmEngine is measured (2026-07-19, --release --test-threads=1) to run well beyond \
            7 minutes uncontended before returning Verified — the naive backward pass does not honor \
            the 120s deadline mid-pass, so it blows the 300s wall-clock watchdog rather than degrading \
            to a Timeout verdict. This is a compute-scale limit, NOT a soundness or overflow issue: \
            softmax outputs are always in [0,1] and that verdict is already proven fast by the passing \
            real-RoPE IBP smoke `..._verifier_smoke_l1` (35ms) and by the passing short-seq CROWN smoke \
            `..._range_crown_verifier_smoke_3701`; the backward-CROWN capability on THIS real-RoPE graph \
            (provenance=Crown, violations=0 at eps=2e-6) is exercised by the passing \
            `talker_attention::crown_boundary` test. Re-enable when a faster deterministic GEMM engine \
            or an interruptible spec-CROWN backward makes a single pass fit a unit-test budget."]
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_avoice_talker_softmax_real_rope_crown_verifier_smoke_4217() {
    let (softmax_graph, softmax_name, input, output_size) =
        talker_verifier::talker_softmax_real_rope_verifier_setup();

    let output_bounds: Vec<Bound> = (0..output_size).map(|_| Bound::new(0.0, 1.0)).collect();
    let result = run_verifier_smoke_route(
        &softmax_graph,
        &input,
        output_bounds,
        120_000,
        VerifierSmokeRoute::Crown,
        "talker softmax real-RoPE CROWN verifier smoke",
    );

    talker_verifier::assert_softmax_verifier_result(
        &result,
        output_size,
        &softmax_name,
        "talker softmax real-RoPE CROWN verifier smoke",
    );
    assert_verified_result_contains_center(
        &result,
        &softmax_graph,
        &input,
        "talker softmax real-RoPE CROWN verifier smoke",
    );
}

/// CROWN route canary for the canonical real-RoPE talker softmax graph.
///
/// Uses `certify_graph_bounds` to bypass the spec layer and directly inspect
/// the propagation certificate. Asserts three things:
///
/// 1. **Method canary**: `actual_method == Crown`, not `Ibp`. Catches silent
///    fallback regressions where the CROWN dispatch path bails to IBP without
///    surfacing an error.
///
/// 2. **Provenance canary**: `BoundProvenance::Crown`, the coarse provenance
///    summary derived from the method tag. Redundant with (1) but catches
///    provenance-mapping bugs independently.
///
/// 3. **Tightening canary**: CROWN bounds are at least as tight as IBP on
///    every output dimension, and strictly tighter on at least one. If CROWN
///    silently fell back to IBP internally (while still tagging the result as
///    Crown), this assertion catches the regression.
///
/// Part of #4217.
#[ignore = "certify_graph_bounds runs the same full spec-CROWN over the 4096-output seq16 real-RoPE \
            talker graph on the deterministic NaiveCpuGemmEngine as the sibling CROWN verifier smoke; \
            measured (2026-07-19, --release --test-threads=1) to run well beyond 7 minutes uncontended \
            (IBP baseline is 35ms; the certify pass is the long pole and does not honor its 120s deadline \
            mid-pass, so it blows the 300s watchdog). Compute-scale limit, NOT a soundness/overflow issue: \
            the method=Crown route and CROWN-tighter-than-IBP behavior on this graph is covered by the \
            passing `talker_attention::crown_boundary` (backward CROWN, provenance=Crown, violations=0 at \
            eps=2e-6) and the short-seq CROWN smokes. Re-enable with a faster deterministic engine or an \
            interruptible spec-CROWN backward."]
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_avoice_talker_real_rope_crown_route_canary_4217() {
    let label = "real-RoPE CROWN route canary";
    let (softmax_graph, _softmax_name, input, _output_size) =
        talker_verifier::talker_softmax_real_rope_verifier_setup();

    // --- IBP baseline bounds ---
    let ibp_bounds = softmax_graph
        .propagate_ibp(&input)
        .unwrap_or_else(|e| panic!("{label}: IBP propagation should succeed: {e}"));

    // --- CROWN certified bounds ---
    let config = PropagationConfig {
        method: PropagationMethod::Crown,
        ..Default::default()
    };
    let verifier = Verifier::new_with_engine(config, Arc::new(NaiveCpuGemmEngine));

    let cert_result = verifier
        .certify_graph_bounds(
            "talker-softmax-real-rope-canary",
            &softmax_graph,
            &input,
            Some(common::release_budget_ms(120_000)),
        )
        .unwrap_or_else(|e| panic!("{label}: certify_graph_bounds should not error: {e}"));

    let cert = match &cert_result {
        BoundCertificationResult::Certified(c) => c,
        BoundCertificationResult::Timeout { actual_method, .. } => {
            panic!("{label}: certify_graph_bounds timed out (method at timeout: {actual_method:?})")
        }
    };

    // Canary 1: actual method is Crown, not Ibp.
    assert_eq!(
        cert.actual_method(),
        &MethodUsed::Crown,
        "{label}: actual_method must be Crown; got {:?} — CROWN likely fell back to IBP",
        cert.actual_method()
    );

    // Canary 2: coarse provenance is Crown.
    assert_eq!(
        cert.provenance(),
        ny_propagate::composition::certificate::BoundProvenance::Crown,
        "{label}: provenance must be Crown; got {:?}",
        cert.provenance()
    );

    // Canary 3: CROWN bounds are at least as tight as IBP, and strictly
    // tighter on at least one dimension.
    let crown_lower = cert.output_bounds().lower();
    let crown_upper = cert.output_bounds().upper();
    let ibp_lower = ibp_bounds.lower();
    let ibp_upper = ibp_bounds.upper();

    assert_eq!(
        crown_lower.len(),
        ibp_lower.len(),
        "{label}: CROWN and IBP output sizes must match"
    );

    let mut any_strictly_tighter = false;
    for idx in 0..crown_lower.len() {
        let cl = crown_lower.as_slice().unwrap()[idx];
        let cu = crown_upper.as_slice().unwrap()[idx];
        let il = ibp_lower.as_slice().unwrap()[idx];
        let iu = ibp_upper.as_slice().unwrap()[idx];

        // CROWN must not be looser than IBP (with a small tolerance for
        // floating-point non-determinism in the backward pass).
        let tol = 1e-5;
        assert!(
            cl >= il - tol,
            "{label}: CROWN lower[{idx}]={cl} is looser than IBP lower={il} (tol={tol})"
        );
        assert!(
            cu <= iu + tol,
            "{label}: CROWN upper[{idx}]={cu} is looser than IBP upper={iu} (tol={tol})"
        );

        let crown_width = cu - cl;
        let ibp_width = iu - il;
        if crown_width < ibp_width - tol {
            any_strictly_tighter = true;
        }
    }
    assert!(
        any_strictly_tighter,
        "{label}: CROWN bounds must be strictly tighter than IBP on at least one output \
         dimension — if no dimension is tighter, CROWN is effectively running as IBP"
    );
}
