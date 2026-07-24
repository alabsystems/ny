// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ny_core::{MethodUsed, NaiveCpuGemmEngine};
use ny_propagate::{
    composition::certificate::{BoundCertificate, BoundCertificationResult},
    PropagationConfig, PropagationMethod, Verifier,
};
use std::sync::Arc;

/// Verify the short-seq identity-RoPE Qwen3-TTS talker attention softmax
/// output through `Verifier::verify_graph(...)` using the local
/// `VerifierSmokeRoute::Crown` policy.
///
/// This exercises the graph verifier's CROWN branch (`verifier/graph.rs:135`),
/// which has distinct routing from the IBP path: it enters graph alpha/CROWN
/// dispatch, can mutate `actual_method` on fallback, and is the verifier-owned
/// route that later avoice property-verification packets depend on.
///
/// Part of #3701.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_avoice_talker_softmax_range_crown_verifier_smoke_3701() {
    crate::test_fixtures::require_test_model_or_skip!("talker_attention_layer0.onnx");
    let (softmax_graph, softmax_name, input, output_size) =
        talker_verifier::talker_softmax_verifier_setup();

    let output_bounds: Vec<Bound> = (0..output_size).map(|_| Bound::new(0.0, 1.0)).collect();
    let result = run_verifier_smoke_route(
        &softmax_graph,
        &input,
        output_bounds,
        120_000,
        VerifierSmokeRoute::Crown,
        "talker softmax CROWN verifier smoke",
    );

    // #3701: CROWN used to fall back to IBP because backward_sqrt_node wrapped
    // UnsupportedConfiguration in InvalidSpec (#3619). Fix (#3840, 035d13c7d)
    // preserves structured errors so CROWN-IBP falls back gracefully.

    talker_verifier::assert_softmax_verifier_result(
        &result,
        output_size,
        &softmax_name,
        "talker softmax CROWN verifier smoke",
    );
    assert_verified_result_contains_center(
        &result,
        &softmax_graph,
        &input,
        "talker softmax CROWN verifier smoke",
    );
}

/// Find the first certified bound element that violates `[0, 1]` (with 1e-6
/// tolerance). Returns `None` when all bounds fit.
fn first_unit_interval_violation(cert: &BoundCertificate) -> Option<(usize, f32, f32)> {
    let lower = cert.output_bounds().lower();
    let upper = cert.output_bounds().upper();
    for (idx, (lo, hi)) in lower.iter().zip(upper.iter()).enumerate() {
        if *lo < -1e-6 || *hi > 1.0 + 1e-6 {
            return Some((idx, *lo, *hi));
        }
    }
    None
}

/// Diagnose the relationship between `verify_graph` and `certify_graph_bounds`
/// results. Panics with a targeted message when they disagree or when bounds
/// exceed `[0, 1]`.
fn diagnose_certify_vs_verify(verify_result: &VerificationResult, cert: &BoundCertificate) {
    let offender = first_unit_interval_violation(cert);
    let any_exceeds_unit = offender.is_some();

    let verify_ok = matches!(verify_result, VerificationResult::Verified { .. });
    let verify_loose = matches!(
        verify_result,
        VerificationResult::Unknown {
            reason: UnknownReason::BoundsTooLoose { .. },
            ..
        }
    );

    eprintln!(
        "#4219 harness: verify={}, cert_in_unit={}, method={:?}",
        if verify_ok {
            "Verified"
        } else if verify_loose {
            "BoundsTooLoose"
        } else {
            "other"
        },
        !any_exceeds_unit,
        cert.actual_method(),
    );

    if verify_ok && !any_exceeds_unit {
        return; // Both agree: tight enough.
    }
    if verify_loose && any_exceeds_unit {
        let (idx, lo, hi) = offender.unwrap();
        panic!(
            "#4219: CROWN propagation regression — certified bounds exceed [0, 1]. \
             First offender: index={idx}, lower={lo}, upper={hi}. \
             Target: propagation stack (design Packet C)"
        );
    }
    assert!(
        !verify_loose || any_exceeds_unit,
        "#4219: verifier/spec mismatch — certified bounds fit [0, 1] but \
         verify_graph returns Unknown(BoundsTooLoose). \
         Target: verifier/spec boundary (design Packet D)"
    );
    panic!(
        "#4219: unexpected outcome — verify={verify_result:?}, \
         exceeds_unit={any_exceeds_unit}, offender={offender:?}"
    );
}

/// Comparison harness for #4219: run both `verify_graph(...)` and
/// `certify_graph_bounds(...)` on the same short-seq talker softmax graph to
/// distinguish propagation tightness regression from verifier/spec plumbing
/// regression. Permanent regression pin. Part of #4219.
#[cfg_attr(not(debug_assertions), ntest::timeout(300000))]
#[test]
fn test_avoice_talker_crown_certify_vs_verify_regression_4219() {
    crate::test_fixtures::require_test_model_or_skip!("talker_attention_layer0.onnx");
    let (softmax_graph, _softmax_name, input, output_size) =
        talker_verifier::talker_softmax_verifier_setup();

    let output_bounds: Vec<Bound> = (0..output_size).map(|_| Bound::new(0.0, 1.0)).collect();

    let config = PropagationConfig {
        method: PropagationMethod::Crown,
        ..Default::default()
    };
    let verifier = Verifier::new_with_engine(config, Arc::new(NaiveCpuGemmEngine));

    // Path 1: verify_graph (includes spec check). 120s is the RELEASE
    // wall-clock budget; debug gets an unbounded budget (release_budget_ms).
    let spec = common::verifier_spec_from_bounded_input(
        &input,
        output_bounds,
        common::release_budget_ms(120_000),
    );
    let verify_result = verifier
        .verify_graph(&softmax_graph, &spec)
        .expect("verify_graph should not error on short-seq talker");

    // Path 2: certify_graph_bounds (bypasses spec check).
    let cert_result = verifier
        .certify_graph_bounds(
            "talker-softmax-short-seq",
            &softmax_graph,
            &input,
            Some(common::release_budget_ms(120_000)),
        )
        .expect("certify_graph_bounds should not error on short-seq talker");

    let cert = match &cert_result {
        BoundCertificationResult::Certified(c) => c,
        BoundCertificationResult::Timeout { .. } => {
            panic!("#4219: certify_graph_bounds timed out — #3701 is a completion contract")
        }
    };

    assert_eq!(
        cert.actual_method(),
        &MethodUsed::Crown,
        "#4219: certified method should be Crown, not {:?}",
        cert.actual_method()
    );

    diagnose_certify_vs_verify(&verify_result, cert);
}
