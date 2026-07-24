// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::common;
use super::super::duration_predictor::verifier_smoke as duration_verifier;
use super::shared::{
    assert_center_contained_in_verified_bounds, run_verifier_smoke_route, VerifierSmokeRoute,
};
use ny_core::{Bound, VerificationResult};

fn assert_duration_verifier_result(
    result: &VerificationResult,
    dur_graph: &ny_propagate::GraphNetwork,
    input: &ny_tensor::BoundedTensor,
    output_size: usize,
    max_duration: f32,
) {
    match result {
        VerificationResult::Verified {
            output_bounds,
            actual_method,
            ..
        } => {
            duration_verifier::assert_bounds_in_duration_range(
                output_bounds,
                output_size,
                max_duration,
                "duration verifier",
            );
            assert_center_contained_in_verified_bounds(
                output_bounds,
                dur_graph,
                input,
                "duration predictor verifier smoke",
            );
            eprintln!(
                "duration predictor verifier smoke: verified via {:?}, \
                 {output_size} bounds in [0, {max_duration}]",
                actual_method
            );
        }
        VerificationResult::Unknown { reason, bounds, .. } => {
            common::assert_unknown_verifier_bounds_sound(
                bounds,
                output_size,
                "duration predictor verifier smoke",
            );
            eprintln!(
                "duration predictor verifier smoke: unknown ({reason:?}), \
                 {} bounds finite+ordered, bounds={bounds:?}",
                bounds.len()
            );
        }
        VerificationResult::Timeout { .. } => {
            panic!("duration predictor verifier smoke timed out");
        }
        VerificationResult::Violated {
            counterexample,
            output,
            ..
        } => {
            panic!(
                "duration predictor verifier smoke violated: expected durations in \
                 [0, {max_duration}]; counterexample={:?}, output={:?}",
                &counterexample[..counterexample.len().min(8)],
                &output[..output.len().min(8)]
            );
        }
    }
}

/// Verify the surrogate duration predictor expected-duration property through
/// `Verifier::verify_graph(...)`. Expected durations from 50 Bernoulli sigmoid
/// bins are always in [0, 50].
///
/// Part of #3654.
#[cfg_attr(not(debug_assertions), ntest::timeout(120000))]
#[test]
fn test_avoice_duration_predictor_expected_duration_verifier_smoke_3654() {
    let (dur_graph, input, output_size, max_duration) =
        duration_verifier::duration_expected_duration_verifier_setup();

    let output_bounds: Vec<Bound> = (0..output_size)
        .map(|_| Bound::new(0.0, max_duration))
        .collect();
    let result = run_verifier_smoke_route(
        &dur_graph,
        &input,
        output_bounds,
        60_000,
        VerifierSmokeRoute::Ibp,
        "duration predictor verifier smoke",
    );
    assert_duration_verifier_result(&result, &dur_graph, &input, output_size, max_duration);
}
