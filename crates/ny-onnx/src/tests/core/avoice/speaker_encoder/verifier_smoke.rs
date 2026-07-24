// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Speaker encoder verifier-smoke helpers.
//!
//! Graph construction and scalar-nonnegative assertions for the speaker
//! verifier smokes. Test functions live in the root
//! `avoice/verifier_smoke.rs` entrypoint (#3950).

use super::super::common;
use ny_core::VerificationResult;
use ny_propagate::GraphNetwork;
use ny_tensor::BoundedTensor;

/// Build the speaker encoder norm-squared graph and bounded input.
pub(crate) fn speaker_norm_sq_verifier_setup() -> (GraphNetwork, BoundedTensor) {
    let (_dot_graph, norm_sq_graph, _reference_embedding) =
        super::cosine_head::build_speaker_cosine_component_graphs();
    let model = super::shared::avoice_speaker_encoder();
    let input = super::shared::bounded_speaker_encoder_cosine_input(
        model,
        super::shared::SPEAKER_ENCODER_SEQUENCE_LEN,
        super::shared::SPEAKER_ENCODER_EPSILON,
    );
    (norm_sq_graph, input)
}

/// Build the speaker encoder norm-squared graph via GraphModel round-trip.
///
/// Exercises the `OnnxModel -> GraphModel -> GraphNetwork` path that
/// builder-style downstream consumers use, then attaches the same
/// `norm_sq` head as the direct path.
pub(crate) fn speaker_graph_model_round_trip_norm_sq_verifier_setup(
) -> (GraphNetwork, BoundedTensor) {
    let (_dot_graph, norm_sq_graph, _reference_embedding) =
        super::cosine_head::build_speaker_cosine_component_graphs_round_trip();
    let model = super::shared::avoice_speaker_encoder();
    let input = super::shared::bounded_speaker_encoder_cosine_input(
        model,
        super::shared::SPEAKER_ENCODER_SEQUENCE_LEN,
        super::shared::SPEAKER_ENCODER_EPSILON,
    );
    (norm_sq_graph, input)
}

/// Assert that a scalar verification result is non-negative when Verified,
/// or structurally sound when Unknown.
pub(crate) fn assert_scalar_nonnegative_verified(result: &VerificationResult, label: &str) {
    match result {
        VerificationResult::Verified {
            output_bounds,
            actual_method,
            ..
        } => {
            assert_eq!(
                output_bounds.len(),
                1,
                "{label}: verifier output should be scalar (1 bound), got {}",
                output_bounds.len()
            );
            assert!(
                output_bounds[0].lower() >= 0.0,
                "{label}: lower bound should be non-negative, got {}",
                output_bounds[0].lower()
            );
            eprintln!(
                "{label}: verified via {:?}, bounds=[{}, {}]",
                actual_method,
                output_bounds[0].lower(),
                output_bounds[0].upper()
            );
        }
        VerificationResult::Unknown { reason, bounds, .. } => {
            common::assert_unknown_verifier_bounds_sound(bounds, 1, label);
            eprintln!(
                "{label}: unknown ({reason:?}), {} output bounds all finite and ordered, \
                 bounds={bounds:?}",
                bounds.len()
            );
        }
        VerificationResult::Timeout { .. } => {
            panic!("{label}: timed out — budget may need increase");
        }
        VerificationResult::Violated {
            counterexample,
            output,
            ..
        } => {
            panic!(
                "{label}: violated — value should always be >= 0; \
                 counterexample={counterexample:?}, output={output:?}"
            );
        }
    }
}
