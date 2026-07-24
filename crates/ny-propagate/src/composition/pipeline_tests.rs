// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for pipeline composition verification.
//!
//! Part of #3920.

use super::*;
use ndarray::{ArrayD, IxDyn};
use ny_core::{HeuristicUsed, MethodUsed};

fn make_bounds(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let l = ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).unwrap();
    let u = ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).unwrap();
    BoundedTensor::new(l, u).unwrap()
}

fn make_bounds_unchecked(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let l = ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).unwrap();
    let u = ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).unwrap();
    BoundedTensor::new_unchecked(l, u).expect("shape-only check")
}

fn make_certificate(
    model_id: &str,
    output_bounds: BoundedTensor,
    actual_method: MethodUsed,
    soundness: SoundnessProvenance,
) -> BoundCertificate {
    BoundCertificate::try_new(model_id, output_bounds, actual_method, soundness)
        .expect("supported stage certificate")
}

#[test]
fn test_pipeline_two_stage() {
    let mut pipeline = PipelineVerifier::new();

    // Stage 1: encoder maps [0, 1] -> [0.2, 0.8]
    pipeline
        .push_stage(
            make_bounds(&[0.0], &[1.0]),
            make_certificate(
                "encoder",
                make_bounds(&[0.2], &[0.8]),
                MethodUsed::Crown,
                SoundnessProvenance::sound(),
            ),
        )
        .unwrap();

    // Stage 2: decoder maps [0.0, 1.0] -> [0.5, 0.9]
    // Input bounds must contain the encoder output
    let decoder_input = make_bounds(&[0.0], &[1.0]);
    pipeline
        .push_stage(
            decoder_input.clone(),
            make_certificate(
                "decoder",
                make_bounds(&[0.5], &[0.9]),
                MethodUsed::Ibp,
                SoundnessProvenance::sound(),
            ),
        )
        .unwrap();

    let cert = pipeline.finalize().unwrap();
    assert_eq!(cert.stages().len(), 2);
    assert_eq!(cert.overall_provenance(), BoundProvenance::Ibp); // weakest link
    assert_eq!(cert.final_bounds().lower().as_slice().unwrap(), &[0.5]);
    assert_eq!(cert.final_bounds().upper().as_slice().unwrap(), &[0.9]);
    assert_eq!(
        cert.stages()[1].certificate().model_id(),
        "decoder",
        "pipeline should preserve stage certificate ordering"
    );
    assert_eq!(
        cert.stages()[1].input_bounds().lower().as_slice().unwrap(),
        decoder_input.lower().as_slice().unwrap(),
        "pipeline certificate should preserve the downstream input witness"
    );
}

#[test]
fn test_pipeline_push_stage_rejects_chain_violation() {
    let mut pipeline = PipelineVerifier::new();

    // Stage 1 output [0.0, 1.0]
    pipeline
        .push_stage(
            make_bounds(&[0.0], &[1.0]),
            make_certificate(
                "encoder",
                make_bounds(&[0.0], &[1.0]),
                MethodUsed::Crown,
                SoundnessProvenance::sound(),
            ),
        )
        .unwrap();

    // Stage 2 input [0.5, 0.8] — doesn't contain [0.0, 1.0]
    let result = pipeline.push_stage(
        make_bounds(&[0.5], &[0.8]),
        make_certificate(
            "decoder",
            make_bounds(&[0.6], &[0.7]),
            MethodUsed::Ibp,
            SoundnessProvenance::sound(),
        ),
    );
    assert!(result.is_err());
}

#[test]
fn test_pipeline_rejects_empty_stage_list() {
    let pipeline = PipelineVerifier::new();
    let result = pipeline.finalize().unwrap_err();
    assert!(
        matches!(result, NyError::InvalidConfig(ref message) if message.contains("no stages")),
        "unexpected empty-pipeline error: {result}"
    );
}

#[test]
fn test_pipeline_crown_is_weaker_than_alpha_crown_3920() {
    let mut pipeline = PipelineVerifier::new();
    pipeline
        .push_stage(
            make_bounds(&[0.0], &[1.0]),
            make_certificate(
                "encoder",
                make_bounds(&[0.25], &[0.75]),
                MethodUsed::AlphaCrown,
                SoundnessProvenance::sound(),
            ),
        )
        .unwrap();
    pipeline
        .push_stage(
            make_bounds(&[0.0], &[1.0]),
            make_certificate(
                "decoder",
                make_bounds(&[0.3], &[0.7]),
                MethodUsed::Crown,
                SoundnessProvenance::sound(),
            ),
        )
        .unwrap();

    let cert = pipeline.finalize().unwrap();
    assert_eq!(cert.overall_provenance(), BoundProvenance::Crown);
}

/// NaN bounds in a stage certificate must be caught before the IEEE 754
/// ordered comparison silently passes them through (#4007).
#[test]
fn test_pipeline_rejects_nan_in_stage_bounds_4007() {
    let mut pipeline = PipelineVerifier::new();

    // Stage 1 output has NaN — must use new_unchecked to bypass
    // BoundedTensor's checked constructor.
    let nan_bounds = make_bounds_unchecked(&[f32::NAN], &[1.0]);

    pipeline
        .push_stage(
            make_bounds(&[0.0], &[1.0]),
            make_certificate(
                "encoder",
                nan_bounds,
                MethodUsed::Crown,
                SoundnessProvenance::sound(),
            ),
        )
        .unwrap(); // first stage has no predecessor — no transition check

    // Stage 2 attempts to chain after the NaN output
    let result = pipeline.push_stage(
        make_bounds(&[0.0], &[2.0]),
        make_certificate(
            "decoder",
            make_bounds(&[0.1], &[0.9]),
            MethodUsed::Ibp,
            SoundnessProvenance::sound(),
        ),
    );
    assert!(
        result.is_err(),
        "push_stage must reject NaN bounds in stage transition"
    );
    let err = result.unwrap_err();
    assert!(
        matches!(err, NyError::NumericalInstability(ref msg) if msg.contains("NaN")),
        "expected NumericalInstability with NaN, got: {err}"
    );
}

#[test]
fn test_pipeline_rejects_nan_in_every_transition_slot_4007() {
    let cases = [
        (
            "previous lower",
            [f32::from_bits(0x7fc0_0001)],
            [1.0],
            [0.0],
            [2.0],
        ),
        (
            "previous upper",
            [0.0],
            [f32::from_bits(0xffc0_0001)],
            [0.0],
            [2.0],
        ),
        (
            "next lower",
            [0.0],
            [1.0],
            [f32::from_bits(0x7fa0_0001)],
            [2.0],
        ),
        ("next upper", [0.0], [1.0], [0.0], [f32::NAN]),
    ];

    for (label, prev_lower, prev_upper, next_lower, next_upper) in cases {
        let mut pipeline = PipelineVerifier::new();
        pipeline
            .push_stage(
                make_bounds(&[0.0], &[1.0]),
                make_certificate(
                    "encoder",
                    make_bounds_unchecked(&prev_lower, &prev_upper),
                    MethodUsed::Crown,
                    SoundnessProvenance::sound(),
                ),
            )
            .unwrap();

        let result = pipeline.push_stage(
            make_bounds_unchecked(&next_lower, &next_upper),
            make_certificate(
                "decoder",
                make_bounds(&[0.1], &[0.9]),
                MethodUsed::Ibp,
                SoundnessProvenance::sound(),
            ),
        );

        let err = result.expect_err("NaN transition slot must be rejected");
        assert!(
            matches!(err, NyError::NumericalInstability(ref msg) if msg.contains("NaN")),
            "{label} NaN should produce NumericalInstability, got: {err}"
        );
    }
}

#[test]
fn test_pipeline_merges_soundness_across_stages_3920() {
    let mut pipeline = PipelineVerifier::new();
    let heuristic = SoundnessProvenance::from_heuristics(vec![HeuristicUsed::SqrtNegativeDomain {
        num_nodes: 1,
    }]);

    pipeline
        .push_stage(
            make_bounds(&[0.0], &[1.0]),
            make_certificate(
                "encoder",
                make_bounds(&[0.25], &[0.75]),
                MethodUsed::Crown,
                heuristic.clone(),
            ),
        )
        .unwrap();
    pipeline
        .push_stage(
            make_bounds(&[0.0], &[1.0]),
            make_certificate(
                "decoder",
                make_bounds(&[0.3], &[0.7]),
                MethodUsed::Ibp,
                heuristic.clone(),
            ),
        )
        .unwrap();

    let cert = pipeline.finalize().unwrap();
    assert_eq!(
        cert.overall_soundness().heuristics_used(),
        heuristic.heuristics_used(),
        "pipeline should deduplicate and preserve stage soundness metadata"
    );
}

/// BetaCrown certificates cannot enter the pipeline because `BoundCertificate::try_new`
/// rejects `MethodUsed::BetaCrown` at construction time. This test guards the public
/// facade path reachable through `ny_api::composition::PipelineVerifier`.
/// Part of #3920 beta-crown certificate guard design.
#[test]
fn test_pipeline_verifier_rejects_beta_crown_stage_contract_3920() {
    let bounds = make_bounds(&[0.0], &[1.0]);
    let result = BoundCertificate::try_new(
        "encoder",
        bounds,
        MethodUsed::BetaCrown,
        SoundnessProvenance::sound(),
    );
    assert!(
        result.is_err(),
        "BoundCertificate::try_new must reject BetaCrown — pipeline cannot accept it"
    );
    let err = result.unwrap_err();
    assert!(
        matches!(err, NyError::UnsupportedOp(ref msg) if msg.contains("BetaCrown")),
        "expected UnsupportedOp for BetaCrown, got: {err}"
    );
}
