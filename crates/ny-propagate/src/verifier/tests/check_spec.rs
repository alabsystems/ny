// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// Licensed under the Apache License, Version 2.0

use super::super::*;
use ndarray::{ArrayD, IxDyn};
use ny_core::{Bound, SoundnessProvenance, VerificationResult};
use ny_tensor::BoundedTensor;

#[ntest::timeout(5000)]
#[test]
fn test_check_spec_verified() {
    let verifier = Verifier::new(PropagationConfig::default());
    let output = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.8, 1.5]).unwrap(),
    )
    .unwrap();
    let required = vec![Bound::new(0.0, 1.0), Bound::new(0.5, 2.0)];

    let result = verifier
        .check_spec(&output, &required, None, SoundnessProvenance::sound())
        .unwrap();

    assert!(matches!(result, VerificationResult::Verified { .. }));
    if let VerificationResult::Verified { output_bounds, .. } = result {
        assert_eq!(output_bounds.len(), 2);
        assert_eq!(output_bounds[0].lower(), 0.5);
        assert_eq!(output_bounds[0].upper(), 0.8);
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_check_spec_unknown_lower_too_low() {
    let verifier = Verifier::new(PropagationConfig::default());
    let output = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-0.5, 1.0]).unwrap(), // -0.5 < required 0.0
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.8, 1.5]).unwrap(),
    )
    .unwrap();
    let required = vec![Bound::new(0.0, 1.0), Bound::new(0.5, 2.0)];

    let result = verifier
        .check_spec(&output, &required, None, SoundnessProvenance::sound())
        .unwrap();

    assert!(matches!(result, VerificationResult::Unknown { .. }));
    if let VerificationResult::Unknown { reason, .. } = result {
        // Now uses BoundsTooLoose with gap instead of string reason
        assert!(matches!(
            reason,
            ny_core::UnknownReason::BoundsTooLoose { .. }
        ));
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_check_spec_unknown_upper_too_high() {
    let verifier = Verifier::new(PropagationConfig::default());
    let output = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.5, 1.5]).unwrap(), // 1.5 > required 1.0
    )
    .unwrap();
    let required = vec![Bound::new(0.0, 1.0), Bound::new(0.5, 2.0)];

    let result = verifier
        .check_spec(&output, &required, None, SoundnessProvenance::sound())
        .unwrap();

    assert!(matches!(result, VerificationResult::Unknown { .. }));
    if let VerificationResult::Unknown { reason, .. } = result {
        // Now uses BoundsTooLoose with gap instead of string reason
        assert!(matches!(
            reason,
            ny_core::UnknownReason::BoundsTooLoose { .. }
        ));
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_check_spec_second_output_violates() {
    let verifier = Verifier::new(PropagationConfig::default());
    let output = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.0]).unwrap(), // second: 0.0 < required 0.5
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.8, 1.5]).unwrap(),
    )
    .unwrap();
    let required = vec![Bound::new(0.0, 1.0), Bound::new(0.5, 2.0)];

    let result = verifier
        .check_spec(&output, &required, None, SoundnessProvenance::sound())
        .unwrap();

    assert!(matches!(result, VerificationResult::Unknown { .. }));
    if let VerificationResult::Unknown { reason, .. } = result {
        // Now uses BoundsTooLoose with gap instead of string reason
        assert!(matches!(
            reason,
            ny_core::UnknownReason::BoundsTooLoose { .. }
        ));
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_check_spec_exactly_at_bounds() {
    let verifier = Verifier::new(PropagationConfig::default());
    let output = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.5]).unwrap(), // exactly at lower bounds
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(), // exactly at upper bounds
    )
    .unwrap();
    let required = vec![Bound::new(0.0, 1.0), Bound::new(0.5, 2.0)];

    let result = verifier
        .check_spec(&output, &required, None, SoundnessProvenance::sound())
        .unwrap();

    assert!(matches!(result, VerificationResult::Verified { .. }));
}

#[ntest::timeout(5000)]
#[test]
fn test_check_spec_with_infinities() {
    let verifier = Verifier::new(PropagationConfig::default());
    let output = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.5]).unwrap(),
    )
    .unwrap();
    let required = vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)];

    let result = verifier
        .check_spec(&output, &required, None, SoundnessProvenance::sound())
        .unwrap();

    assert!(matches!(result, VerificationResult::Verified { .. }));
}

/// Regression test for #2238: empty output spec must be rejected, not trivially verified.
///
/// Before #2238, an empty spec would iterate zero times in check_spec and report
/// `Verified` — a false positive (the most dangerous verification bug class).
#[ntest::timeout(5000)]
#[test]
fn test_check_spec_empty_returns_error_2238() {
    let verifier = Verifier::new(PropagationConfig::default());
    let output = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap(),
    )
    .unwrap();
    let required: Vec<Bound> = vec![];

    let result = verifier.check_spec(&output, &required, None, SoundnessProvenance::sound());

    assert!(
        result.is_err(),
        "Empty spec must be rejected, not trivially verified"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("empty"),
        "Error should mention empty output_bounds, got: {err_msg}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_check_spec_multidimensional_output() {
    let verifier = Verifier::new(PropagationConfig::default());
    // 2x2 output, will be flattened to 4 elements
    let output = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, 0.6, 0.7, 0.8]).unwrap(),
    )
    .unwrap();
    let required = vec![
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
    ];

    let result = verifier
        .check_spec(&output, &required, None, SoundnessProvenance::sound())
        .unwrap();

    assert!(matches!(result, VerificationResult::Verified { .. }));
}

#[ntest::timeout(5000)]
#[test]
fn test_check_spec_partial_match() {
    // First output matches, second doesn't
    let verifier = Verifier::new(PropagationConfig::default());
    let output = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.8, 2.5]).unwrap(), // second upper too high
    )
    .unwrap();
    let required = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 2.0)];

    let result = verifier
        .check_spec(&output, &required, None, SoundnessProvenance::sound())
        .unwrap();

    // Should detect the violation in second output
    assert!(matches!(result, VerificationResult::Unknown { .. }));
    if let VerificationResult::Unknown { reason, .. } = result {
        // Now uses BoundsTooLoose with gap instead of string reason
        assert!(matches!(
            reason,
            ny_core::UnknownReason::BoundsTooLoose { .. }
        ));
    }
}

/// Regression test for #2230: zip truncation silently skips spec requirements
/// when the network produces fewer outputs than the spec requires.
#[ntest::timeout(5000)]
#[test]
fn test_check_spec_fewer_outputs_than_required_returns_error() {
    let verifier = Verifier::new(PropagationConfig::default());
    // Network produces 1 output
    let output = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.8]).unwrap(),
    )
    .unwrap();
    // Spec requires 3 outputs — prior to fix, requirements 2 and 3 were silently skipped
    let required = vec![
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
    ];

    let result = verifier.check_spec(&output, &required, None, SoundnessProvenance::sound());

    assert!(
        result.is_err(),
        "Expected error for output/spec dimension mismatch"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("1 output bounds") && err_msg.contains("requires 3"),
        "Error message should mention the dimension mismatch, got: {err_msg}"
    );
}

/// Extra outputs beyond spec requirements are fine — only the spec-required
/// outputs are checked.
#[ntest::timeout(5000)]
#[test]
fn test_check_spec_more_outputs_than_required_is_ok() {
    let verifier = Verifier::new(PropagationConfig::default());
    // Network produces 3 outputs
    let output = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5, 0.5, 0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.8, 0.8, 0.8]).unwrap(),
    )
    .unwrap();
    // Spec only requires 1 output
    let required = vec![Bound::new(0.0, 1.0)];

    let result = verifier
        .check_spec(&output, &required, None, SoundnessProvenance::sound())
        .unwrap();

    assert!(matches!(result, VerificationResult::Verified { .. }));
}
