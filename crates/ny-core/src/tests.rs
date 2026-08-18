// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ny_test_utils::{assert_f32_close, assert_f32_nan as assert_nan};

fn assert_contains(haystack: &str, needle: &str) {
    assert!(
        haystack.contains(needle),
        "expected {haystack:?} to contain {needle:?}"
    );
}

#[test]
fn test_bound_operations() {
    let a = Bound::new(0.0, 1.0);
    let b = Bound::new(0.5, 1.5);

    assert!(a.contains(0.5), "{a:?} should contain 0.5");
    assert!(!a.contains(1.5), "{a:?} should not contain 1.5");

    let intersection = a.intersect(&b).unwrap();
    assert_eq!(intersection.lower, 0.5);
    assert_eq!(intersection.upper, 1.0);

    let union = a.union(&b);
    assert_eq!(union.lower, 0.0);
    assert_eq!(union.upper, 1.5);
}

#[test]
#[should_panic(expected = "Bound::new: invalid bound")]
fn test_bound_new_rejects_inverted_bounds() {
    let _ = Bound::new(1.0, 0.5);
}

#[test]
#[should_panic(expected = "Bound::new: lower bound is not finite")]
fn test_bound_new_rejects_nan_lower() {
    let _ = Bound::new(f32::NAN, 1.0);
}

#[test]
#[should_panic(expected = "Bound::new: upper bound is not finite")]
fn test_bound_new_rejects_infinite_upper() {
    let _ = Bound::new(0.0, f32::INFINITY);
}

#[test]
#[should_panic(expected = "Bound::new_allow_infinite: lower bound is NaN")]
fn test_bound_new_allow_infinite_rejects_nan_lower() {
    let _ = Bound::new_allow_infinite(f32::NAN, 1.0);
}

#[test]
#[should_panic(expected = "Bound::new_allow_infinite: upper bound is NaN")]
fn test_bound_new_allow_infinite_rejects_nan_upper() {
    let _ = Bound::new_allow_infinite(0.0, f32::NAN);
}

#[test]
#[should_panic(expected = "Bound::new_allow_infinite: invalid bound")]
fn test_bound_new_allow_infinite_rejects_inverted_bounds() {
    let _ = Bound::new_allow_infinite(1.0, 0.5);
}

#[test]
fn test_concrete_bound() {
    let b = Bound::concrete(0.5);
    assert_eq!(b.width(), 0.0);
    assert!(b.is_tight(0.001), "{b:?} should be tight within 0.001");
}

// Tests to catch surviving mutations
#[test]
fn test_bound_width_computation() {
    // Test width = upper - lower (not lower - upper or any other formula)
    let b = Bound::new(1.0, 3.0);
    assert_eq!(b.width(), 2.0); // 3.0 - 1.0 = 2.0

    // Verify width is positive for valid bounds
    let b2 = Bound::new(-5.0, 5.0);
    assert_eq!(b2.width(), 10.0); // 5.0 - (-5.0) = 10.0

    // Test with negative bounds
    let b3 = Bound::new(-10.0, -3.0);
    assert_eq!(b3.width(), 7.0); // -3.0 - (-10.0) = 7.0

    // Ensure width distinguishes different bounds
    let narrow = Bound::new(0.0, 0.1);
    let wide = Bound::new(0.0, 10.0);
    assert!(
        narrow.width() < wide.width(),
        "narrow bound {narrow:?} should be tighter than {wide:?}"
    );
    assert_eq!(narrow.width(), 0.1);
    assert_eq!(wide.width(), 10.0);
}

#[test]
fn test_bound_is_tight_boundary_conditions() {
    // Test boundary conditions for is_tight
    let b = Bound::new(0.0, 0.5);
    assert_eq!(b.width(), 0.5);

    // Exactly at epsilon boundary - should be tight (width <= epsilon)
    assert!(b.is_tight(0.5), "{b:?} should be tight at epsilon 0.5");

    // Epsilon just below width - should NOT be tight
    for epsilon in [0.49, 0.4, 0.1] {
        assert!(
            !b.is_tight(epsilon),
            "{b:?} should not be tight at epsilon {epsilon}"
        );
    }

    // Epsilon just above width - should be tight
    for epsilon in [0.51, 1.0] {
        assert!(
            b.is_tight(epsilon),
            "{b:?} should be tight at epsilon {epsilon}"
        );
    }

    // Concrete bounds are always tight
    let concrete = Bound::concrete(5.0);
    for epsilon in [0.0, 0.0001, f32::EPSILON] {
        assert!(
            concrete.is_tight(epsilon),
            "{concrete:?} should be tight at epsilon {epsilon}"
        );
    }

    // Wide bounds need large epsilon
    let wide = Bound::new(0.0, 100.0);
    assert!(
        !wide.is_tight(99.0),
        "{wide:?} should not be tight at epsilon 99.0"
    );
    for epsilon in [100.0, 101.0] {
        assert!(
            wide.is_tight(epsilon),
            "{wide:?} should be tight at epsilon {epsilon}"
        );
    }
}

#[test]
fn test_bound_is_unbounded_all_cases() {
    // Finite bounds - NOT unbounded
    let finite = Bound::new(-1e10, 1e10);
    assert!(!finite.is_unbounded(), "{finite:?} should stay bounded");

    // Lower is infinite
    let lower_inf = Bound::new_allow_infinite(f32::NEG_INFINITY, 0.0);
    assert!(
        lower_inf.is_unbounded(),
        "{lower_inf:?} should be unbounded"
    );

    // Upper is infinite
    let upper_inf = Bound::new_allow_infinite(0.0, f32::INFINITY);
    assert!(
        upper_inf.is_unbounded(),
        "{upper_inf:?} should be unbounded"
    );

    // Both infinite
    let both_inf = Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY);
    assert!(both_inf.is_unbounded(), "{both_inf:?} should be unbounded");

    // Edge case: very large but finite
    let large = Bound::new(-f32::MAX, f32::MAX);
    assert!(!large.is_unbounded(), "{large:?} should remain bounded");

    // Zero bounds - NOT unbounded
    let zero = Bound::concrete(0.0);
    assert!(!zero.is_unbounded(), "{zero:?} should remain bounded");
}

#[test]
fn test_verification_result_is_verified_all_variants() {
    // Verified variant - MUST return true
    let verified = VerificationResult::Verified {
        provenance: SoundnessProvenance::default(),
        output_bounds: vec![Bound::new(0.0, 1.0)],
        proof: None,
        actual_method: None,
    };
    assert!(
        verified.is_verified(),
        "verified results must report is_verified()"
    );

    // Violated variant - MUST return false
    let violated = VerificationResult::Violated {
        provenance: SoundnessProvenance::default(),
        counterexample: vec![0.5],
        output: vec![1.5],
        details: None,
        actual_method: None,
    };
    assert!(
        !violated.is_verified(),
        "violated results must not report is_verified()"
    );

    // Unknown variant - MUST return false
    let unknown = VerificationResult::Unknown {
        provenance: SoundnessProvenance::default(),
        bounds: vec![Bound::new(-1.0, 2.0)],
        reason: UnknownReason::BoundsTooLoose { gap: None },
        actual_method: None,
    };
    assert!(
        !unknown.is_verified(),
        "unknown results must not report is_verified()"
    );

    // Timeout variant with partial bounds - MUST return false
    let timeout_partial = VerificationResult::Timeout {
        provenance: SoundnessProvenance::default(),
        partial_bounds: Some(vec![Bound::new(0.0, 1.0)]),
        actual_method: None,
    };
    assert!(
        !timeout_partial.is_verified(),
        "timeout results with partial bounds must not report is_verified()"
    );

    // Timeout variant without partial bounds - MUST return false
    let timeout_none = VerificationResult::Timeout {
        provenance: SoundnessProvenance::default(),
        partial_bounds: None,
        actual_method: None,
    };
    assert!(
        !timeout_none.is_verified(),
        "timeout results without bounds must not report is_verified()"
    );
}

#[test]
fn test_bound_width_distinguishes_bounds() {
    // Ensure width is sensitive to both lower and upper changes
    let base = Bound::new(0.0, 1.0);
    let wider_upper = Bound::new(0.0, 2.0);
    let wider_lower = Bound::new(-1.0, 1.0);

    assert!(
        wider_upper.width() > base.width(),
        "{wider_upper:?} should be wider than {base:?}"
    );
    assert!(
        wider_lower.width() > base.width(),
        "{wider_lower:?} should be wider than {base:?}"
    );
    assert_eq!(wider_upper.width(), wider_lower.width()); // Both are width 2.0
}

#[test]
fn test_bound_from_range_inclusive() {
    let range = 0.5f32..=1.5f32;
    let bound: Bound = range.into();
    assert_eq!(bound.lower, 0.5);
    assert_eq!(bound.upper, 1.5);
    assert_eq!(bound.width(), 1.0);
}

#[test]
fn test_intersect_disjoint_returns_none() {
    let a = Bound::new(0.0, 1.0);
    let b = Bound::new(2.0, 3.0);
    assert!(
        a.intersect(&b).is_none(),
        "disjoint bounds {a:?} and {b:?} should not intersect"
    );
}

#[test]
fn test_custom_op_schema_registry_resolution_precedence() {
    let mut registry = CustomOpSchemaRegistry::default();
    registry
        .register(CustomOpSpec::new("custom", "Foo", None))
        .unwrap();
    registry
        .register(CustomOpSpec::new("custom", "Foo", Some(2)))
        .unwrap();
    registry
        .register(CustomOpSpec::new("custom", "Foo", Some(5)))
        .unwrap();

    let exact = registry.resolve("custom", "Foo", Some(5)).unwrap();
    assert_eq!(exact.opset_version, Some(5));

    let fallback_version = registry.resolve("custom", "Foo", Some(4)).unwrap();
    assert_eq!(fallback_version.opset_version, Some(2));

    let fallback_unversioned = registry.resolve("custom", "Foo", Some(1)).unwrap();
    assert_eq!(fallback_unversioned.opset_version, None);

    let prefer_unversioned = registry.resolve("custom", "Foo", None).unwrap();
    assert_eq!(prefer_unversioned.opset_version, None);

    let mut version_only = CustomOpSchemaRegistry::default();
    version_only
        .register(CustomOpSpec::new("custom", "Bar", Some(3)))
        .unwrap();
    version_only
        .register(CustomOpSpec::new("custom", "Bar", Some(7)))
        .unwrap();
    let highest = version_only.resolve("custom", "Bar", None).unwrap();
    assert_eq!(highest.opset_version, Some(7));
}

#[test]
fn test_custom_op_schema_registry_debug_listing() {
    let mut registry = CustomOpSchemaRegistry::default();
    registry
        .register(CustomOpSpec::new("custom", "Foo", None))
        .unwrap();
    let listing = registry.debug_listing();
    assert_contains(&listing, "CustomOpSchemaRegistry:");
    assert_contains(&listing, "domain=\"custom\"");
    assert_contains(&listing, "op_type=\"Foo\"");
    assert_contains(&listing, "opset_version=any");
}

#[test]
fn test_custom_op_schema_registry_rejects_duplicate_attributes() {
    let schema = CustomOpSchema::new(
        1,
        Some(2),
        1,
        Some(1),
        vec![
            CustomOpAttribute::new("alpha", CustomOpAttributeType::Float),
            CustomOpAttribute::new("alpha", CustomOpAttributeType::Float),
        ],
    );
    let spec = CustomOpSpec::with_schema("custom", "DupAttr", None, schema);
    let mut registry = CustomOpSchemaRegistry::default();
    let err = registry.register(spec).unwrap_err();
    match err {
        NyError::InvalidSpec(msg) => {
            assert_contains(&msg, "duplicate attribute");
            assert_contains(&msg, "alpha");
        }
        other => panic!("expected InvalidSpec, got {other:?}"),
    }
}

#[test]
fn test_custom_op_schema_registry_rejects_invalid_arity() {
    let schema = CustomOpSchema::new(3, Some(1), 1, Some(2), Vec::new());
    let spec = CustomOpSpec::with_schema("custom", "BadArity", None, schema);
    let mut registry = CustomOpSchemaRegistry::default();
    let err = registry.register(spec).unwrap_err();
    match err {
        NyError::InvalidSpec(msg) => {
            assert_contains(&msg, "min_inputs");
            assert_contains(&msg, "max_inputs");
        }
        other => panic!("expected InvalidSpec, got {other:?}"),
    }
}

#[test]
fn test_custom_op_schema_registry_rejects_default_type_mismatch() {
    let schema = CustomOpSchema::new(
        1,
        Some(1),
        1,
        Some(1),
        vec![CustomOpAttribute {
            name: "axis".to_string(),
            attr_type: CustomOpAttributeType::Int,
            required: false,
            default_value: Some(CustomOpAttributeValue::Float(1.0)),
        }],
    );
    let spec = CustomOpSpec::with_schema("custom", "BadDefault", None, schema);
    let mut registry = CustomOpSchemaRegistry::default();
    let err = registry.register(spec).unwrap_err();
    match err {
        NyError::InvalidSpec(msg) => {
            assert_contains(&msg, "default type mismatch");
            assert_contains(&msg, "axis");
        }
        other => panic!("expected InvalidSpec, got {other:?}"),
    }
}

#[test]
fn test_custom_op_schema_registry_rejects_duplicate_op_registration() {
    let mut registry = CustomOpSchemaRegistry::default();
    registry
        .register(CustomOpSpec::new("custom", "MyOp", Some(1)))
        .expect("first registration should succeed");

    // Exact duplicate: same domain, op_type, opset_version
    let err = registry
        .register(CustomOpSpec::new("custom", "MyOp", Some(1)))
        .unwrap_err();
    match err {
        NyError::InvalidSpec(msg) => {
            assert_contains(&msg, "Duplicate");
            assert_contains(&msg, "custom");
            assert_contains(&msg, "MyOp");
        }
        other => panic!("expected InvalidSpec, got {other:?}"),
    }
}

#[test]
fn test_custom_op_schema_registry_rejects_invalid_output_arity() {
    // min_outputs > max_outputs
    let schema = CustomOpSchema::new(1, Some(1), 5, Some(2), Vec::new());
    let spec = CustomOpSpec::with_schema("custom", "BadOutputArity", None, schema);
    let mut registry = CustomOpSchemaRegistry::default();
    let err = registry.register(spec).unwrap_err();
    match err {
        NyError::InvalidSpec(msg) => {
            assert_contains(&msg, "min_outputs");
            assert_contains(&msg, "max_outputs");
        }
        other => panic!("expected InvalidSpec, got {other:?}"),
    }
}

#[test]
fn test_custom_op_schema_registry_rejects_empty_attribute_name() {
    let schema = CustomOpSchema::new(
        1,
        Some(1),
        1,
        Some(1),
        vec![CustomOpAttribute {
            name: "   ".to_string(), // whitespace-only name
            attr_type: CustomOpAttributeType::Int,
            required: false,
            default_value: None,
        }],
    );
    let spec = CustomOpSpec::with_schema("custom", "EmptyAttrName", None, schema);
    let mut registry = CustomOpSchemaRegistry::default();
    let err = registry.register(spec).unwrap_err();
    match err {
        NyError::InvalidSpec(msg) => {
            assert_contains(&msg, "empty attribute name");
        }
        other => panic!("expected InvalidSpec, got {other:?}"),
    }
}

#[test]
fn test_contains_edge_cases() {
    let b = Bound::new(0.0, 1.0);
    // Boundary values should be contained (inclusive)
    assert!(b.contains(0.0), "{b:?} should contain its lower bound");
    assert!(b.contains(1.0), "{b:?} should contain its upper bound");
    // Just outside should not be contained
    assert!(
        !b.contains(-0.0001),
        "{b:?} should not contain values below its lower bound"
    );
    assert!(
        !b.contains(1.0001),
        "{b:?} should not contain values above its upper bound"
    );
}

// Tests for informative counterexample types
#[test]
fn test_violated_constraint_detect_below_lower() {
    let output = vec![0.5, -0.1, 0.8];
    let bounds = vec![
        Bound::new(0.0, 1.0), // ok
        Bound::new(0.0, 1.0), // violated: -0.1 < 0.0
        Bound::new(0.0, 1.0), // ok
    ];

    let vc = ViolatedConstraint::detect(&output, &bounds).unwrap();
    assert_eq!(vc.output_idx(), 1);
    assert_eq!(vc.actual_value(), -0.1);
    assert_eq!(vc.violation_type(), ViolationType::BelowLower);
    assert_f32_close(
        vc.violation_amount(),
        0.1,
        1e-6,
        "below-lower violation amount",
    );
}

#[test]
fn test_violated_constraint_detect_above_upper() {
    let output = vec![0.5, 0.3, 1.5];
    let bounds = vec![
        Bound::new(0.0, 1.0), // ok
        Bound::new(0.0, 1.0), // ok
        Bound::new(0.0, 1.0), // violated: 1.5 > 1.0
    ];

    let vc = ViolatedConstraint::detect(&output, &bounds).unwrap();
    assert_eq!(vc.output_idx(), 2);
    assert_eq!(vc.actual_value(), 1.5);
    assert_eq!(vc.violation_type(), ViolationType::AboveUpper);
    assert_f32_close(
        vc.violation_amount(),
        0.5,
        1e-6,
        "above-upper violation amount",
    );
}

#[test]
fn test_violated_constraint_detect_no_violation() {
    let output = vec![0.5, 0.3, 0.8];
    let bounds = vec![
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
    ];

    assert!(
        ViolatedConstraint::detect(&output, &bounds).is_none(),
        "outputs within bounds should not produce a violated constraint"
    );
}

#[test]
fn test_violated_constraint_explain() {
    let vc = ViolatedConstraint::new(0, 1.5, Bound::new(0.0, 1.0), ViolationType::AboveUpper, 0.5);

    let explanation = vc.explain();
    assert_contains(&explanation, "Output[0]");
    assert_contains(&explanation, "1.5");
    assert_contains(&explanation, "upper bound");
    assert_contains(&explanation, "1.0");
}

#[test]
fn test_layer_output_new() {
    let values = vec![1.0, -2.0, 3.0, 0.5];
    let layer = LayerOutput::new(0, Some("relu1".to_string()), LayerType::ReLU, values);

    assert_eq!(layer.layer_idx(), 0);
    assert_eq!(layer.layer_name(), Some("relu1"));
    assert_eq!(*layer.layer_type(), LayerType::ReLU);
    assert_eq!(layer.min_value(), -2.0);
    assert_eq!(layer.max_value(), 3.0);
    assert_eq!(layer.values().len(), 4);
}

#[test]
fn test_layer_output_new_empty_values_nan_min_max() {
    let layer = LayerOutput::new(0, None, LayerType::Linear, Vec::new());
    assert!(
        layer.values().is_empty(),
        "empty layer output should retain an empty value vector"
    );
    assert_nan(layer.min_value(), "empty layer output min");
    assert_nan(layer.max_value(), "empty layer output max");
}

#[test]
fn test_layer_output_new_non_finite_values_returns_error() {
    let values = vec![1.0, f32::NAN, 2.0];
    let err = LayerOutput::try_new(0, None, LayerType::Linear, values)
        .expect_err("expected non-finite values to be rejected");
    match err {
        NyError::NumericalInstability(message) => {
            assert_contains(&message, "non-finite value at index 1");
        }
        other => panic!("unexpected error type: {other:?}"),
    }
}

#[test]
#[should_panic(expected = "non-finite value at index 1")]
fn test_layer_output_new_non_finite_values_panics_nan() {
    let values = vec![1.0, f32::NAN, 2.0];
    let _ = LayerOutput::new(0, None, LayerType::Linear, values);
}

#[test]
#[should_panic(expected = "non-finite value at index 1")]
fn test_layer_output_new_non_finite_values_panics_inf() {
    let values = vec![1.0, f32::INFINITY, 2.0];
    let _ = LayerOutput::new(0, None, LayerType::Linear, values);
}

#[test]
fn test_layer_output_try_new_non_finite_values_inf_error() {
    let values = vec![1.0, f32::INFINITY, 2.0];
    let err = LayerOutput::try_new(0, None, LayerType::Linear, values)
        .expect_err("expected non-finite values to be rejected");
    match err {
        NyError::NumericalInstability(message) => {
            assert_contains(&message, "non-finite value at index 1");
        }
        other => panic!("unexpected error type: {other:?}"),
    }
}

#[test]
fn test_informative_counterexample_new() {
    let input = vec![0.5, 0.5];
    let output = vec![1.5]; // violates [0, 1]
    let bounds = vec![Bound::new(0.0, 1.0)];

    let ce = InformativeCounterexample::new(input.clone(), output.clone(), Some(&bounds));

    assert_eq!(ce.input(), input.as_slice());
    assert_eq!(ce.output(), output.as_slice());
    assert!(
        ce.violated_constraint().is_some(),
        "counterexample should record the violated constraint"
    );
    assert_contains(ce.explanation(), "Property violated");
}

#[test]
fn test_informative_counterexample_bounds_length_mismatch_note() {
    let input = vec![0.5];
    let output = vec![1.5];
    let bounds = vec![Bound::new(0.0, 1.0), Bound::new(0.0, 2.0)];

    let ce = InformativeCounterexample::new(input, output, Some(&bounds));

    assert!(
        ce.violated_constraint().is_none(),
        "mismatched output bounds should not fabricate a violated constraint"
    );
    assert_contains(ce.explanation(), "output_bounds length mismatch");
}

#[test]
fn test_informative_counterexample_format_trace() {
    let mut ce = InformativeCounterexample::new(vec![1.0], vec![2.0], None);

    // Empty trace
    assert_contains(&ce.format_trace(), "No layer trace");

    // Add layers
    ce.add_layer_output(LayerOutput::new(0, None, LayerType::Linear, vec![1.5]));
    ce.add_layer_output(LayerOutput::new(
        1,
        Some("relu".to_string()),
        LayerType::ReLU,
        vec![2.0],
    ));

    let trace_str = ce.format_trace();
    assert_contains(&trace_str, "Layer   0");
    assert_contains(&trace_str, "Layer   1");
    assert_contains(&trace_str, "Linear");
    assert_contains(&trace_str, "ReLU");
}

#[test]
fn test_informative_counterexample_with_trace() {
    let trace = vec![
        LayerOutput::new(0, None, LayerType::Linear, vec![1.0]),
        LayerOutput::new(1, None, LayerType::ReLU, vec![1.0]),
    ];

    let ce = InformativeCounterexample::new(vec![0.5], vec![1.0], None).with_trace(trace);

    assert_eq!(ce.trace().len(), 2);
}

// Tests for VerificationProof
#[test]
fn test_verification_proof_alethe() {
    let proof_text = "(assume h1 (> x 0))".to_string();
    let proof = VerificationProof::alethe(proof_text.clone());

    assert_eq!(proof.format(), ProofFormat::Alethe);
    assert!(
        proof.num_steps().is_none(),
        "alethe proofs without explicit steps should report no step count"
    );
    let stats = proof
        .stats()
        .expect("invariant: VerificationProof::alethe always sets stats");
    assert_eq!(stats.size_bytes(), proof_text.len());
    assert_eq!(proof.as_bytes(), proof_text.as_bytes());
}

#[test]
fn test_verification_proof_alethe_with_stats() {
    let proof_text = "(step s1 (resolution h1 h2))".to_string();
    let stats = ProofStats::new(2, 1, 0, 0); // size_bytes will be overwritten

    let proof = VerificationProof::alethe_with_stats(proof_text.clone(), 3, stats);

    assert_eq!(proof.format(), ProofFormat::Alethe);
    assert_eq!(proof.num_steps(), Some(3));
    let stats = proof
        .stats()
        .expect("invariant: VerificationProof::alethe_with_stats always sets stats");
    assert_eq!(stats.num_assumptions(), 2);
    assert_eq!(stats.num_resolutions(), 1);
    assert_eq!(stats.size_bytes(), proof_text.len());
}

#[test]
fn test_verification_proof_as_text() {
    // Alethe format should return text
    let proof_text = "(step s1 (resolution))";
    let proof = VerificationProof::alethe(proof_text.to_string());
    assert_eq!(proof.as_text(), Some(proof_text));

    // Binary format (Drat) should return None
    let binary_proof =
        VerificationProof::from_parts(ProofFormat::Drat, vec![0x01, 0x02, 0x03], None, None);
    assert!(
        binary_proof.as_text().is_none(),
        "DRAT proofs are binary and should not expose text"
    );

    // BoundTrace format should return None
    let bound_proof = VerificationProof::from_parts(
        ProofFormat::BoundTrace,
        vec![0xDE, 0xAD, 0xBE, 0xEF],
        None,
        None,
    );
    assert!(
        bound_proof.as_text().is_none(),
        "bound-trace proofs should not expose text"
    );

    // LFSC format should return text
    let lfsc_proof = VerificationProof::from_parts(
        ProofFormat::Lfsc,
        b"(check (holds true))".to_vec(),
        None,
        None,
    );
    assert!(
        lfsc_proof.as_text().is_some(),
        "LFSC proofs should expose text payloads"
    );
}

#[test]
fn test_verification_proof_as_bytes() {
    let proof_text = "proof content";
    let proof = VerificationProof::alethe(proof_text.to_string());
    assert_eq!(proof.as_bytes(), proof_text.as_bytes());

    // Binary data
    let binary_data = vec![0x01, 0x02, 0x03, 0x04];
    let binary_proof =
        VerificationProof::from_parts(ProofFormat::Drat, binary_data.clone(), None, None);
    assert_eq!(binary_proof.as_bytes(), &binary_data);
}

#[test]
fn test_proof_stats_default() {
    let stats = ProofStats::default();
    assert_eq!(stats.num_assumptions(), 0);
    assert_eq!(stats.num_resolutions(), 0);
    assert_eq!(stats.num_theory_lemmas(), 0);
    assert_eq!(stats.size_bytes(), 0);
}

#[test]
fn test_proof_format_equality() {
    assert_eq!(ProofFormat::Alethe, ProofFormat::Alethe);
    assert_ne!(ProofFormat::Alethe, ProofFormat::Lfsc);
    assert_ne!(ProofFormat::Drat, ProofFormat::BoundTrace);
}

#[test]
fn test_verification_proof_save_to_file() {
    use std::io::Read;
    let proof_text = "test proof data";
    let proof = VerificationProof::alethe(proof_text.to_string());

    // Use unique temp directory to avoid race conditions in parallel test runs.
    let temp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let temp_path = temp_dir.path().join("test_proof_export.alethe");

    // Save and read back
    proof.save_to_file(&temp_path).unwrap();
    let mut file = std::fs::File::open(&temp_path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    assert_eq!(contents, proof_text);
    // temp_dir auto-cleans on drop
}

// Tests for NyError
#[test]
fn test_ny_error_shape_mismatch_display() {
    let err = NyError::ShapeMismatch {
        expected: vec![2, 3],
        got: vec![2, 4],
    };
    let msg = format!("{err}");
    assert_contains(&msg, "Shape mismatch");
    assert_contains(&msg, "[2, 3]");
    assert_contains(&msg, "[2, 4]");
}

#[test]
fn test_ny_error_unsupported_layer_display() {
    let err = NyError::UnsupportedLayer("CustomOp".to_string());
    let msg = format!("{err}");
    assert_contains(&msg, "Unsupported layer type");
    assert_contains(&msg, "CustomOp");
}

#[test]
fn test_ny_error_unsupported_op_display() {
    let err = NyError::UnsupportedOp("Einsum".to_string());
    let msg = format!("{err}");
    assert_contains(&msg, "Unsupported operation");
    assert_contains(&msg, "Einsum");
}

#[test]
fn test_ny_error_model_load_display() {
    let err = NyError::ModelLoad("file not found".to_string());
    let msg = format!("{err}");
    assert_contains(&msg, "Model loading failed");
    assert_contains(&msg, "file not found");
}

#[test]
fn test_ny_error_invalid_spec_display() {
    let err = NyError::InvalidSpec("empty bounds".to_string());
    let msg = format!("{err}");
    assert_contains(&msg, "Invalid specification");
    assert_contains(&msg, "empty bounds");
}

#[test]
fn test_ny_error_numerical_instability_display() {
    let err = NyError::NumericalInstability("NaN detected".to_string());
    let msg = format!("{err}");
    assert_contains(&msg, "Numerical instability");
    assert_contains(&msg, "NaN");
}

#[test]
fn test_ny_error_unsupported_configuration_display() {
    let err = NyError::UnsupportedConfiguration("dynamic shapes".to_string());
    let msg = format!("{err}");
    assert!(
        msg.contains("Unsupported configuration"),
        "Expected 'Unsupported configuration' in display message, got: {msg}"
    );
    assert_contains(&msg, "dynamic shapes");
}

#[test]
fn test_ny_error_layer_error_display() {
    let inner = Box::new(NyError::UnsupportedOp("op1".to_string()));
    let err = NyError::LayerError {
        layer_index: 5,
        layer_type: "MatMul".to_string(),
        source: inner,
    };
    let msg = format!("{err}");
    assert_contains(&msg, "Layer 5");
    assert_contains(&msg, "MatMul");
    assert_contains(&msg, "failed");
}

#[test]
fn test_ny_error_source_returns_inner_for_layer_error() {
    let inner = Box::new(NyError::UnsupportedOp("op1".to_string()));
    let err = NyError::LayerError {
        layer_index: 0,
        layer_type: "Test".to_string(),
        source: inner,
    };
    assert!(
        std::error::Error::source(&err).is_some(),
        "LayerError should expose its wrapped source error"
    );
}

#[test]
fn test_ny_error_source_returns_none_for_non_layer_error() {
    let err = NyError::UnsupportedOp("test".to_string());
    assert!(
        std::error::Error::source(&err).is_none(),
        "UnsupportedOp should not expose a nested source"
    );

    let err2 = NyError::ModelLoad("test".to_string());
    assert!(
        std::error::Error::source(&err2).is_none(),
        "ModelLoad should not expose a nested source"
    );
}

#[test]
fn test_ny_error_shape_mismatch_constructor() {
    // Different shapes should work fine
    let err = NyError::shape_mismatch(vec![1, 2], vec![3, 4]);
    match err {
        NyError::ShapeMismatch { expected, got } => {
            assert_eq!(expected, vec![1, 2]);
            assert_eq!(got, vec![3, 4]);
        }
        _ => panic!("Expected ShapeMismatch variant"),
    }
}

#[test]
fn test_ny_error_shape_mismatch_identical_shapes_no_panic() {
    let err = NyError::shape_mismatch(vec![1, 2, 3], vec![1, 2, 3]);
    match err {
        NyError::ShapeMismatch { expected, got } => {
            assert_eq!(expected, vec![1, 2, 3]);
            assert_eq!(got, vec![1, 2, 3]);
        }
        _ => panic!("Expected ShapeMismatch variant"),
    }
}

#[test]
fn test_ny_error_debug_impl() {
    let err = NyError::UnsupportedLayer("Test".to_string());
    let debug_str = format!("{err:?}");
    assert_contains(&debug_str, "UnsupportedLayer");
    assert_contains(&debug_str, "Test");
}

// Tests for VerificationSpec
#[test]
fn test_verification_spec_basic() {
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        vec![Bound::new(0.0, 1.0)],
        Some(5000),
        None,
    )
    .expect("valid test spec");

    assert_eq!(spec.input_bounds().len(), 2);
    assert_eq!(spec.output_bounds().len(), 1);
    assert_eq!(spec.timeout_ms(), Some(5000));
    assert!(
        spec.input_shape().is_none(),
        "spec without explicit shape should report no input shape"
    );
}

#[test]
fn test_verification_spec_with_input_shape() {
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(0.0, 1.0); 6],
        vec![Bound::new(0.0, 1.0)],
        None,
        Some(vec![2, 3]),
    )
    .expect("valid test spec");

    assert_eq!(spec.input_shape(), Some(&[2, 3][..]));
    assert!(
        spec.timeout_ms().is_none(),
        "spec without timeout should report no timeout"
    );
}

#[test]
fn test_verification_spec_serialization() {
    let spec = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0)],
        vec![Bound::new(0.0, 2.0)],
        Some(1000),
        Some(vec![1]),
    )
    .expect("valid test spec");

    // Serialize to JSON
    let json = serde_json::to_string(&spec).unwrap();
    assert_contains(&json, "input_bounds");
    assert_contains(&json, "output_bounds");
    assert_contains(&json, "timeout_ms");

    // Deserialize back
    let deserialized: VerificationSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.input_bounds().len(), 1);
    assert_eq!(deserialized.output_bounds().len(), 1);
    assert_eq!(deserialized.timeout_ms(), Some(1000));
}

// Tests for LayerType
#[test]
fn test_layer_type_debug() {
    assert_eq!(format!("{:?}", LayerType::Linear), "Linear");
    assert_eq!(format!("{:?}", LayerType::ReLU), "ReLU");
    assert_eq!(format!("{:?}", LayerType::Softmax), "Softmax");
    assert_eq!(
        format!("{:?}", LayerType::MultiHeadAttention),
        "MultiHeadAttention"
    );
}

#[test]
fn test_layer_type_display() {
    assert_eq!(format!("{}", LayerType::Linear), "Linear");
    assert_eq!(format!("{}", LayerType::Conv2d), "Conv2d");
    assert_eq!(format!("{}", LayerType::LeakyRelu), "LeakyRelu");
    assert_eq!(format!("{}", LayerType::HardSwish), "HardSwish");
    assert_eq!(format!("{}", LayerType::Erf), "Erf");
    assert_eq!(
        format!("{}", LayerType::MultiHeadAttention),
        "MultiHeadAttention"
    );
}

#[test]
fn test_layer_type_equality() {
    assert_eq!(LayerType::Linear, LayerType::Linear);
    assert_ne!(LayerType::Linear, LayerType::ReLU);
    assert_eq!(LayerType::GELU, LayerType::GELU);
    assert_ne!(LayerType::Conv1d, LayerType::Conv2d);
}

#[test]
fn test_layer_type_clone() {
    let layer = LayerType::LayerNorm;
    let cloned = layer.clone();
    assert_eq!(layer, cloned);
}

#[test]
fn test_layer_type_serialization() {
    let layer = LayerType::Softmax;
    let json = serde_json::to_string(&layer).unwrap();
    assert_eq!(json, "\"Softmax\"");

    let deserialized: LayerType = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized, LayerType::Softmax);
}

#[test]
fn test_layer_type_all_activations_distinct() {
    // Ensure all activation types are distinct
    let activations = vec![
        LayerType::ReLU,
        LayerType::LeakyRelu,
        LayerType::GELU,
        LayerType::SiLU,
        LayerType::Sigmoid,
        LayerType::Tanh,
        LayerType::Erf,
        LayerType::Softplus,
        LayerType::Tan,
        LayerType::Arctan,
        LayerType::Elu,
        LayerType::Selu,
        LayerType::PRelu,
        LayerType::HardSigmoid,
        LayerType::HardSwish,
        LayerType::Celu,
        LayerType::Mish,
        LayerType::ThresholdedRelu,
        LayerType::Softsign,
        LayerType::Snake,
    ];

    // Check all pairs are distinct
    for (i, a) in activations.iter().enumerate() {
        for (j, b) in activations.iter().enumerate() {
            if i != j {
                assert_ne!(a, b, "Activation types {a:?} and {b:?} should be distinct");
            }
        }
    }
}

// Tests for ViolationType
#[test]
fn test_violation_type_equality() {
    assert_eq!(ViolationType::BelowLower, ViolationType::BelowLower);
    assert_eq!(ViolationType::AboveUpper, ViolationType::AboveUpper);
    assert_ne!(ViolationType::BelowLower, ViolationType::AboveUpper);
}

#[test]
fn test_violation_type_serialization() {
    let below = ViolationType::BelowLower;
    let json = serde_json::to_string(&below).unwrap();
    let deserialized: ViolationType = serde_json::from_str(&json).unwrap();
    assert_eq!(below, deserialized);
}

// Tests for shape_mismatch_err! macro
#[test]
fn test_shape_mismatch_err_macro() {
    let err = shape_mismatch_err!(vec![1, 2], vec![3, 4]);
    match err {
        NyError::ShapeMismatch { expected, got } => {
            assert_eq!(expected, vec![1, 2]);
            assert_eq!(got, vec![3, 4]);
        }
        _ => panic!("Expected ShapeMismatch"),
    }
}

// Tests for Bound serde
#[test]
fn test_bound_serialization() {
    let bound = Bound::new(-1.5, 2.5);
    let json = serde_json::to_string(&bound).unwrap();
    assert_contains(&json, "-1.5");
    assert_contains(&json, "2.5");

    let deserialized: Bound = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.lower, -1.5);
    assert_eq!(deserialized.upper, 2.5);
}

// Tests for VerificationResult serialization
#[test]
fn test_verification_result_verified_serialization() {
    let result = VerificationResult::Verified {
        provenance: SoundnessProvenance::default(),
        output_bounds: vec![Bound::new(0.0, 1.0)],
        proof: None,
        actual_method: Some(MethodUsed::Crown),
    };
    let json = serde_json::to_string(&result).unwrap();
    let deserialized: VerificationResult = serde_json::from_str(&json).unwrap();
    assert!(
        deserialized.is_verified(),
        "verified results should remain verified after serde round-trip"
    );
    assert_eq!(deserialized.actual_method(), Some("Crown"));
    assert_eq!(deserialized.actual_method_tag(), Some(&MethodUsed::Crown));
}

#[test]
fn test_verification_result_violated_serialization() {
    let result = VerificationResult::Violated {
        provenance: SoundnessProvenance::default(),
        counterexample: vec![0.5],
        output: vec![1.5],
        details: None,
        actual_method: None,
    };
    let json = serde_json::to_string(&result).unwrap();
    let deserialized: VerificationResult = serde_json::from_str(&json).unwrap();
    assert!(
        !deserialized.is_verified(),
        "violated results should stay non-verified after serde round-trip"
    );
}

#[test]
fn test_verification_result_unknown_serialization() {
    let result = VerificationResult::Unknown {
        provenance: SoundnessProvenance::default(),
        bounds: vec![Bound::new(-1.0, 2.0)],
        reason: UnknownReason::BoundsTooLoose { gap: Some(0.5) },
        actual_method: Some(MethodUsed::Ibp),
    };
    let json = serde_json::to_string(&result).unwrap();
    assert_contains(&json, "Unknown");
    assert_contains(&json, "bounds_too_loose");
    assert_contains(&json, "Ibp");
}

#[test]
fn test_verification_result_timeout_serialization() {
    let result = VerificationResult::Timeout {
        provenance: SoundnessProvenance::default(),
        partial_bounds: Some(vec![Bound::new(0.0, 1.0)]),
        actual_method: None,
    };
    let json = serde_json::to_string(&result).unwrap();
    assert_contains(&json, "Timeout");
}

// MethodUsed serde round-trip tests
#[test]
fn test_method_used_known_variant_serde_roundtrip() {
    let method = MethodUsed::Crown;
    let json = serde_json::to_value(&method).unwrap();
    assert_eq!(json, serde_json::json!("Crown"));
    let deserialized: MethodUsed = serde_json::from_value(json).unwrap();
    assert_eq!(deserialized, MethodUsed::Crown);
}

#[test]
fn test_method_used_unknown_string_serde_roundtrip() {
    let method = MethodUsed::Other("CustomVerifier".to_string());
    let json = serde_json::to_value(&method).unwrap();
    assert_eq!(json, serde_json::json!("CustomVerifier"));
    let deserialized: MethodUsed = serde_json::from_value(json).unwrap();
    assert_eq!(
        deserialized,
        MethodUsed::Other("CustomVerifier".to_string())
    );
}

#[test]
fn test_method_used_from_str_all_known_variants() {
    assert_eq!(MethodUsed::from("Ibp"), MethodUsed::Ibp);
    assert_eq!(MethodUsed::from("Crown"), MethodUsed::Crown);
    assert_eq!(MethodUsed::from("AlphaCrown"), MethodUsed::AlphaCrown);
    assert_eq!(MethodUsed::from("SdpCrown"), MethodUsed::SdpCrown);
    assert_eq!(MethodUsed::from("BetaCrown"), MethodUsed::BetaCrown);
    assert_eq!(MethodUsed::from("SmtRefiner"), MethodUsed::SmtRefiner);
    assert_eq!(
        MethodUsed::from("LazySmtRefiner"),
        MethodUsed::LazySmtRefiner
    );
    assert_eq!(MethodUsed::from("Mip"), MethodUsed::Mip);
    assert_eq!(MethodUsed::from("MipHiGHS"), MethodUsed::MipHiGHS);
    assert_eq!(MethodUsed::from("MipVnnlib"), MethodUsed::MipVnnlib);
    assert_eq!(MethodUsed::from("Ibp_f64"), MethodUsed::IbpF64);
    assert_eq!(MethodUsed::from("Crown_f64"), MethodUsed::CrownF64);
    assert_eq!(
        MethodUsed::from("NewMethod"),
        MethodUsed::Other("NewMethod".to_string())
    );
}

#[test]
fn test_method_used_deref_enables_as_deref() {
    let opt: Option<MethodUsed> = Some(MethodUsed::BetaCrown);
    assert_eq!(opt.as_deref(), Some("BetaCrown"));
    let none: Option<MethodUsed> = None;
    assert_eq!(none.as_deref(), None);
}

// =============================================================================
// UnknownReason Tests
// =============================================================================

#[test]
fn test_unknown_reason_display() {
    use crate::UnknownReason;

    // BoundsTooLoose with gap
    let reason = UnknownReason::BoundsTooLoose { gap: Some(0.1234) };
    assert_eq!(format!("{reason}"), "Bounds too loose (gap: 0.1234)");

    // BoundsTooLoose without gap
    let reason = UnknownReason::BoundsTooLoose { gap: None };
    assert_eq!(format!("{reason}"), "Bounds too loose");

    // SmtUnknown with reason
    let reason = UnknownReason::SmtUnknown {
        solver_reason: Some("quantifier instantiation".to_string()),
    };
    assert_eq!(
        format!("{reason}"),
        "SMT solver returned unknown: quantifier instantiation"
    );

    // SmtUnknown without reason
    let reason = UnknownReason::SmtUnknown {
        solver_reason: None,
    };
    assert_eq!(format!("{reason}"), "SMT solver returned unknown");

    // ResourceLimit
    let reason = UnknownReason::ResourceLimit {
        resource: "memory".to_string(),
        limit: 1000,
        used: 1500,
    };
    assert_eq!(
        format!("{reason}"),
        "Resource limit exceeded: memory (1500 used, 1000 limit)"
    );

    // UnsupportedOp
    let reason = UnknownReason::UnsupportedOp {
        op_name: "Einsum".to_string(),
    };
    assert_eq!(format!("{reason}"), "Unsupported operation: Einsum");

    // SatTrustPolicy
    let reason = UnknownReason::SatTrustPolicy {
        policy: "DowngradeSat".to_string(),
    };
    assert_eq!(
        format!("{reason}"),
        "SAT downgraded by trust policy: DowngradeSat"
    );

    // PotentialViolation
    let reason = UnknownReason::PotentialViolation;
    assert_eq!(format!("{reason}"), "Potential violation region found");

    // Other
    let reason = UnknownReason::Other {
        message: "custom reason".to_string(),
    };
    assert_eq!(format!("{reason}"), "custom reason");
}

#[test]
fn test_unknown_reason_from_string() {
    use crate::UnknownReason;

    // Pattern: "too loose" -> BoundsTooLoose
    let reason = UnknownReason::from("bounds too loose".to_string());
    assert!(
        matches!(&reason, UnknownReason::BoundsTooLoose { gap: None }),
        "expected BoundsTooLoose without a gap, got {reason:?}"
    );

    // Pattern: "SMT" -> SmtUnknown
    let reason = UnknownReason::from("SMT solver failed".to_string());
    assert!(
        matches!(
            &reason,
            UnknownReason::SmtUnknown {
                solver_reason: Some(_)
            }
        ),
        "expected SmtUnknown for SMT failure, got {reason:?}"
    );

    // Pattern: "solver" -> SmtUnknown
    let reason = UnknownReason::from("solver returned unknown".to_string());
    assert!(
        matches!(
            &reason,
            UnknownReason::SmtUnknown {
                solver_reason: Some(_)
            }
        ),
        "expected SmtUnknown for solver failure, got {reason:?}"
    );

    // Pattern: "trust policy" -> SatTrustPolicy
    let reason = UnknownReason::from("SAT trust policy disallowed".to_string());
    assert!(
        matches!(&reason, UnknownReason::SatTrustPolicy { .. }),
        "expected SatTrustPolicy for trust-policy strings, got {reason:?}"
    );

    // Pattern: "downgraded" -> SatTrustPolicy
    let reason = UnknownReason::from("result downgraded".to_string());
    assert!(
        matches!(&reason, UnknownReason::SatTrustPolicy { .. }),
        "expected SatTrustPolicy for downgraded strings, got {reason:?}"
    );

    // Pattern: "potential violation" -> PotentialViolation
    let reason = UnknownReason::from("potential violation found".to_string());
    assert!(
        matches!(&reason, UnknownReason::PotentialViolation),
        "expected PotentialViolation for potential-violation strings, got {reason:?}"
    );

    // Unrecognized -> Other
    let reason = UnknownReason::from("some random reason".to_string());
    assert!(
        matches!(&reason, UnknownReason::Other { .. }),
        "expected Other for unmatched strings, got {reason:?}"
    );
}

#[test]
fn test_unknown_reason_serialization() {
    use crate::UnknownReason;

    // BoundsTooLoose serializes with type tag
    let reason = UnknownReason::BoundsTooLoose { gap: Some(0.5) };
    let json = serde_json::to_string(&reason).unwrap();
    assert_contains(&json, "\"type\":\"bounds_too_loose\"");
    assert_contains(&json, "\"gap\":0.5");

    // SmtUnknown serializes with type tag
    let reason = UnknownReason::SmtUnknown {
        solver_reason: Some("timeout".to_string()),
    };
    let json = serde_json::to_string(&reason).unwrap();
    assert_contains(&json, "\"type\":\"smt_unknown\"");
    assert_contains(&json, "timeout");

    // ResourceLimit serializes all fields
    let reason = UnknownReason::ResourceLimit {
        resource: "iterations".to_string(),
        limit: 100,
        used: 150,
    };
    let json = serde_json::to_string(&reason).unwrap();
    assert_contains(&json, "\"type\":\"resource_limit\"");
    assert_contains(&json, "\"resource\":\"iterations\"");
    assert_contains(&json, "\"limit\":100");
    assert_contains(&json, "\"used\":150");

    // PotentialViolation has no extra fields
    let reason = UnknownReason::PotentialViolation;
    let json = serde_json::to_string(&reason).unwrap();
    assert_contains(&json, "\"type\":\"potential_violation\"");
}

// ============================================================================
// Tests for VerificationSpec builder API
// ============================================================================

#[test]
fn test_verification_spec_new_basic() {
    let spec = VerificationSpec::new(
        vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
        vec![Bound::new(0.0, 1.0)],
    )
    .expect("should create valid spec");

    assert_eq!(spec.input_bounds().len(), 2);
    assert_eq!(spec.output_bounds().len(), 1);
    assert!(
        spec.timeout_ms().is_none(),
        "new specs should default to no timeout"
    );
    assert!(
        spec.input_shape().is_none(),
        "new specs should default to no input shape"
    );
}

#[test]
fn test_verification_spec_new_rejects_empty_input() {
    let result = VerificationSpec::new(vec![], vec![Bound::new(0.0, 1.0)]);
    let err = result.expect_err("empty input bounds must be rejected");
    assert!(
        matches!(&err, NyError::InvalidSpec(_)),
        "expected InvalidSpec for empty input bounds, got {err:?}"
    );
}

#[test]
fn test_verification_spec_new_rejects_empty_output() {
    let result = VerificationSpec::new(vec![Bound::new(-1.0, 1.0)], vec![]);
    let err = result.expect_err("empty output bounds must be rejected");
    assert!(
        matches!(&err, NyError::InvalidSpec(_)),
        "expected InvalidSpec for empty output bounds, got {err:?}"
    );
}

#[test]
fn test_verification_spec_with_timeout() {
    let spec = VerificationSpec::new(vec![Bound::new(-1.0, 1.0)], vec![Bound::new(0.0, 1.0)])
        .unwrap()
        .with_timeout_ms(5000);

    assert_eq!(spec.timeout_ms(), Some(5000));
}

#[test]
fn test_verification_spec_with_input_shape_valid() {
    let spec = VerificationSpec::new(
        vec![Bound::new(0.0, 1.0); 6], // 6 elements
        vec![Bound::new(0.0, 1.0)],
    )
    .unwrap()
    .with_input_shape(vec![2, 3]) // 2*3 = 6
    .expect("shape product matches");

    assert_eq!(spec.input_shape(), Some(&[2, 3][..]));
}

#[test]
fn test_verification_spec_with_input_shape_mismatch() {
    let result = VerificationSpec::new(
        vec![Bound::new(0.0, 1.0); 6], // 6 elements
        vec![Bound::new(0.0, 1.0)],
    )
    .unwrap()
    .with_input_shape(vec![2, 2]); // 2*2 = 4 != 6

    let err = result.expect_err("mismatched input shapes must be rejected");
    assert!(
        matches!(&err, NyError::InvalidSpec(_)),
        "expected InvalidSpec for mismatched input shapes, got {err:?}"
    );
}

#[test]
fn test_verification_spec_builder_chain() {
    let spec = VerificationSpec::new(
        vec![Bound::new(-1.0, 1.0); 4],
        vec![Bound::new(0.0, 1.0), Bound::new(-0.5, 0.5)],
    )
    .unwrap()
    .with_timeout_ms(10_000)
    .with_input_shape(vec![2, 2])
    .expect("full chain");

    assert_eq!(spec.input_bounds().len(), 4);
    assert_eq!(spec.output_bounds().len(), 2);
    assert_eq!(spec.timeout_ms(), Some(10_000));
    assert_eq!(spec.input_shape(), Some(&[2, 2][..]));
}

// ============================================================================
// Regression tests for VerificationSpec shape product overflow (#2371)
// ============================================================================

#[test]
fn test_verification_spec_from_parts_shape_overflow() {
    let result = VerificationSpec::from_parts(
        vec![Bound::new(0.0, 1.0)],
        vec![Bound::new(0.0, 1.0)],
        None,
        Some(vec![usize::MAX, 2]),
    );
    assert!(
        result.is_err(),
        "overflowing input shapes must be rejected by VerificationSpec::from_parts"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("overflows"),
        "expected overflow error, got: {err_msg}"
    );
}

#[test]
fn test_verification_spec_with_input_shape_overflow() {
    let spec = VerificationSpec::new(vec![Bound::new(0.0, 1.0)], vec![Bound::new(0.0, 1.0)])
        .expect("valid spec for overflow test");
    let result = spec.with_input_shape(vec![usize::MAX, 2]);
    assert!(
        result.is_err(),
        "overflowing input shapes must be rejected by with_input_shape"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("overflows"),
        "expected overflow error, got: {err_msg}"
    );
}

/// NaN output must be detected as a violation, not silently passed through.
/// IEEE 754: NaN < x and NaN > x are both false, so without a NaN guard
/// a NaN counterexample would return None (no violation). (#3291 F1)
#[test]
fn test_violated_constraint_detect_nan_output() {
    let output = vec![0.5, f32::NAN, 0.8];
    let bounds = vec![
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
    ];

    let vc = ViolatedConstraint::detect(&output, &bounds);
    assert!(vc.is_some(), "NaN output must be detected as a violation");

    let vc = vc.unwrap();
    assert_eq!(vc.output_idx(), 1);
    assert_nan(vc.actual_value(), "NaN output violation actual value");
    assert_eq!(vc.violation_type(), ViolationType::BelowLower);
    assert_eq!(vc.violation_amount(), f32::INFINITY);
}

/// NaN in the first position should be caught immediately.
#[test]
fn test_violated_constraint_detect_nan_first_element() {
    let output = vec![f32::NAN];
    let bounds = vec![Bound::new(-1.0, 1.0)];

    let vc = ViolatedConstraint::detect(&output, &bounds).unwrap();
    assert_eq!(vc.output_idx(), 0);
    assert_nan(
        vc.actual_value(),
        "first-element NaN violation actual value",
    );
}

/// All-NaN output should detect violation at index 0.
#[test]
fn test_violated_constraint_detect_all_nan() {
    let output = vec![f32::NAN, f32::NAN, f32::NAN];
    let bounds = vec![
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
        Bound::new(0.0, 1.0),
    ];

    let vc = ViolatedConstraint::detect(&output, &bounds).unwrap();
    assert_eq!(vc.output_idx(), 0, "Should detect first NaN");
}
