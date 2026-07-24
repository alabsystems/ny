// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::Bound;

#[test]
#[should_panic(expected = "Bound::concrete: value must be finite")]
fn concrete_rejects_nan() {
    let _ = Bound::concrete(f32::NAN);
}

#[test]
#[should_panic(expected = "Bound::concrete: value must be finite")]
fn concrete_rejects_infinite() {
    let _ = Bound::concrete(f32::INFINITY);
}

#[test]
#[should_panic(expected = "Bound::union: self bound contains NaN endpoint")]
fn union_rejects_nan_endpoints() {
    let invalid = Bound {
        lower: f32::NAN,
        upper: 1.0,
    };
    let valid = Bound::new(0.0, 2.0);
    let _ = invalid.union(&valid);
}

// --- Intersect NaN guards (#2640) ---
// Bound::intersect must return None when any endpoint is NaN,
// preventing silent NaN absorption via IEEE 754 max/min semantics.

#[test]
fn intersect_nan_lower_self_returns_none() {
    let nan_bound = Bound {
        lower: f32::NAN,
        upper: 5.0,
    };
    let valid = Bound::new(1.0, 3.0);
    assert!(
        nan_bound.intersect(&valid).is_none(),
        "intersect must reject NaN lower in self"
    );
}

#[test]
fn intersect_nan_upper_self_returns_none() {
    let nan_bound = Bound {
        lower: 1.0,
        upper: f32::NAN,
    };
    let valid = Bound::new(0.0, 2.0);
    assert!(
        nan_bound.intersect(&valid).is_none(),
        "intersect must reject NaN upper in self"
    );
}

#[test]
fn intersect_nan_lower_other_returns_none() {
    let valid = Bound::new(1.0, 3.0);
    let nan_bound = Bound {
        lower: f32::NAN,
        upper: 5.0,
    };
    assert!(
        valid.intersect(&nan_bound).is_none(),
        "intersect must reject NaN lower in other"
    );
}

#[test]
fn intersect_nan_upper_other_returns_none() {
    let valid = Bound::new(0.0, 2.0);
    let nan_bound = Bound {
        lower: 1.0,
        upper: f32::NAN,
    };
    assert!(
        valid.intersect(&nan_bound).is_none(),
        "intersect must reject NaN upper in other"
    );
}

#[test]
fn intersect_both_nan_returns_none() {
    let nan_bound = Bound {
        lower: f32::NAN,
        upper: f32::NAN,
    };
    let valid = Bound::new(1.0, 3.0);
    assert!(
        nan_bound.intersect(&valid).is_none(),
        "intersect must reject all-NaN bound as self"
    );
    assert!(
        valid.intersect(&nan_bound).is_none(),
        "intersect must reject all-NaN bound as other"
    );
}

#[test]
fn intersect_valid_bounds_still_works() {
    // Regression: valid intersections must not be broken by the NaN guard.
    let a = Bound::new(1.0, 5.0);
    let b = Bound::new(3.0, 7.0);
    let result = a.intersect(&b).expect("valid intersection should succeed");
    assert_eq!(result.lower(), 3.0);
    assert_eq!(result.upper(), 5.0);
}

#[test]
fn intersect_disjoint_still_returns_none() {
    let a = Bound::new(1.0, 2.0);
    let b = Bound::new(3.0, 4.0);
    assert!(
        a.intersect(&b).is_none(),
        "disjoint bounds [1,2] and [3,4] must not intersect"
    );
}

// --- Fallible constructors (#2630) ---

#[test]
fn try_new_valid_returns_ok() {
    let b = Bound::try_new(-1.0, 2.0).expect("invariant: valid inputs");
    assert_eq!(b.lower(), -1.0);
    assert_eq!(b.upper(), 2.0);
}

#[test]
fn try_new_nan_lower_returns_err() {
    let result = Bound::try_new(f32::NAN, 1.0);
    assert!(result.is_err(), "try_new must reject NaN lower");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not finite"), "got: {msg}");
}

#[test]
fn try_new_nan_upper_returns_err() {
    let result = Bound::try_new(0.0, f32::NAN);
    assert!(result.is_err(), "try_new must reject NaN upper");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not finite"), "got: {msg}");
}

#[test]
fn try_new_inf_lower_returns_err() {
    let result = Bound::try_new(f32::NEG_INFINITY, 1.0);
    assert!(result.is_err(), "try_new must reject -Inf lower");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not finite"), "got: {msg}");
}

#[test]
fn try_new_inf_upper_returns_err() {
    let result = Bound::try_new(0.0, f32::INFINITY);
    assert!(result.is_err(), "try_new must reject +Inf upper");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not finite"), "got: {msg}");
}

#[test]
fn try_new_inverted_returns_err() {
    let result = Bound::try_new(5.0, 1.0);
    assert!(result.is_err(), "try_new must reject inverted [5,1]");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("inverted"), "got: {msg}");
}

#[test]
fn try_new_allow_infinite_valid_returns_ok() {
    let b = Bound::try_new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)
        .expect("invariant: valid inputs");
    assert_eq!(b.lower(), f32::NEG_INFINITY);
    assert_eq!(b.upper(), f32::INFINITY);
}

#[test]
fn try_new_allow_infinite_nan_lower_returns_err() {
    let result = Bound::try_new_allow_infinite(f32::NAN, 1.0);
    assert!(
        result.is_err(),
        "try_new_allow_infinite must reject NaN lower"
    );
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("NaN"), "got: {msg}");
}

#[test]
fn try_new_allow_infinite_nan_upper_returns_err() {
    let result = Bound::try_new_allow_infinite(0.0, f32::NAN);
    assert!(
        result.is_err(),
        "try_new_allow_infinite must reject NaN upper"
    );
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("NaN"), "got: {msg}");
}

#[test]
fn try_new_allow_infinite_inverted_returns_err() {
    let result = Bound::try_new_allow_infinite(5.0, 1.0);
    assert!(
        result.is_err(),
        "try_new_allow_infinite must reject inverted [5,1]"
    );
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("inverted"), "got: {msg}");
}

#[test]
fn try_concrete_valid_returns_ok() {
    let b = Bound::try_concrete(2.5).expect("invariant: valid inputs");
    assert_eq!(b.lower(), 2.5);
    assert_eq!(b.upper(), 2.5);
    assert_eq!(b.width(), 0.0);
}

#[test]
fn try_concrete_nan_returns_err() {
    let result = Bound::try_concrete(f32::NAN);
    assert!(result.is_err(), "try_concrete must reject NaN");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not finite"), "got: {msg}");
}

#[test]
fn try_concrete_inf_returns_err() {
    let result = Bound::try_concrete(f32::INFINITY);
    assert!(result.is_err(), "try_concrete must reject +Inf");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not finite"), "got: {msg}");
}

#[test]
fn try_concrete_neg_inf_returns_err() {
    let result = Bound::try_concrete(f32::NEG_INFINITY);
    assert!(result.is_err(), "try_concrete must reject -Inf");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not finite"), "got: {msg}");
}

// --- Serde deserialization validation tests (#2367) ---

#[test]
fn deserialize_bound_nan_lower_rejected() {
    let json = r#"{"lower": null, "upper": 1.0}"#;
    // NaN is not representable in JSON, but null could be deserialized as NaN
    // by some formats. Use a direct NaN-producing JSON with serde_json's
    // "arbitrary_precision" or test via the f32 NaN encoding path.
    // Since JSON doesn't have NaN, we test inverted bounds (which JSON can
    // express) and also test the round-trip path where Serialize could
    // produce NaN strings in non-JSON formats.
    let result: Result<Bound, _> = serde_json::from_str(json);
    assert!(result.is_err(), "null should not deserialize as Bound");
}

#[test]
fn deserialize_bound_inverted_rejected() {
    // JSON: lower > upper — must be rejected.
    let json = r#"{"lower": 5.0, "upper": 1.0}"#;
    let result: Result<Bound, _> = serde_json::from_str(json);
    assert!(
        result.is_err(),
        "inverted Bound [5.0, 1.0] should be rejected during deserialization"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("inverted interval"),
        "error should mention 'inverted interval', got: {err_msg}"
    );
}

#[test]
fn deserialize_bound_valid_accepted() {
    let json = r#"{"lower": -1.0, "upper": 2.5}"#;
    let bound: Bound = serde_json::from_str(json).expect("invariant: valid JSON");
    assert_eq!(bound.lower(), -1.0);
    assert_eq!(bound.upper(), 2.5);
}

#[test]
fn deserialize_bound_infinite_accepted() {
    // Infinite bounds are valid (used for output specs like [-inf, +inf]).
    // serde_json doesn't support Infinity literals in standard mode.
    // Verify the invariants the custom Deserialize checks:
    // (1) infinity is not NaN, so NaN rejection doesn't fire
    // (2) -inf <= +inf, so inverted-interval rejection doesn't fire
    // (3) Bound can be constructed with infinite endpoints via new_allow_infinite
    assert!(!f32::INFINITY.is_nan(), "infinity must not be NaN");
    assert!(!f32::NEG_INFINITY.is_nan(), "neg infinity must not be NaN");
    // Verify new_allow_infinite accepts infinities (same code path as Deserialize).
    let bound = Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY);
    assert_eq!(bound.lower(), f32::NEG_INFINITY);
    assert_eq!(bound.upper(), f32::INFINITY);

    let bound = Bound::new_allow_infinite(0.0, f32::INFINITY);
    assert_eq!(bound.upper(), f32::INFINITY);

    let bound = Bound::new_allow_infinite(f32::NEG_INFINITY, 0.0);
    assert_eq!(bound.lower(), f32::NEG_INFINITY);
}

#[test]
fn deserialize_bound_equal_endpoints_accepted() {
    // Point bounds: lower == upper.
    let json = r#"{"lower": 3.0, "upper": 3.0}"#;
    let bound: Bound = serde_json::from_str(json).expect("invariant: valid JSON");
    assert_eq!(bound.lower(), 3.0);
    assert_eq!(bound.upper(), 3.0);
}

#[test]
fn deserialize_bound_roundtrip_preserves_values() {
    let original = Bound::new(1.5, 7.25);
    let json = serde_json::to_string(&original).expect("invariant: serializes");
    let restored: Bound = serde_json::from_str(&json).expect("invariant: deserializes");
    assert_eq!(original, restored);
}

#[test]
fn deserialize_verification_spec_empty_input_bounds_rejected() {
    use crate::VerificationSpec;
    let json = r#"{"input_bounds": [], "output_bounds": [{"lower": 0.0, "upper": 1.0}]}"#;
    let result: Result<VerificationSpec, _> = serde_json::from_str(json);
    assert!(result.is_err(), "empty input_bounds should be rejected");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("input_bounds cannot be empty"),
        "error should mention empty input_bounds, got: {err_msg}"
    );
}

#[test]
fn deserialize_verification_spec_empty_output_bounds_rejected() {
    use crate::VerificationSpec;
    let json = r#"{"input_bounds": [{"lower": 0.0, "upper": 1.0}], "output_bounds": []}"#;
    let result: Result<VerificationSpec, _> = serde_json::from_str(json);
    assert!(result.is_err(), "empty output_bounds should be rejected");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("output_bounds cannot be empty"),
        "error should mention empty output_bounds, got: {err_msg}"
    );
}

#[test]
fn deserialize_verification_spec_inverted_bound_rejected() {
    use crate::VerificationSpec;
    // Inverted Bound inside VerificationSpec input_bounds.
    let json = r#"{"input_bounds": [{"lower": 5.0, "upper": 1.0}], "output_bounds": [{"lower": 0.0, "upper": 1.0}]}"#;
    let result: Result<VerificationSpec, _> = serde_json::from_str(json);
    assert!(
        result.is_err(),
        "VerificationSpec with inverted Bound should be rejected"
    );
}

#[test]
fn deserialize_verification_spec_valid_accepted() {
    use crate::VerificationSpec;
    let json = r#"{"input_bounds": [{"lower": -1.0, "upper": 1.0}], "output_bounds": [{"lower": 0.0, "upper": 1.0}]}"#;
    let spec: VerificationSpec =
        serde_json::from_str(json).expect("invariant: valid VerificationSpec JSON");
    assert_eq!(spec.input_bounds().len(), 1);
    assert_eq!(spec.output_bounds().len(), 1);
    assert_eq!(spec.input_bounds()[0].lower(), -1.0);
}

#[test]
fn deserialize_verification_spec_roundtrip() {
    use crate::VerificationSpec;
    let spec = VerificationSpec::new(
        vec![Bound::new(-1.0, 1.0), Bound::new(0.0, 2.0)],
        vec![Bound::new(0.0, 1.0)],
    )
    .expect("invariant: valid spec");
    let json = serde_json::to_string(&spec).expect("invariant: serializes");
    let restored: VerificationSpec = serde_json::from_str(&json).expect("invariant: deserializes");
    assert_eq!(restored.input_bounds().len(), 2);
    assert_eq!(restored.output_bounds().len(), 1);
}

#[test]
fn deserialize_verification_spec_shape_overflow_rejected() {
    use crate::VerificationSpec;
    // input_shape product [usize::MAX, 2] overflows — must be rejected (#2371).
    let json = r#"{
        "input_bounds": [{"lower": 0.0, "upper": 1.0}],
        "output_bounds": [{"lower": 0.0, "upper": 1.0}],
        "input_shape": [18446744073709551615, 2]
    }"#;
    let result: Result<VerificationSpec, _> = serde_json::from_str(json);
    assert!(
        result.is_err(),
        "overflowing input_shape should be rejected"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("overflow"),
        "error should mention overflow, got: {err_msg}"
    );
}

#[test]
fn deserialize_verification_spec_input_shape_mismatch_rejected() {
    use crate::VerificationSpec;
    // input_shape product is 2*3=6, but only 2 input_bounds — must be rejected.
    let json = r#"{"input_bounds": [{"lower": 0.0, "upper": 1.0}, {"lower": 0.0, "upper": 1.0}], "output_bounds": [{"lower": 0.0, "upper": 1.0}], "input_shape": [2, 3]}"#;
    let result: Result<VerificationSpec, _> = serde_json::from_str(json);
    assert!(
        result.is_err(),
        "input_shape product mismatch should be rejected"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("does not match"),
        "error should mention mismatch, got: {err_msg}"
    );
}

#[test]
fn deserialize_verification_spec_input_shape_valid_accepted() {
    use crate::VerificationSpec;
    // input_shape product is 2*3=6, matching 6 input_bounds.
    let json = r#"{
        "input_bounds": [
            {"lower": 0.0, "upper": 1.0}, {"lower": 0.0, "upper": 1.0},
            {"lower": 0.0, "upper": 1.0}, {"lower": 0.0, "upper": 1.0},
            {"lower": 0.0, "upper": 1.0}, {"lower": 0.0, "upper": 1.0}
        ],
        "output_bounds": [{"lower": 0.0, "upper": 1.0}],
        "input_shape": [2, 3]
    }"#;
    let spec: VerificationSpec =
        serde_json::from_str(json).expect("invariant: valid VerificationSpec with input_shape");
    assert_eq!(spec.input_bounds().len(), 6);
    assert_eq!(spec.input_shape(), Some(&[2, 3][..]));
}

#[test]
fn deserialize_verification_spec_with_input_shape_roundtrip() {
    use crate::VerificationSpec;
    let spec = VerificationSpec::new(vec![Bound::new(0.0, 1.0); 6], vec![Bound::new(0.0, 1.0)])
        .expect("invariant: valid spec")
        .with_input_shape(vec![2, 3])
        .expect("invariant: valid shape");
    let json = serde_json::to_string(&spec).expect("invariant: serializes");
    let restored: VerificationSpec = serde_json::from_str(&json).expect("invariant: deserializes");
    assert_eq!(restored.input_shape(), Some(&[2, 3][..]));
    assert_eq!(restored.input_bounds().len(), 6);
}
