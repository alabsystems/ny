// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for #2778: VerificationSpec contract parity.
//!
//! All public entry paths (`new`, `from_parts`, `Deserialize`) must enforce the
//! same bound contract:
//! - Input bounds: finite, non-NaN, ordered.
//! - Output bounds: may be infinite, non-NaN, ordered.

use crate::Bound;

#[test]
fn spec_new_accepts_infinite_output_bounds_2778() {
    use crate::VerificationSpec;
    let result = VerificationSpec::new(
        vec![Bound::new(-1.0, 1.0)],
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
    );
    assert!(
        result.is_ok(),
        "new() should accept infinite output bounds, got: {:?}",
        result.unwrap_err()
    );
    let spec = result.expect("invariant: infinite output bounds accepted");
    assert_eq!(spec.output_bounds()[0].lower(), f32::NEG_INFINITY);
    assert_eq!(spec.output_bounds()[0].upper(), f32::INFINITY);
}

#[test]
fn spec_new_accepts_half_infinite_output_bounds_2778() {
    use crate::VerificationSpec;
    // Lower-unbounded: output >= 3.0
    let result = VerificationSpec::new(
        vec![Bound::new(0.0, 1.0)],
        vec![Bound::new_allow_infinite(3.0, f32::INFINITY)],
    );
    assert!(result.is_ok(), "new() should accept [3.0, +inf] output");

    // Upper-unbounded: output <= -2.0
    let result = VerificationSpec::new(
        vec![Bound::new(0.0, 1.0)],
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, -2.0)],
    );
    assert!(result.is_ok(), "new() should accept [-inf, -2.0] output");
}

#[test]
fn spec_new_rejects_infinite_input_bounds_2778() {
    use crate::VerificationSpec;
    let result = VerificationSpec::new(
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        vec![Bound::new(0.0, 1.0)],
    );
    assert!(result.is_err(), "new() should reject infinite input bounds");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("non-finite"),
        "error should mention non-finite, got: {err_msg}"
    );
}

#[test]
fn spec_from_parts_rejects_infinite_input_bounds_2778() {
    use crate::VerificationSpec;
    let result = VerificationSpec::from_parts(
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, 1.0)],
        vec![Bound::new(0.0, 1.0)],
        None,
        None,
    );
    assert!(
        result.is_err(),
        "from_parts() should reject infinite input bounds"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("non-finite"),
        "error should mention non-finite, got: {err_msg}"
    );
}

#[test]
fn spec_from_parts_accepts_infinite_output_bounds_2778() {
    use crate::VerificationSpec;
    let result = VerificationSpec::from_parts(
        vec![Bound::new(-1.0, 1.0)],
        vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)],
        Some(5000),
        None,
    );
    assert!(
        result.is_ok(),
        "from_parts() should accept infinite output bounds, got: {:?}",
        result.unwrap_err()
    );
}

#[test]
fn spec_parity_finite_input_finite_output_all_accept_2778() {
    use crate::VerificationSpec;
    // All three paths should accept: finite input + finite output.
    let input = vec![Bound::new(-1.0, 1.0)];
    let output = vec![Bound::new(0.0, 2.0)];

    VerificationSpec::new(input.clone(), output.clone())
        .expect("new() should accept finite input and finite output");
    VerificationSpec::from_parts(input, output, None, None)
        .expect("from_parts() should accept finite input and finite output");

    let json = r#"{"input_bounds": [{"lower": -1.0, "upper": 1.0}], "output_bounds": [{"lower": 0.0, "upper": 2.0}]}"#;
    let deser: Result<VerificationSpec, _> = serde_json::from_str(json);
    assert!(
        deser.is_ok(),
        "serde should accept finite-input/finite-output"
    );
}

#[test]
fn spec_parity_infinite_input_all_reject_2778() {
    use crate::VerificationSpec;
    // All three paths should reject: infinite input bounds.
    let inf_input = vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY)];
    let output = vec![Bound::new(0.0, 1.0)];

    assert!(
        VerificationSpec::new(inf_input.clone(), output.clone()).is_err(),
        "new() should reject infinite input"
    );
    assert!(
        VerificationSpec::from_parts(inf_input, output, None, None).is_err(),
        "from_parts() should reject infinite input"
    );
    // Note: standard JSON cannot represent f32::INFINITY literals, so the
    // serde path is tested via constructor parity. The Bound deserialization
    // itself allows infinities (tested separately in deserialize_bound_infinite_accepted),
    // but the VerificationSpec Deserialize impl rejects non-finite input bounds
    // after Bound deserialization succeeds.
}
