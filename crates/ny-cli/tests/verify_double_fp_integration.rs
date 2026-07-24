// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `ny verify --double-fp` verdict semantics.
//!
//! The f64 path must apply the same VNN-LIB property gate and exit-code
//! contract as the standard path: f64 propagation completing is NOT a
//! "verified" verdict on its own. A property is "safe" only when the
//! asserted output region is unreachable given the computed bounds.
//!
//! Fixture: `crossing_relu.nnet` computes f(x) = |x|, so on X_0 in [-1, 1]
//! the true output range is [0, 1] (bound propagation adds float slack).
//! The `.vnnlib` fixtures use the standard VNN-LIB convention: the output
//! assert is the NEGATED safety property (the violation to rule out).

#[path = "common/vnncomp.rs"]
mod vnncomp_support;

use ny_test_utils::{require_model, test_models_dir};
use std::io::Write;
use tempfile::NamedTempFile;
use vnncomp_support::{parse_json_output, run_ny};

/// Exit code contract: 0 = verified, 2 = unknown.
const EXIT_VERIFIED: i32 = 0;
const EXIT_UNKNOWN: i32 = 2;

fn crossing_relu_model() -> String {
    let model = test_models_dir().join("crossing_relu.nnet");
    require_model(&model);
    model.to_str().expect("model path is UTF-8").to_string()
}

fn crossing_relu_unsafe_property() -> String {
    let property = test_models_dir().join("crossing_relu_unsafe.vnnlib");
    require_model(&property);
    property
        .to_str()
        .expect("property path is UTF-8")
        .to_string()
}

/// A property the f64 bounds cannot prove must NOT come back "verified".
///
/// The fixture asserts the violation Y_0 >= 0.5 (negated safety property),
/// which IS reachable within the computed output box (true range [0, 1]), so
/// no box argument can rule it out. The standard path reports unknown / exit 2
/// for this instance; --double-fp must agree instead of reporting the
/// completion of bound propagation as a verdict.
#[ntest::timeout(60000)]
#[test]
fn double_fp_unprovable_property_is_not_verified() {
    let model = crossing_relu_model();
    let property = crossing_relu_unsafe_property();

    let output = run_ny(&[
        "verify",
        &model,
        "--property",
        &property,
        "--method",
        "crown",
        "--double-fp",
        "--json",
    ]);

    let json = parse_json_output(&output, "verify --double-fp (unprovable property)");
    assert_eq!(
        json["status"], "unknown",
        "bounds cannot prove the violation Y_0 >= 0.5 unreachable; full JSON: {json}"
    );
    assert_eq!(json["property_status"], "unknown", "full JSON: {json}");
    assert_eq!(json["double_fp"], true, "full JSON: {json}");
    assert_eq!(json["method"], "Crown_f64", "full JSON: {json}");
    assert_eq!(
        output.status.code(),
        Some(EXIT_UNKNOWN),
        "an unproven property must exit UNKNOWN, not VERIFIED"
    );
}

/// A property whose asserted region is box-unreachable must still verify.
///
/// Y_0 >= 3 cannot hold when the f64 output upper bound is ~1, so the
/// property gate proves "safe" and the run exits 0. Guards against
/// over-correcting the f64 path into never verifying anything.
#[ntest::timeout(60000)]
#[test]
fn double_fp_box_unreachable_property_is_verified() {
    let model = crossing_relu_model();

    let vnnlib_content = "\
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 -1.0))
(assert (<= X_0 1.0))
(assert (>= Y_0 3.0))
";
    let mut vnnlib_file =
        NamedTempFile::new().expect("failed to create temp file for VNN-LIB property");
    vnnlib_file
        .write_all(vnnlib_content.as_bytes())
        .expect("failed to write VNN-LIB property");
    let property = vnnlib_file
        .path()
        .to_str()
        .expect("temp path is UTF-8")
        .to_string();

    let output = run_ny(&[
        "verify",
        &model,
        "--property",
        &property,
        "--method",
        "crown",
        "--double-fp",
        "--json",
    ]);

    let json = parse_json_output(&output, "verify --double-fp (box-unreachable property)");
    assert_eq!(
        json["status"], "verified",
        "Y_0 >= 3 is unreachable given upper bound ~1; full JSON: {json}"
    );
    assert_eq!(json["property_status"], "safe", "full JSON: {json}");
    assert_eq!(json["method"], "Crown_f64", "full JSON: {json}");
    assert_eq!(output.status.code(), Some(EXIT_VERIFIED));
}

/// `--strict` must reject the silent alpha -> fixed-slope f64 CROWN fallback.
///
/// The f64 engine has no alpha optimization: AlphaCrown requests run plain
/// f64 CROWN. Under --strict that substitution is an error, not a silent
/// downgrade.
#[ntest::timeout(60000)]
#[test]
fn double_fp_strict_rejects_alpha_fallback() {
    let model = crossing_relu_model();
    let property = crossing_relu_unsafe_property();

    let output = run_ny(&[
        "verify",
        &model,
        "--property",
        &property,
        "--method",
        "alpha",
        "--double-fp",
        "--strict",
    ]);

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "--strict must fail on the alpha -> Crown_f64 fallback; stderr: {stderr}"
    );
    assert!(
        stderr.contains("Strict mode"),
        "error must name the strict-mode fallback; stderr: {stderr}"
    );
}

/// `--strict` must NOT reject crown/ibp runs carried out in f64: those are
/// the requested method in double precision, not a fallback.
#[ntest::timeout(60000)]
#[test]
fn double_fp_strict_allows_matching_method_in_f64() {
    let model = crossing_relu_model();
    let property = crossing_relu_unsafe_property();

    for (method, expected_tag) in [("crown", "Crown_f64"), ("ibp", "Ibp_f64")] {
        let output = run_ny(&[
            "verify",
            &model,
            "--property",
            &property,
            "--method",
            method,
            "--double-fp",
            "--strict",
            "--json",
        ]);

        let json = parse_json_output(&output, "verify --double-fp --strict (matching method)");
        assert_eq!(
            json["method"], expected_tag,
            "method {method}: full JSON: {json}"
        );
        assert_eq!(
            json["status"], "unknown",
            "method {method}: full JSON: {json}"
        );
        assert_eq!(
            output.status.code(),
            Some(EXIT_UNKNOWN),
            "method {method}: {method} in f64 is not a fallback and must not trip --strict"
        );
    }
}
