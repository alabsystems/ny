// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::parse_vnnlib;

#[ntest::timeout(10000)]
#[test]
fn test_version_detection_none() {
    // No version declaration - version should be None
    let content = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0))
(assert (<= X_0 1))
"#;
    let spec = parse_vnnlib(content).unwrap();
    assert!(spec.version.is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_version_detection_v1() {
    // VNN-LIB 1.0 version declaration
    let content = r#"
(vnnlib-version 1.0)
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0))
(assert (<= X_0 1))
"#;
    let spec = parse_vnnlib(content).unwrap();
    // Version 1.0 should preserve the decimal point
    assert_eq!(spec.version, Some("1.0".to_string()));
}

#[ntest::timeout(10000)]
#[test]
fn test_version_detection_v2() {
    // VNN-LIB 2.0 version declaration - should parse but warn
    let content = r#"
(vnnlib-version 2.0)
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0))
(assert (<= X_0 1))
"#;
    let spec = parse_vnnlib(content).unwrap();
    // Version 2.0 should preserve the decimal point
    assert_eq!(spec.version, Some("2.0".to_string()));
    // Note: warning is emitted via tracing, tested via integration
}

#[ntest::timeout(10000)]
#[test]
fn test_version_detection_string_format() {
    // Version as string symbol (some files use "2.0" not just 2.0)
    let content = r#"
(vnnlib-version 2.1)
(declare-const X_0 Real)
(declare-const Y_0 Real)
(assert (>= X_0 0))
(assert (<= X_0 1))
"#;
    let spec = parse_vnnlib(content).unwrap();
    // f64 parsing of 2.1 gives "2.1"
    assert_eq!(spec.version, Some("2.1".to_string()));
}

#[ntest::timeout(10000)]
#[test]
fn test_v10_syntax_still_works() {
    // VNN-LIB 1.0 style file should still work with version declaration
    let content = r#"
(vnnlib-version 1.0)
; ACAS Xu property style
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)

(assert (<= X_0 1.0))
(assert (>= X_0 -1.0))
(assert (<= X_1 0.5))
(assert (>= X_1 -0.5))

(assert (<= Y_0 Y_1))
"#;
    let spec = parse_vnnlib(content).unwrap();
    assert_eq!(spec.num_inputs, 2);
    assert_eq!(spec.num_outputs, 2);
    assert_eq!(spec.output_constraints.len(), 1);
    assert!(spec.version.is_some());
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_syntax_without_version() {
    // VNN-LIB 2.0 syntax without version declaration should parse but warn.
    // The v2.0-specific declarations are parsed into input/output sizes.
    let content = r#"
(declare-input X Float32 [2])
(declare-output Y Float32 [2])
"#;
    // Should parse without panicking - just logs warning
    let spec = parse_vnnlib(content).unwrap();
    // No version declared
    assert!(spec.version.is_none());
    assert_eq!(spec.num_inputs, 2);
    assert_eq!(spec.num_outputs, 2);
    assert_eq!(spec.input_bounds.len(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_v20_syntax_with_version() {
    // VNN-LIB 2.0 file with version declaration - v2.0 syntax parsed
    let content = r#"
(vnnlib-version 2.0)
(declare-input X Float32 [2])
(declare-output Y Float32 [2])
"#;
    let spec = parse_vnnlib(content).unwrap();
    // Version should be recorded
    assert_eq!(spec.version, Some("2.0".to_string()));
    assert_eq!(spec.num_inputs, 2);
    assert_eq!(spec.num_outputs, 2);
}
