// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for Whisper utility functions.

use super::eps_sweep;

// eps_sweep tests
// ============================================================

#[test]
fn test_eps_sweep_linear_basic() {
    let result = eps_sweep(0.1, 1.0, 5, true).unwrap();
    assert_eq!(result.len(), 5);
    // Linear: 0.1, 0.325, 0.55, 0.775, 1.0
    assert!((result[0] - 0.1).abs() < 1e-6);
    assert!((result[4] - 1.0).abs() < 1e-6);
    // Check linear spacing
    let diff1 = result[1] - result[0];
    let diff2 = result[2] - result[1];
    assert!((diff1 - diff2).abs() < 1e-6);
}

#[test]
fn test_eps_sweep_log_basic() {
    let result = eps_sweep(0.001, 1.0, 4, false).unwrap();
    assert_eq!(result.len(), 4);
    // Log scale: 0.001, 0.01, 0.1, 1.0 (ratio = 1000, each step = 10x)
    assert!((result[0] - 0.001).abs() < 1e-6);
    assert!((result[3] - 1.0).abs() < 1e-6);
    // Check geometric spacing (ratio between consecutive should be constant)
    let ratio1 = result[1] / result[0];
    let ratio2 = result[2] / result[1];
    assert!((ratio1 - ratio2).abs() < 0.01);
}

#[test]
fn test_eps_sweep_single_step() {
    let result = eps_sweep(0.5, 2.0, 1, true).unwrap();
    assert_eq!(result.len(), 1);
    assert!((result[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_eps_sweep_two_steps() {
    let result = eps_sweep(0.1, 0.9, 2, true).unwrap();
    assert_eq!(result.len(), 2);
    assert!((result[0] - 0.1).abs() < 1e-6);
    assert!((result[1] - 0.9).abs() < 1e-6);
}

#[test]
fn test_eps_sweep_zero_steps_error() {
    let result = eps_sweep(0.1, 1.0, 0, true);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("steps must be >= 1"));
}

#[test]
fn test_eps_sweep_negative_epsilon_error() {
    let result = eps_sweep(-0.1, 1.0, 5, true);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("must be > 0"));
}

#[test]
fn test_eps_sweep_zero_epsilon_error() {
    let result = eps_sweep(0.0, 1.0, 5, true);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("must be > 0"));
}

#[test]
fn test_eps_sweep_nan_error() {
    let result = eps_sweep(f32::NAN, 1.0, 5, true);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("must be > 0"));
}

#[test]
fn test_eps_sweep_min_greater_than_max_error() {
    let result = eps_sweep(2.0, 1.0, 5, true);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("epsilon_min must be <= epsilon_max"));
}

#[test]
fn test_eps_sweep_equal_min_max() {
    let result = eps_sweep(0.5, 0.5, 3, true).unwrap();
    assert_eq!(result.len(), 3);
    for v in &result {
        assert!((v - 0.5).abs() < 1e-6);
    }
}

// ============================================================
