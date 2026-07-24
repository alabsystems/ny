// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Output index and project/Inf/NaN regression tests.

use ndarray::{ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::BoundedTensor;

use crate::pgd_attack::attacker::eval::output_value;
use crate::pgd_attack::attacker::PgdAttacker;
use crate::pgd_attack::config::PgdConfig;

/// Regression test for #2082/#2091: output_value must return InvalidSpec for
/// out-of-bounds indices. Before the fix, it returned `unwrap_or(0.0)`, silently
/// corrupting PGD gradients and violation checks.
#[test]
fn output_value_oob_returns_invalid_spec() {
    let output = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0f32, 2.0, 3.0]).unwrap();

    // In-bounds: should succeed
    assert_eq!(output_value(&output, 0).unwrap(), 1.0);
    assert_eq!(output_value(&output, 2).unwrap(), 3.0);

    // Out-of-bounds: should return InvalidSpec error
    let err = output_value(&output, 3).unwrap_err();
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );

    // Way out of bounds
    let err = output_value(&output, 100).unwrap_err();
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );

    // Empty output: any index is OOB
    let empty_output = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
    let err = output_value(&empty_output, 0).unwrap_err();
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec for empty output, got {err:?}"
    );
}

/// Regression test for #2721: project() replaces NaN with lower bound.
#[ntest::timeout(10000)]
#[test]
fn test_project_nan_replaced_with_lower_bound() {
    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 1,
        num_steps: 1,
        step_size: 0.01,
        spsa_delta: 0.001,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0_f32, -1.0, 0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0_f32, 1.0, 2.0]).unwrap(),
    )
    .unwrap();
    let x = ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, 0.5, f32::NAN]).unwrap();
    let projected = attacker.project(&x, &bounds);
    assert_eq!(projected[[0]], 0.0, "NaN → lower bound");
    assert_eq!(projected[[1]], 0.5, "normal value unchanged");
    assert_eq!(projected[[2]], 0.5, "NaN → lower bound");
    assert!(!projected.iter().any(|v| v.is_nan()), "no NaN in result");
}

/// Regression test for #2721: project() clamps Inf correctly.
#[ntest::timeout(10000)]
#[test]
fn test_project_inf_clamped() {
    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 1,
        num_steps: 1,
        step_size: 0.01,
        spsa_delta: 0.001,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0_f32, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0_f32, 1.0]).unwrap(),
    )
    .unwrap();
    let x = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, f32::NEG_INFINITY]).unwrap();
    let projected = attacker.project(&x, &bounds);
    assert_eq!(projected[[0]], 1.0, "+Inf → upper bound");
    assert_eq!(projected[[1]], -1.0, "-Inf → lower bound");
}
